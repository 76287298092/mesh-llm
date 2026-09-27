use crate::{
    artifact::{
        reader::VerifiedArtifact,
        schema::{Object, ObjectKind},
    },
    engine::layout::Layout,
};
use anyhow::{Context as _, Result, anyhow, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

use super::driver::{Buffer, Context};

const VERIFY_CHUNK_BYTES: usize = 1024 * 1024;

/// Tensor objects copied into one device arena and verified by readback.
pub(super) struct ResidentWeights<'ctx> {
    arena: Buffer<'ctx>,
    layout: Layout,
    objects: Vec<Object>,
}

impl<'ctx> ResidentWeights<'ctx> {
    /// Copy verified tensor objects into a single canonical arena.
    ///
    /// The artifact reader rechecks each object's digest while streaming it into
    /// the arena. The arena remains local and is dropped if any copy or hash check
    /// fails, so no partially loaded weight set is returned.
    pub(super) fn load(
        context: &'ctx Context,
        artifact: &mut VerifiedArtifact,
        objects: &[Object],
    ) -> Result<Self> {
        validate_objects(artifact, objects)?;
        let layout = Layout::new(
            objects
                .iter()
                .map(|object| (object.name.clone(), object.length)),
        )?;
        let arena_bytes =
            usize::try_from(layout.bytes).context("weight arena size does not fit usize")?;
        let arena = Buffer::new(context, arena_bytes)?;
        for object in objects {
            copy_object_to_arena(artifact, &arena, &layout, object)?;
        }
        Ok(Self {
            arena,
            layout,
            objects: objects.to_vec(),
        })
    }

    /// Return the device pointer at the start of a named tensor region.
    pub(super) fn pointer(&self, name: &str) -> Result<u64> {
        let region = self.layout.region(name)?;
        self.arena
            .pointer()
            .checked_add(region.offset)
            .ok_or_else(|| anyhow!("resident weight device pointer offset overflows for `{name}`"))
    }

    /// Return the canonical arena layout.
    pub(super) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Read back and hash each tensor in bounded chunks.
    ///
    /// Hash mismatches are reported as `matches: false`; driver transfer errors
    /// are returned as errors. No whole-object host allocation is made.
    pub(super) fn verify(&self) -> Result<Vec<Value>> {
        let mut reports = Vec::with_capacity(self.objects.len());
        for object in &self.objects {
            reports.push(verify_object(&self.arena, &self.layout, object)?);
        }
        Ok(reports)
    }
}

fn validate_objects(artifact: &VerifiedArtifact, objects: &[Object]) -> Result<()> {
    let directory = artifact.directory();
    for object in objects {
        ensure!(
            matches!(&object.kind, ObjectKind::Tensor),
            "resident weights only accept tensor objects: {}",
            object.name
        );
        let index = directory
            .objects
            .binary_search_by(|candidate| candidate.name.as_str().cmp(&object.name))
            .map_err(|_| {
                anyhow!(
                    "resident weight object is absent from the artifact: {}",
                    object.name
                )
            })?;
        ensure!(
            &directory.objects[index] == object,
            "resident weight metadata does not match the artifact for {}",
            object.name
        );
    }
    Ok(())
}

fn copy_object_to_arena(
    artifact: &mut VerifiedArtifact,
    arena: &Buffer<'_>,
    layout: &Layout,
    object: &Object,
) -> Result<()> {
    let region = layout.region(&object.name)?;
    let offset =
        usize::try_from(region.offset).context("weight region offset does not fit usize")?;
    let length =
        usize::try_from(region.length).context("weight region length does not fit usize")?;
    let mut sink = BufferRegionSink::new(arena, offset, length);
    let copied = artifact
        .copy_object(&object.name, &mut sink)
        .with_context(|| format!("stream artifact tensor {} into device arena", object.name))?;
    ensure!(
        copied == object.length,
        "artifact tensor {} copied {copied} bytes, expected {}",
        object.name,
        object.length
    );
    sink.finish(&object.name)
}

fn verify_object(arena: &Buffer<'_>, layout: &Layout, object: &Object) -> Result<Value> {
    let region = layout.region(&object.name)?;
    let region_offset =
        usize::try_from(region.offset).context("weight region offset does not fit usize")?;
    let object_bytes =
        usize::try_from(object.length).context("weight length does not fit usize")?;
    let mut chunk = vec![0_u8; object_bytes.min(VERIFY_CHUNK_BYTES)];
    let mut position = 0_usize;
    let mut digest = Sha256::new();
    while position < object_bytes {
        let chunk_bytes = (object_bytes - position).min(VERIFY_CHUNK_BYTES);
        let offset = region_offset
            .checked_add(position)
            .context("weight readback offset overflows usize")?;
        arena
            .download_at(offset, &mut chunk[..chunk_bytes])
            .with_context(|| format!("read back resident tensor {}", object.name))?;
        digest.update(&chunk[..chunk_bytes]);
        position = position
            .checked_add(chunk_bytes)
            .context("weight readback position overflows usize")?;
    }
    let sha256 = hex::encode(digest.finalize());
    Ok(json!({
        "name": object.name,
        "bytes": object.length,
        "sha256": sha256,
        "matches": sha256 == object.sha256,
    }))
}

struct BufferRegionSink<'buffer, 'ctx> {
    arena: &'buffer Buffer<'ctx>,
    region_offset: usize,
    region_length: usize,
    written: usize,
}

impl<'buffer, 'ctx> BufferRegionSink<'buffer, 'ctx> {
    fn new(arena: &'buffer Buffer<'ctx>, region_offset: usize, region_length: usize) -> Self {
        Self {
            arena,
            region_offset,
            region_length,
            written: 0,
        }
    }

    fn finish(self, name: &str) -> Result<()> {
        ensure!(
            self.written == self.region_length,
            "artifact tensor {name} copied {} bytes, expected {}",
            self.written,
            self.region_length
        );
        Ok(())
    }
}

impl Write for BufferRegionSink<'_, '_> {
    fn write(&mut self, source: &[u8]) -> io::Result<usize> {
        if source.is_empty() {
            return Ok(0);
        }
        let end = checked_sink_end(self.region_length, self.written, source.len())?;
        let offset = self
            .region_offset
            .checked_add(self.written)
            .ok_or_else(|| sink_error("device arena offset overflows usize"))?;
        self.arena
            .upload_at(offset, source)
            .map_err(|error| io::Error::other(error.to_string()))?;
        self.written = end;
        Ok(source.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn checked_sink_end(
    region_length: usize,
    position: usize,
    write_length: usize,
) -> io::Result<usize> {
    let end = position
        .checked_add(write_length)
        .ok_or_else(|| sink_error("artifact tensor write range overflows usize"))?;
    if position > region_length || end > region_length {
        return Err(sink_error(
            "artifact tensor write exceeds its device region",
        ));
    }
    Ok(end)
}

fn sink_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::checked_sink_end;

    #[test]
    fn sink_range_accepts_exact_end_and_rejects_overrun() {
        assert_eq!(checked_sink_end(8, 4, 4).unwrap(), 8);
        assert!(checked_sink_end(8, 7, 2).is_err());
    }

    #[test]
    fn sink_range_rejects_position_overflow() {
        assert!(checked_sink_end(usize::MAX, usize::MAX, 1).is_err());
    }
}
