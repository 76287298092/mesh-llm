//! One bounded CUDA allocation for canonical persistent model state.

use super::driver::{Buffer, Context};
use crate::engine::layout::Layout;
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

const TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;

pub(super) struct ResidentState<'ctx> {
    arena: Buffer<'ctx>,
    layout: Layout,
}

impl<'ctx> ResidentState<'ctx> {
    pub(super) fn new(ctx: &'ctx Context, layout: &Layout) -> Result<Self> {
        let canonical = canonical_layout(layout)?;
        let bytes = usize::try_from(canonical.bytes)
            .context("resident-state layout size does not fit usize")?;
        let arena = Buffer::new(ctx, bytes)?;
        initialize_zero(&arena)?;
        Ok(Self {
            arena,
            layout: canonical,
        })
    }

    pub(super) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(super) fn belongs_to(&self, context: &Context) -> bool {
        self.arena.belongs_to(context)
    }

    pub(super) fn fork<'a>(&self, context: &'a Context) -> Result<ResidentState<'a>> {
        ensure!(
            self.belongs_to(context),
            "resident state belongs to another CUDA context"
        );
        let layout = canonical_layout(&self.layout)?;
        let bytes = usize::try_from(layout.bytes)
            .context("resident-state layout size does not fit usize")?;
        ensure!(
            self.arena.len() == bytes,
            "resident-state arena extent differs from its layout"
        );
        let arena = Buffer::new(context, bytes)?;
        arena.copy_from_at(0, &self.arena, 0, bytes)?;
        Ok(ResidentState { arena, layout })
    }

    pub(super) fn pointer(&self, name: &str, bytes: usize) -> Result<u64> {
        let region = self.layout.region(name)?;
        ensure!(
            region.length == u64::try_from(bytes)?,
            "state region extent mismatch: {name}"
        );
        self.arena
            .pointer()
            .checked_add(region.offset)
            .context("state pointer overflow")
    }

    pub(super) fn copy_from(&mut self, name: &str, source: &Buffer<'_>) -> Result<()> {
        self.pointer(name, source.len())?;
        self.arena.copy_from_at(
            usize::try_from(self.layout.region(name)?.offset)?,
            source,
            0,
            source.len(),
        )
    }

    pub(super) fn copy_state_range(
        &mut self,
        name: &str,
        source: &ResidentState<'_>,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        ensure!(self.layout == source.layout, "state copy layouts differ");
        let start = self.region_range(name, offset, bytes)?;
        self.arena.copy_from_at(start, &source.arena, start, bytes)
    }

    pub(super) fn copy_region_to(
        &self,
        name: &str,
        offset: usize,
        destination: &Buffer<'_>,
        destination_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let start = self.region_range(name, offset, bytes)?;
        destination.copy_from_at(destination_offset, &self.arena, start, bytes)
    }

    fn region_range(&self, name: &str, offset: usize, bytes: usize) -> Result<usize> {
        let region = self.layout.region(name)?;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= region.length as usize),
            "state copy exceeds region {name}"
        );
        usize::try_from(region.offset)?
            .checked_add(offset)
            .context("state region offset overflow")
    }

    pub(super) fn read_region(&self, name: &str, bytes: &mut [u8]) -> Result<()> {
        self.pointer(name, bytes.len())?;
        self.arena
            .download_at(usize::try_from(self.layout.region(name)?.offset)?, bytes)
    }

    pub(super) fn verify_zero(&self) -> Result<Value> {
        let bytes_checked = usize::try_from(self.layout.bytes)
            .context("resident-state layout size does not fit usize")?;
        ensure!(
            self.arena.len() == bytes_checked,
            "resident-state arena extent differs from its layout"
        );
        let mut chunk = vec![0_u8; TRANSFER_CHUNK_BYTES];
        let mut offset = 0_usize;
        let mut nonzero_bytes = 0_u64;
        while offset < bytes_checked {
            let length = (bytes_checked - offset).min(chunk.len());
            let bytes = &mut chunk[..length];
            self.arena.download_at(offset, bytes)?;
            let nonzero = u64::try_from(bytes.iter().filter(|&&byte| byte != 0).count())?;
            nonzero_bytes = nonzero_bytes
                .checked_add(nonzero)
                .context("resident-state nonzero byte count overflows u64")?;
            offset = offset
                .checked_add(length)
                .context("resident-state verification offset overflows usize")?;
        }
        Ok(json!({
            "bytes_checked": self.layout.bytes,
            "nonzero_bytes": nonzero_bytes,
            "passed": nonzero_bytes == 0,
            "regions": self.layout.regions.len(),
        }))
    }
}

fn canonical_layout(layout: &Layout) -> Result<Layout> {
    let canonical = Layout::new(
        layout
            .regions
            .iter()
            .map(|region| (region.name.clone(), region.length)),
    )?;
    ensure!(
        canonical == *layout,
        "resident-state layout is not canonical"
    );
    Ok(canonical)
}

fn initialize_zero(arena: &Buffer<'_>) -> Result<()> {
    let zeroes = vec![0_u8; TRANSFER_CHUNK_BYTES];
    let mut offset = 0_usize;
    while offset < arena.len() {
        let length = (arena.len() - offset).min(zeroes.len());
        arena.upload_at(offset, &zeroes[..length])?;
        offset = offset
            .checked_add(length)
            .context("resident-state initialization offset overflows usize")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::canonical_layout;
    use crate::engine::layout::Layout;

    #[test]
    fn canonical_layout_recomputation_rejects_forged_offsets_and_sizes() {
        let valid = Layout::new([
            ("layers.00.state".to_owned(), 128),
            ("layers.01.state".to_owned(), 512),
        ])
        .unwrap();
        assert_eq!(canonical_layout(&valid).unwrap(), valid);

        let mut bad_offset = valid.clone();
        bad_offset.regions[1].offset += 256;
        assert!(canonical_layout(&bad_offset).is_err());

        let mut bad_size = valid;
        bad_size.bytes += 256;
        assert!(canonical_layout(&bad_size).is_err());
    }
}
