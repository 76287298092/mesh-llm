//! Direct pinned NInfer source with verified canonical runtime views.
//! No intermediate artifact is written, and no executable object claims a
//! physical source offset. Copies always use the original open descriptor.
use super::native_views::{self, Transform, View};
use crate::artifact::{
    ninfer::NinferArtifact,
    schema::{DType, Directory, Object, ObjectKind, SourceCheckpoint, align_up},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

pub(crate) const SOURCE_REPOSITORY: &str = "Neroued/ninfer:artifact";
const RECIPE_NAME: &str = "native-view-recipe.json";
const MAX_TRANSFORM_BYTES: u64 = 128 * 1024 * 1024;

mod mtp;
pub mod mtp_qualification;

/// Pinned model bytes and a virtual logical inventory. Opening streams two full
/// file hashes plus all canonical view bytes; it does not allocate a full model.
pub struct NativeModelSource {
    source: NinferArtifact,
    directory: Directory,
    views: Vec<View>,
    recipe: Vec<u8>,
    mtp: Option<mtp::VerifiedMtp>,
    dirty: bool,
    verification_seconds: f64,
}

impl NativeModelSource {
    pub fn open(path: &Path) -> Result<Self> {
        let started = Instant::now();
        let mut source = NinferArtifact::open(path)?;
        ensure!(
            source.file_bytes() == native_views::SOURCE_BYTES,
            "native source byte count differs from pinned artifact"
        );
        validate_source_pin(source.file_bytes(), &source.file_sha256()?)?;
        let raw: serde_json::Value = serde_json::from_slice(source.directory_json())?;
        let views = native_views::plan(&raw)?;
        let recipe = recipe_bytes(source.directory_json())?;
        let directory = canonical_directory(&mut source, &views, &recipe)?;
        super::inventory::validate(&directory)?;
        // Catch in-place changes while view hashes and permutations were computed.
        validate_source_pin(source.file_bytes(), &source.file_sha256()?)?;
        Ok(Self {
            source,
            directory,
            views,
            recipe,
            mtp: None,
            dirty: false,
            verification_seconds: started.elapsed().as_secs_f64(),
        })
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    pub fn verification_report(&self) -> serde_json::Value {
        serde_json::json!({
            "container": "ninfer-single-file-v3",
            "source_sha256": native_views::SOURCE_SHA256,
            "source_bytes": self.source.file_bytes(),
            "source_artifact_id": hex::encode(self.source.artifact_id()),
            "canonical_tensors": self.views.len(),
            "canonical_tensor_bytes": native_views::TEXT_BYTES,
            "startup_verification_seconds": self.verification_seconds,
            "whole_file_hash_passes": 2,
            "stream_buffer_bytes": 65_536,
            "transform_allocation_limit_bytes": MAX_TRANSFORM_BYTES,
            "transform_total_buffer_limit_bytes": 2 * MAX_TRANSFORM_BYTES,
            "virtual_offsets": true,
            "source_dirty": self.dirty,
            "full_ninfer_arithmetic_parity": false,
        })
    }

    /// Rehash canonical output during transfer. Error means that any destination
    /// bytes are invalid and must be discarded without execution. Further copies
    /// fail closed, even when the original error was a destination-write failure.
    pub fn copy_object(&mut self, name: &str, destination: &mut impl Write) -> Result<u64> {
        ensure!(!self.dirty, "native source is dirty after a failed copy");
        let result = self.copy_verified(name, destination);
        if result.is_err() {
            self.dirty = true;
        }
        result
    }

    fn copy_verified(&mut self, name: &str, destination: &mut impl Write) -> Result<u64> {
        let index = self
            .directory
            .objects
            .binary_search_by(|o| o.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("native canonical object not found: {name}"))?;
        let object = &self.directory.objects[index];
        if name == RECIPE_NAME {
            let mut checked = HashWriter::new(destination);
            checked.write_all(&self.recipe)?;
            checked.finish(object.length, Some(&object.sha256))?;
        } else {
            let index = self
                .views
                .binary_search_by(|v| v.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("native canonical view missing: {name}"))?;
            copy_view(
                &mut self.source,
                &self.views[index],
                destination,
                Some(&object.sha256),
            )?;
        }
        Ok(object.length)
    }
}

fn validate_source_pin(bytes: u64, digest: &str) -> Result<()> {
    ensure!(
        bytes == native_views::SOURCE_BYTES,
        "native source byte count differs from pinned artifact"
    );
    ensure!(
        digest == native_views::SOURCE_SHA256,
        "native source SHA256 differs from pinned artifact"
    );
    Ok(())
}

fn recipe_bytes(raw_directory: &[u8]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&serde_json::json!({
        "schema_version": 1,
        "mapping_contract": native_views::MODEL_ID,
        "tensor_abi": native_views::LAYOUT,
        "source_sha256": native_views::SOURCE_SHA256,
        "source_bytes": native_views::SOURCE_BYTES,
        "raw_directory_json": std::str::from_utf8(raw_directory)?,
        "transform_contract": "copy; fused-row-split; conv-channel-tap; nvfp4-scale-inverse-v1",
        "target": "text-only; no MTP; no synthetic KV scales",
    }))?)
}

fn canonical_directory(
    source: &mut impl RangeSource,
    views: &[View],
    recipe: &[u8],
) -> Result<Directory> {
    let recipe_sha256 = hex::encode(Sha256::digest(recipe));
    let mut objects = Vec::with_capacity(views.len() + 1);
    objects.push(Object {
        name: RECIPE_NAME.into(),
        kind: ObjectKind::Recipe,
        dtype: DType::U8,
        shape: vec![recipe.len() as u64],
        layout: "raw-v1".into(),
        offset: 0,
        length: recipe.len() as u64,
        sha256: recipe_sha256.clone(),
    });
    for view in views {
        let digest = copy_view(source, view, &mut io::sink(), None)?;
        objects.push(view.object(0, digest));
    }
    objects.sort_by(|a, b| a.name.cmp(&b.name));
    let mut end = 0;
    for object in &mut objects {
        object.offset = align_up(end)?;
        end = object
            .offset
            .checked_add(object.length)
            .context("virtual object extent overflow")?;
    }
    let directory = Directory {
        schema_version: 1,
        identity: ModelIdentity {
            model_id: native_views::MODEL_ID.into(),
            weights_id: format!("sha256:{}", native_views::SOURCE_SHA256),
        },
        recipe_sha256,
        source: SourceCheckpoint {
            repository: SOURCE_REPOSITORY.into(),
            revision: native_views::SOURCE_SHA256.into(),
        },
        objects,
    };
    // Placement validation applies to the virtual inventory only. Its identity
    // intentionally binds the whole native source, not mspec identity::calculate.
    directory.validate(end)?;
    Ok(directory)
}

trait RangeSource {
    fn read_range(
        &mut self,
        id: &str,
        offset: u64,
        length: u64,
        destination: &mut impl Write,
    ) -> Result<u64>;
}

impl RangeSource for NinferArtifact {
    fn read_range(
        &mut self,
        id: &str,
        offset: u64,
        length: u64,
        destination: &mut impl Write,
    ) -> Result<u64> {
        self.read_object_range(id, offset, length, destination)
    }
}

fn copy_view(
    source: &mut impl RangeSource,
    view: &View,
    destination: &mut impl Write,
    expected: Option<&str>,
) -> Result<String> {
    let mut checked = HashWriter::new(destination);
    if matches!(view.transform, Transform::Copy) {
        ensure!(
            view.source_bytes == view.bytes,
            "native copy view extent mismatch"
        );
        let copied =
            source.read_range(&view.storage, view.offset, view.source_bytes, &mut checked)?;
        ensure!(
            copied == view.source_bytes,
            "native copy source returned a short range"
        );
    } else {
        ensure!(
            view.source_bytes <= MAX_TRANSFORM_BYTES && view.bytes <= MAX_TRANSFORM_BYTES,
            "native transform exceeds bounded allocation limit"
        );
        let mut input = Vec::with_capacity(usize::try_from(view.source_bytes)?);
        let copied =
            source.read_range(&view.storage, view.offset, view.source_bytes, &mut input)?;
        ensure!(
            copied == view.source_bytes,
            "native transform source returned a short range"
        );
        let output = view.materialize(&input)?;
        checked.write_all(&output)?;
    }
    checked.finish(view.bytes, expected)
}

struct HashWriter<'a, W> {
    destination: &'a mut W,
    digest: Sha256,
    bytes: u64,
}

impl<'a, W: Write> HashWriter<'a, W> {
    fn new(destination: &'a mut W) -> Self {
        Self {
            destination,
            digest: Sha256::new(),
            bytes: 0,
        }
    }

    fn finish(self, length: u64, expected: Option<&str>) -> Result<String> {
        ensure!(self.bytes == length, "native canonical byte count mismatch");
        let digest = hex::encode(self.digest.finalize());
        if let Some(expected) = expected {
            ensure!(
                digest == expected,
                "native canonical checksum mismatch; discard partial weights"
            );
        }
        Ok(digest)
    }
}

impl<W: Write> Write for HashWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.destination.write(bytes)?;
        self.digest.update(&bytes[..written]);
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("native hash byte count overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.destination.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSource(Vec<u8>);
    impl RangeSource for FakeSource {
        fn read_range(
            &mut self,
            id: &str,
            offset: u64,
            length: u64,
            destination: &mut impl Write,
        ) -> Result<u64> {
            ensure!(id == "storage", "unknown fake storage");
            let end = offset.checked_add(length).context("fake range overflow")?;
            let bytes = self
                .0
                .get(usize::try_from(offset)?..usize::try_from(end)?)
                .context("fake range out of bounds")?;
            for chunk in bytes.chunks(3) {
                destination.write_all(chunk)?;
            }
            Ok(length)
        }
    }

    fn view(transform: Transform) -> View {
        View {
            name: "tensors/a".into(),
            dtype: DType::U8,
            shape: vec![6],
            storage: "storage".into(),
            offset: 1,
            source_bytes: 6,
            bytes: 6,
            transform,
        }
    }

    #[test]
    fn copy_streams_checked_subrange_and_rejects_changed_digest() {
        let mut source = FakeSource(vec![9, 1, 2, 3, 4, 5, 6, 9]);
        let mut bytes = Vec::new();
        let view = view(Transform::Copy);
        let digest = copy_view(&mut source, &view, &mut bytes, None).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4, 5, 6]);
        assert_eq!(digest, hex::encode(Sha256::digest(&bytes)));
        source.0[3] ^= 1;
        assert!(copy_view(&mut source, &view, &mut io::sink(), Some(&digest)).is_err());
    }

    #[test]
    fn transformed_hash_is_canonical_not_source_hash() {
        let mut source = FakeSource(vec![9, 1, 2, 3, 4, 5, 6, 9]);
        let view = view(Transform::Rows {
            row_bytes: 2,
            order: vec![2, 0, 1],
        });
        let mut bytes = Vec::new();
        let digest = copy_view(&mut source, &view, &mut bytes, None).unwrap();
        assert_eq!(bytes, [5, 6, 1, 2, 3, 4]);
        assert_eq!(digest, hex::encode(Sha256::digest(&bytes)));
        assert_ne!(digest, hex::encode(Sha256::digest(&source.0[1..7])));
        copy_view(&mut source, &view, &mut io::sink(), Some(&digest)).unwrap();
    }

    #[test]
    fn checked_bounds_and_allocation_limits_reject_before_reading() {
        let mut source = FakeSource(vec![0; 8]);
        let mut v = view(Transform::Copy);
        v.offset = u64::MAX;
        assert!(copy_view(&mut source, &v, &mut io::sink(), None).is_err());
        let mut v = view(Transform::Conv { channels: 1 });
        v.source_bytes = MAX_TRANSFORM_BYTES + 1;
        assert!(copy_view(&mut source, &v, &mut io::sink(), None).is_err());
    }

    #[test]
    fn virtual_directory_binds_real_recipe_and_tensor_hashes() {
        let mut source = FakeSource(vec![9, 1, 2, 3, 4, 5, 6, 9]);
        let recipe = recipe_bytes(b"{ \"x\": 1 }").unwrap();
        let directory =
            canonical_directory(&mut source, &[view(Transform::Copy)], &recipe).unwrap();
        assert_eq!(
            directory.recipe_sha256,
            hex::encode(Sha256::digest(&recipe))
        );
        assert_eq!(
            directory.objects[1].sha256,
            hex::encode(Sha256::digest([1, 2, 3, 4, 5, 6]))
        );
        assert_ne!(directory.objects[1].offset, 1);
        let decoded: serde_json::Value = serde_json::from_slice(&recipe).unwrap();
        assert_eq!(decoded["raw_directory_json"], "{ \"x\": 1 }");
    }

    // Deliberately bypass model admission for a tiny copy/lifetime test only.
    // The public open below must reject this otherwise structurally valid file.
    fn tiny_source() -> (tempfile::NamedTempFile, NativeModelSource) {
        let raw = serde_json::to_vec(&serde_json::json!({
            "components":{"text":{"config":{}}},
            "objects":[{"id":"storage", "kind":"resource", "encoding":"raw_bytes_v1", "offset":0, "bytes":8}],
            "bindings":{}, "uses":[], "files":[{"path":null, "payload_bytes":8}],
            "metadata":{}, "provenance":{}
        })).unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"NINFER\0\x03").unwrap();
        file.write_all(&(raw.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&[0; 16]).unwrap();
        file.write_all(&raw).unwrap();
        file.write_all(&vec![0; 4096 - 32 - raw.len()]).unwrap();
        file.write_all(&[9, 1, 2, 3, 4, 5, 6, 9]).unwrap();
        file.flush().unwrap();
        let mut source = NinferArtifact::open(file.path()).unwrap();
        let recipe = recipe_bytes(&raw).unwrap();
        let views = vec![view(Transform::Copy)];
        let directory = canonical_directory(&mut source, &views, &recipe).unwrap();
        (
            file,
            NativeModelSource {
                source,
                directory,
                views,
                recipe,
                mtp: None,
                dirty: false,
                verification_seconds: 0.0,
            },
        )
    }

    #[test]
    fn native_profile_rejects_mtp_and_legacy_cpu_references() {
        let (_file, source) = tiny_source();
        let artifact = crate::artifact::model_source::ModelArtifact::Ninfer(Box::new(source));
        assert!(
            artifact
                .require_mtp()
                .unwrap_err()
                .to_string()
                .contains("Q8/Q4")
        );
        assert!(
            artifact
                .require_legacy_reference()
                .unwrap_err()
                .to_string()
                .contains("BF16")
        );
    }

    #[test]
    fn changed_source_poisoning_prevents_subsequent_copies() {
        use std::io::{Seek, SeekFrom};
        let (mut file, mut source) = tiny_source();
        assert!(NativeModelSource::open(file.path()).is_err());
        file.seek(SeekFrom::Start(4098)).unwrap();
        file.write_all(&[42]).unwrap();
        file.flush().unwrap();
        assert!(source.copy_object("tensors/a", &mut Vec::new()).is_err());
        assert!(source.dirty);
        // Even unrelated cached recipe bytes cannot be returned after failure.
        let mut untouched = Vec::new();
        assert!(source.copy_object(RECIPE_NAME, &mut untouched).is_err());
        assert!(untouched.is_empty());
    }

    #[test]
    fn destination_failure_poisoning_prevents_execution_retry() {
        struct FailedWriter;
        impl Write for FailedWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("failed destination"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let (_file, mut source) = tiny_source();
        assert!(source.copy_object("tensors/a", &mut FailedWriter).is_err());
        assert!(source.dirty);
    }

    #[test]
    fn native_mtp_parent_copy_before_resolution_dirties_source() {
        let (_file, mut source) = tiny_source();
        assert!(
            source
                .copy_native_mtp_parent("physical-parent", &mut Vec::new())
                .is_err()
        );
        assert!(source.dirty);
        assert!(source.copy_object(RECIPE_NAME, &mut Vec::new()).is_err());
    }

    #[test]
    fn failed_native_mtp_resolution_latches_source_dirty() {
        let (_file, mut source) = tiny_source();
        assert!(source.native_mtp_views().is_err());
        assert!(source.dirty);
        assert!(source.copy_object(RECIPE_NAME, &mut Vec::new()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn path_replacement_does_not_reopen_source() {
        let (file, mut source) = tiny_source();
        let mut replacement =
            tempfile::NamedTempFile::new_in(file.path().parent().unwrap()).unwrap();
        replacement.write_all(b"unrelated replacement").unwrap();
        replacement.persist(file.path()).unwrap();
        let mut output = Vec::new();
        source.copy_object("tensors/a", &mut output).unwrap();
        assert_eq!(output, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn source_pin_requires_both_exact_hash_and_length() {
        validate_source_pin(native_views::SOURCE_BYTES, native_views::SOURCE_SHA256).unwrap();
        assert!(
            validate_source_pin(native_views::SOURCE_BYTES - 1, native_views::SOURCE_SHA256)
                .is_err()
        );
        assert!(validate_source_pin(native_views::SOURCE_BYTES, &"0".repeat(64)).is_err());
    }
}
