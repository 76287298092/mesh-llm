//! Model-facing source dispatch. This does not change either container's framing.
use super::{reader::VerifiedArtifact, schema::Directory};
use crate::packages::qwen3_8_27b::native_source::NativeModelSource;
use anyhow::{Context, Result, bail, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

/// Verified logical tensors backed by a retained source descriptor. Native object
/// offsets describe a virtual canonical address space, never source-file offsets.
pub enum ModelArtifact {
    Mspec(Box<VerifiedArtifact>),
    Ninfer(Box<NativeModelSource>),
}

impl ModelArtifact {
    /// Select by magic, not filename. Each backend independently validates the
    /// actual descriptor it opens; no trust is derived from this sniff.
    pub fn open(path: &Path) -> Result<Self> {
        let mut magic = [0; 9];
        File::open(path)
            .context("open model source")?
            .read_exact(&mut magic)
            .context("read model source magic")?;
        if &magic == b"MESHSPEC\0" {
            Ok(Self::Mspec(Box::new(VerifiedArtifact::open(path)?)))
        } else if &magic[..7] == b"NINFER\0" {
            Ok(Self::Ninfer(Box::new(NativeModelSource::open(path)?)))
        } else {
            bail!("unsupported model source magic; expected mspec or single-file NInfer v3")
        }
    }

    pub fn open_for_identity(path: &Path, expected: &ModelIdentity) -> Result<Self> {
        expected.validate()?;
        let artifact = Self::open(path)?;
        artifact.require_identity(expected)?;
        Ok(artifact)
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.directory().identity
    }

    pub fn directory(&self) -> &Directory {
        match self {
            Self::Mspec(source) => source.directory(),
            Self::Ninfer(source) => source.directory(),
        }
    }

    pub fn require_identity(&self, expected: &ModelIdentity) -> Result<()> {
        expected.validate()?;
        ensure!(
            self.identity() == expected,
            "model source identity is not supported"
        );
        Ok(())
    }

    /// Any failed copy may leave partial bytes. Discard that destination and do
    /// not execute it. A failed native copy permanently poisons that source.
    pub fn copy_object(&mut self, name: &str, destination: &mut impl Write) -> Result<u64> {
        match self {
            Self::Mspec(source) => source.copy_object(name, destination),
            Self::Ninfer(source) => source.copy_object(name, destination),
        }
    }

    /// Startup I/O is separate from model prefill/decode timing.
    pub fn verification_report(&self) -> serde_json::Value {
        match self {
            Self::Mspec(_) => serde_json::json!({"container": "mspec"}),
            Self::Ninfer(source) => source.verification_report(),
        }
    }

    pub(crate) fn require_legacy_reference(&self) -> Result<()> {
        ensure!(
            matches!(self, Self::Mspec(_)),
            "native NInfer CPU reference is unsupported: reference assumes BF16 embedding and GDN parameters"
        );
        Ok(())
    }

    pub(crate) fn require_mtp(&self) -> Result<()> {
        ensure!(
            matches!(self, Self::Mspec(_)),
            "native NInfer MTP is unsupported until Q8/Q4 MTP consumers are implemented; target-text execution only"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mspec_dispatch_preserves_identity_inventory_and_bytes() {
        use crate::artifact::{
            schema::{DType, ObjectKind, SourceCheckpoint},
            writer::{ObjectSource, write_artifact},
        };
        let temporary = tempfile::tempdir().unwrap();
        let input = temporary.path().join("recipe.json");
        std::fs::write(&input, b"test-recipe").unwrap();
        let path = temporary.path().join("valid.ninfer");
        let sources = [ObjectSource {
            name: "recipe".into(),
            kind: ObjectKind::Recipe,
            dtype: DType::U8,
            shape: vec![11],
            layout: "raw-v1".into(),
            path: input,
            range: None,
            expected_sha256: None,
        }];
        let written = write_artifact(
            &path,
            "test:model",
            SourceCheckpoint {
                repository: "test/model".into(),
                revision: "a".repeat(40),
            },
            &sources,
        )
        .unwrap();
        let mut source =
            ModelArtifact::open_for_identity(&path, &written.directory.identity).unwrap();
        assert!(matches!(source, ModelArtifact::Mspec(_)));
        assert_eq!(source.directory(), &written.directory);
        source.require_legacy_reference().unwrap();
        source.require_mtp().unwrap();
        let mut bytes = Vec::new();
        assert_eq!(source.copy_object("recipe", &mut bytes).unwrap(), 11);
        assert_eq!(bytes, b"test-recipe");
        let mut wrong = written.directory.identity;
        wrong.model_id.push_str(":different");
        assert!(source.require_identity(&wrong).is_err());
        assert!(ModelArtifact::open_for_identity(&path, &wrong).is_err());
    }

    #[test]
    fn rejects_unknown_magic_regardless_of_extension() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fake.ninfer");
        std::fs::write(&path, b"not a container").unwrap();
        assert!(
            ModelArtifact::open(&path)
                .err()
                .unwrap()
                .to_string()
                .contains("magic")
        );
    }

    #[test]
    fn native_magic_uses_native_framing_even_with_mspec_extension() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fake.mspec");
        let mut header = [0; 32];
        header[..8].copy_from_slice(b"NINFER\0\x02");
        std::fs::write(&path, header).unwrap();
        let error = ModelArtifact::open(&path).err().unwrap().to_string();
        assert!(error.contains("unsupported NInfer version"), "{error}");
    }
}
