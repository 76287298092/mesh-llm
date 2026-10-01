use super::{HashWriter, NativeModelSource, validate_source_pin};
use crate::{artifact::ninfer::Binding, packages::qwen3_8_27b::native_mtp_views::NativeMtpViews};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Write},
};

pub(super) struct VerifiedMtp {
    views: NativeMtpViews,
    parents: BTreeMap<String, NativeMtpParent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NativeMtpParent {
    object_id: String,
    bytes: u64,
    sha256: String,
}

pub(super) fn resolve(source: &mut NativeModelSource) -> Result<VerifiedMtp> {
    validate_source_pin(source.source.file_bytes(), &source.source.file_sha256()?)?;
    let views = NativeMtpViews::resolve(&mut source.source)?;
    let mut parents = BTreeMap::new();
    for id in selected_parent_ids(&views, source.source.directory())? {
        let (bytes, is_tensor) = source
            .source
            .directory()
            .objects
            .iter()
            .find(|record| record.id == id)
            .map(|record| (record.bytes, record.kind == "tensor"))
            .with_context(|| format!("native MTP parent not found: {id}"))?;
        ensure!(is_tensor, "native MTP parent is not a tensor");
        parents.insert(
            id.clone(),
            NativeMtpParent {
                object_id: id.clone(),
                bytes,
                sha256: hash_parent(&mut source.source, &id, bytes)?,
            },
        );
    }
    validate_source_pin(source.source.file_bytes(), &source.source.file_sha256()?)?;
    Ok(VerifiedMtp { views, parents })
}

impl NativeModelSource {
    pub(crate) fn native_mtp_parents(&mut self) -> Result<impl Iterator<Item = &NativeMtpParent>> {
        self.native_mtp_views()?;
        Ok(self
            .mtp
            .as_ref()
            .context("native MTP parents are unavailable")?
            .parents
            .values())
    }

    /// Resolve checked native MTP metadata against this source and retain its
    /// verified physical-parent hashes.
    ///
    /// # Errors
    /// Returns an error if source pinning, checked metadata, or any selected
    /// physical-parent read fails. Failures dirty this source.
    pub fn native_mtp_views(&mut self) -> Result<&NativeMtpViews> {
        ensure!(!self.dirty, "native source is dirty after a failed copy");
        if self.mtp.is_none() {
            match resolve(self) {
                Ok(verified) => self.mtp = Some(verified),
                Err(error) => {
                    self.dirty = true;
                    return Err(error);
                }
            }
        }
        Ok(&self
            .mtp
            .as_ref()
            .context("native MTP views are unavailable")?
            .views)
    }

    /// Copy a complete selected physical parent, preserving its packed planes.
    ///
    /// # Errors
    /// Returns an error if checked MTP views have not been resolved, the object is
    /// not a selected parent, or the copy fails its saved length/hash. Any error
    /// dirties this source; discard destination bytes and do not execute them.
    pub fn copy_native_mtp_parent(
        &mut self,
        object_id: &str,
        destination: &mut impl Write,
    ) -> Result<u64> {
        ensure!(!self.dirty, "native source is dirty after a failed copy");
        let result = match &self.mtp {
            Some(verified) => verified.copy_parent(&mut self.source, object_id, destination),
            None => Err(anyhow::anyhow!(
                "native MTP views must be resolved before copying a parent"
            )),
        };
        if result.is_err() {
            self.dirty = true;
        }
        result
    }
}

impl VerifiedMtp {
    pub(super) fn copy_parent(
        &self,
        source: &mut crate::artifact::ninfer::NinferArtifact,
        object_id: &str,
        destination: &mut impl Write,
    ) -> Result<u64> {
        let parent = self
            .parents
            .get(object_id)
            .context("native MTP physical parent was not selected")?;
        parent.copy_from(source, destination)
    }
}

impl NativeMtpParent {
    pub(crate) fn object_id(&self) -> &str {
        &self.object_id
    }

    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    fn copy_from(
        &self,
        source: &mut crate::artifact::ninfer::NinferArtifact,
        destination: &mut impl Write,
    ) -> Result<u64> {
        let mut checked = HashWriter::new(destination);
        let copied = source.read_object_range(&self.object_id, 0, self.bytes, &mut checked)?;
        ensure!(copied == self.bytes, "short native MTP parent copy");
        checked.finish(self.bytes, Some(&self.sha256))?;
        Ok(self.bytes)
    }
}

pub(super) fn selected_parent_ids(
    views: &NativeMtpViews,
    directory: &crate::artifact::ninfer::Directory,
) -> Result<BTreeSet<String>> {
    let mut ids = BTreeSet::new();
    for view in [
        &views.fc,
        &views.query_gate,
        &views.key,
        &views.value,
        &views.attention_output,
        &views.mlp_gate,
        &views.mlp_up,
        &views.mlp_down,
    ] {
        ids.insert(view.object_id.clone());
    }
    ids.extend([
        views.norms.embedding.object_id.clone(),
        views.norms.hidden.object_id.clone(),
        views.norms.final_norm.object_id.clone(),
        views.norms.input.object_id.clone(),
        views.norms.post_attention.object_id.clone(),
        views.norms.query.object_id.clone(),
        views.norms.key.object_id.clone(),
    ]);
    ids.insert(views.proposal_head.object_id.clone());
    let proposal_map = directory
        .bindings
        .get("proposal/token_ids")
        .context("native proposal token-map binding is missing")?;
    match proposal_map {
        Binding::Object { object } => {
            ids.insert(object.clone());
        }
        Binding::Parts { .. } => anyhow::bail!("native proposal token map must bind one object"),
    }
    Ok(ids)
}

fn hash_parent(
    source: &mut crate::artifact::ninfer::NinferArtifact,
    object_id: &str,
    bytes: u64,
) -> Result<String> {
    let mut sink = io::sink();
    let mut checked = HashWriter::new(&mut sink);
    let copied = source.read_object_range(object_id, 0, bytes, &mut checked)?;
    ensure!(copied == bytes, "short native MTP parent read");
    checked.finish(bytes, None)
}

#[cfg(test)]
mod tests {
    use super::{NativeMtpParent, hash_parent};
    use crate::artifact::ninfer::NinferArtifact;
    use sha2::{Digest, Sha256};
    use std::io::{Seek, SeekFrom, Write};

    fn tiny_parent() -> (tempfile::NamedTempFile, NinferArtifact) {
        let directory = serde_json::json!({
            "components": {"text": {"config": {}}},
            "objects": [{"id": "parent", "kind": "tensor", "format": "bf16", "layout": "contiguous_le_v1", "shape": [2], "offset": 0, "bytes": 4}],
            "bindings": {},
            "uses": [],
            "files": [{"path": null, "payload_bytes": 4}],
            "metadata": {},
            "provenance": {}
        });
        let bytes = serde_json::to_vec(&directory).unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"NINFER\0\x03").unwrap();
        file.write_all(&u64::try_from(bytes.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(&[0; 16]).unwrap();
        file.write_all(&bytes).unwrap();
        file.write_all(&vec![0; 4096 - 32 - bytes.len()]).unwrap();
        file.write_all(&[11, 12, 13, 14]).unwrap();
        file.flush().unwrap();
        let source = NinferArtifact::open(file.path()).unwrap();
        (file, source)
    }

    #[test]
    fn physical_parent_copy_preserves_packed_bytes_and_verifies_digest() {
        let (_file, mut source) = tiny_parent();
        let expected = hash_parent(&mut source, "parent", 4).unwrap();
        let parent = NativeMtpParent {
            object_id: "parent".into(),
            bytes: 4,
            sha256: expected.clone(),
        };
        let mut copied = Vec::new();
        assert_eq!(parent.copy_from(&mut source, &mut copied).unwrap(), 4);
        assert_eq!(copied, [11, 12, 13, 14]);
        assert_eq!(expected, hex::encode(Sha256::digest(&copied)));
    }

    #[test]
    fn physical_parent_copy_rejects_same_length_source_edits() {
        let (mut file, mut source) = tiny_parent();
        let parent = NativeMtpParent {
            object_id: "parent".into(),
            bytes: 4,
            sha256: hash_parent(&mut source, "parent", 4).unwrap(),
        };
        file.seek(SeekFrom::Start(4097)).unwrap();
        file.write_all(&[99]).unwrap();
        file.flush().unwrap();
        let mut copied = Vec::new();
        assert!(parent.copy_from(&mut source, &mut copied).is_err());
        assert_eq!(copied, [11, 99, 13, 14]);
    }

    #[test]
    fn parent_copy_fails_when_destination_rejects_bytes() {
        let (_file, mut source) = tiny_parent();
        let parent = NativeMtpParent {
            object_id: "parent".into(),
            bytes: 4,
            sha256: hash_parent(&mut source, "parent", 4).unwrap(),
        };
        let mut destination = std::io::Cursor::new([0_u8; 2]);

        let result = parent.copy_from(&mut source, &mut destination);

        assert!(result.is_err());
    }
}
