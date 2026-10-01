//! Host-only qualification of verified native MTP physical-parent transfers.

use super::{HashWriter, NativeModelSource};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{io, path::Path};

/// Hash every distinct selected physical parent through the retained descriptor.
///
/// # Errors
/// Returns an error on source pin, metadata, parent length, checksum, or copy
/// failure. A failed native transfer dirties the source and yields no success report.
pub fn qualify(path: &Path) -> Result<Value> {
    let mut source = NativeModelSource::open(path)?;
    let views = source.native_mtp_views()?.clone();
    let parents = source.native_mtp_parents()?.cloned().collect::<Vec<_>>();
    let mut hashes = Vec::with_capacity(parents.len());
    let mut total_bytes = 0_u64;
    for parent in parents {
        let object = source
            .source
            .directory()
            .objects
            .iter()
            .find(|object| object.id == parent.object_id())
            .cloned()
            .context("native MTP physical parent metadata missing")?;
        let mut sink = io::sink();
        let mut checked = HashWriter::new(&mut sink);
        let copied = source.copy_native_mtp_parent(parent.object_id(), &mut checked)?;
        ensure!(
            copied == parent.bytes(),
            "native MTP qualification length mismatch"
        );
        let sha256 = checked.finish(parent.bytes(), Some(parent.sha256()))?;
        total_bytes = total_bytes
            .checked_add(copied)
            .context("native MTP qualification total byte count overflow")?;
        hashes.push(json!({
            "physical_object_id": parent.object_id(),
            "format": object.format,
            "layout": object.layout,
            "shape": object.shape,
            "expected_bytes": parent.bytes(),
            "copied_bytes": copied,
            "sha256": sha256,
        }));
    }
    Ok(json!({
        "schema_version": 1,
        "kind": "native-mtp-physical-parent-qualification",
        "all_passed": true,
        "host_only": true,
        "native_mtp_admitted": false,
        "model_executable": false,
        "gpu_execution": false,
        "dense_dequantization": false,
        "text_tensors_loaded": false,
        "source_verification": source.verification_report(),
        "physical_parent_count": hashes.len(),
        "physical_parent_bytes": total_bytes,
        "logical_q8_views": 8,
        "logical_norm_views": 7,
        "proposal_rows": views.proposal_tokens.len(),
        "parent_hashes": hashes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_rejects_unpinned_source() {
        use std::io::Write;

        let raw = serde_json::to_vec(&json!({
            "components": {"text": {"config": {}}},
            "objects": [{"id": "parent", "kind": "tensor", "format": "bf16",
                "layout": "contiguous_le_v1", "shape": [2], "offset": 0, "bytes": 4}],
            "bindings": {}, "uses": [],
            "files": [{"path": null, "payload_bytes": 4}],
            "metadata": {}, "provenance": {},
        }))
        .unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"NINFER\0\x03").unwrap();
        file.write_all(&u64::try_from(raw.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(&[0; 16]).unwrap();
        file.write_all(&raw).unwrap();
        file.write_all(&vec![0; 4096 - 32 - raw.len()]).unwrap();
        file.write_all(&[11, 12, 13, 14]).unwrap();
        file.flush().unwrap();

        let result = qualify(file.path());

        assert!(result.unwrap_err().to_string().contains("pinned artifact"));
    }
}
