//! First real-weight operation: embedding lookup and layer-zero input norm.
use crate::{artifact::reader::VerifiedArtifact, kernels::EmbeddingNormInput};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    let inventory = super::inventory::validate(artifact.directory())?;
    let identity = artifact.identity().clone();
    let mut table = Vec::with_capacity(248320 * 5120 * 2);
    artifact.copy_object(
        "tensors/model.language_model.embed_tokens.weight",
        &mut table,
    )?;
    let mut weight = Vec::new();
    artifact.copy_object(
        "tensors/model.language_model.layers.0.input_layernorm.weight",
        &mut weight,
    )?;
    ensure!(
        table.len() == 248320 * 5120 * 2 && weight.len() == 5120 * 2,
        "entry weight extent mismatch"
    );
    let input = EmbeddingNormInput {
        table,
        weight,
        width: 5120,
        epsilon: 1e-6,
        batches: vec![
            vec![248044],
            vec![0, 248319, 42, 248044, 248046, 42, 1],
            (0..128).map(|i| (i * 7919) % 248320).collect(),
        ],
    };
    let mut report = crate::kernels::embedding_norm_check(ptx, device, &input)?;
    report["identity"] = json!(identity);
    report["compiled_inventory"] = json!(inventory);
    report["model_executable"] = json!(false);
    report["model_prefill_tokens_per_second"] = Value::Null;
    report["model_decode_tokens_per_second"] = Value::Null;
    report["model_context_tokens"] = Value::Null;
    Ok(report)
}
