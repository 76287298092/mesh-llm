//! Resident layer-three full-attention projection and preparation trial.

use crate::{
    artifact::reader::VerifiedArtifact,
    kernels::{AttentionInput, EmbeddingNormInput, Fp8Projection},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

const VOCABULARY: usize = 248_320;
const HIDDEN: usize = 5_120;
const QUERY_CHANNELS: usize = 12_288;
const KV_CHANNELS: usize = 1_024;
const HEAD_WIDTH: usize = 256;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    let inventory = super::inventory::validate(artifact.directory())?;
    let identity = artifact.identity().clone();

    let mut table = Vec::with_capacity(VOCABULARY * HIDDEN * 2);
    artifact.copy_object(
        "tensors/model.language_model.embed_tokens.weight",
        &mut table,
    )?;
    let mut entry_norm = Vec::with_capacity(HIDDEN * 2);
    artifact.copy_object(
        "tensors/model.language_model.layers.3.input_layernorm.weight",
        &mut entry_norm,
    )?;
    ensure!(
        table.len() == VOCABULARY * HIDDEN * 2 && entry_norm.len() == HIDDEN * 2,
        "layer-three entry weight extent mismatch"
    );

    let entry = EmbeddingNormInput {
        table,
        weight: entry_norm,
        width: HIDDEN,
        epsilon: 1e-6,
        batches: vec![
            vec![248_044],
            (0..17).map(|i| (i * 7_919) % VOCABULARY as u32).collect(),
        ],
    };
    let projections = [
        load_fp8_projection(&mut artifact, "q_proj", QUERY_CHANNELS)?,
        load_fp8_projection(&mut artifact, "k_proj", KV_CHANNELS)?,
        load_fp8_projection(&mut artifact, "v_proj", KV_CHANNELS)?,
    ];
    let q_norm = load_bf16_weight(&mut artifact, "q_norm")?;
    let k_norm = load_bf16_weight(&mut artifact, "k_norm")?;

    let input = AttentionInput {
        entry,
        projections,
        q_norm,
        k_norm,
        query_heads: 24,
        kv_heads: 4,
        head_width: HEAD_WIDTH,
        rotary_dim: 64,
        rope_theta: 10_000_000.0,
        positions: vec![vec![0], (0..17).collect()],
    };
    let mut report = crate::kernels::attention_check(ptx, device, &input)?;
    report["identity"] = json!(identity);
    report["compiled_inventory"] = json!(inventory);
    report["model_executable"] = json!(false);
    report["trial_scope"] = json!({
        "synthetic_hidden_input": true,
        "layers_0_to_2_executed": false,
        "layer_3_projection_and_attention_preparation": true,
        "full_attention_executed": false,
        "full_model_executed": false,
    });
    for key in [
        "model_prefill_tokens_per_second",
        "model_decode_tokens_per_second",
        "model_context_tokens",
    ] {
        report[key] = Value::Null;
    }
    Ok(report)
}

fn load_fp8_projection(
    artifact: &mut VerifiedArtifact,
    name: &str,
    channels: usize,
) -> Result<Fp8Projection> {
    let prefix = format!("tensors/model.language_model.layers.3.self_attn.{name}");
    let mut weights = Vec::with_capacity(channels * HIDDEN);
    let mut scales = Vec::with_capacity(channels * 2);
    artifact.copy_object(&format!("{prefix}.weight"), &mut weights)?;
    artifact.copy_object(&format!("{prefix}.weight_scale"), &mut scales)?;
    ensure!(
        weights.len() == channels * HIDDEN && scales.len() == channels * 2,
        "layer-three {name} extent mismatch"
    );
    Ok(Fp8Projection {
        name: name.into(),
        weights,
        scales,
        channels,
    })
}

fn load_bf16_weight(artifact: &mut VerifiedArtifact, name: &str) -> Result<Vec<u8>> {
    let key = format!("tensors/model.language_model.layers.3.self_attn.{name}.weight");
    let mut weight = Vec::with_capacity(HEAD_WIDTH * 2);
    artifact.copy_object(&key, &mut weight)?;
    ensure!(
        weight.len() == HEAD_WIDTH * 2,
        "layer-three {name} extent mismatch"
    );
    Ok(weight)
}
