//! Resident complete layer-three full-attention trial on synthetic hidden input.

use crate::{
    artifact::reader::VerifiedArtifact,
    kernels::{
        AttentionInput, EmbeddingNormInput, Fp8Projection, Nvfp4Mlp, Nvfp4Projection,
        ResidualNormWeights,
    },
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

const VOCABULARY: usize = 248_320;
const HIDDEN: usize = 5_120;
const QUERY_CHANNELS: usize = 12_288;
const KV_CHANNELS: usize = 1_024;
const HEAD_WIDTH: usize = 256;
const MLP_CHANNELS: usize = 17_408;
const ATTENTION_OUTPUT_WIDTH: usize = 6_144;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    let inventory = super::inventory::validate(artifact.directory())?;
    let identity = artifact.identity().clone();
    let input = load_input(&mut artifact)?;
    let mut report = crate::kernels::attention_check(ptx, device, &input)?;
    report["identity"] = json!(identity);
    report["compiled_inventory"] = json!(inventory);
    report["model_executable"] = json!(false);
    report["trial_scope"] = json!({
        "synthetic_hidden_input": true,
        "layers_0_to_2_executed": false,
        "layer_3_projection_and_attention_preparation": true,
        "causal_attention_core_executed": true,
        "full_attention_layer_components_executed": true,
        "full_attention_executed": true,
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

pub(super) fn load_input(artifact: &mut VerifiedArtifact) -> Result<AttentionInput> {
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
        load_fp8_projection(artifact, "q_proj", QUERY_CHANNELS, HIDDEN)?,
        load_fp8_projection(artifact, "k_proj", KV_CHANNELS, HIDDEN)?,
        load_fp8_projection(artifact, "v_proj", KV_CHANNELS, HIDDEN)?,
    ];
    let q_norm = load_bf16_weight(artifact, "q_norm")?;
    let k_norm = load_bf16_weight(artifact, "k_norm")?;
    let output_projection =
        load_fp8_projection(artifact, "o_proj", HIDDEN, ATTENTION_OUTPUT_WIDTH)?;
    let mut post_attention_weight = Vec::with_capacity(HIDDEN * 2);
    artifact.copy_object(
        "tensors/model.language_model.layers.3.post_attention_layernorm.weight",
        &mut post_attention_weight,
    )?;
    ensure!(
        post_attention_weight.len() == HIDDEN * 2,
        "layer-three post-attention norm extent mismatch"
    );
    let mlp = Nvfp4Mlp {
        gate: load_nvfp4_projection(artifact, "gate_proj", MLP_CHANNELS, HIDDEN)?,
        up: load_nvfp4_projection(artifact, "up_proj", MLP_CHANNELS, HIDDEN)?,
        down: load_nvfp4_projection(artifact, "down_proj", HIDDEN, MLP_CHANNELS)?,
    };

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
        output_projection,
        post_attention_norm: ResidualNormWeights {
            weight: post_attention_weight,
            epsilon: 1e-6,
        },
        mlp,
    };
    Ok(input)
}

fn load_fp8_projection(
    artifact: &mut VerifiedArtifact,
    name: &str,
    channels: usize,
    input_width: usize,
) -> Result<Fp8Projection> {
    let prefix = format!("tensors/model.language_model.layers.3.self_attn.{name}");
    let expected_weights = channels
        .checked_mul(input_width)
        .context("layer-three FP8 projection extent overflows usize")?;
    let mut weights = Vec::with_capacity(expected_weights);
    let mut scales = Vec::with_capacity(channels * 2);
    artifact.copy_object(&format!("{prefix}.weight"), &mut weights)?;
    artifact.copy_object(&format!("{prefix}.weight_scale"), &mut scales)?;
    ensure!(
        weights.len() == expected_weights && scales.len() == channels * 2,
        "layer-three {name} extent mismatch"
    );
    Ok(Fp8Projection {
        name: name.into(),
        weights,
        scales,
        channels,
    })
}

fn load_nvfp4_projection(
    artifact: &mut VerifiedArtifact,
    name: &str,
    channels: usize,
    input_width: usize,
) -> Result<Nvfp4Projection> {
    ensure!(
        input_width.is_multiple_of(16),
        "layer-three NVFP4 input width must be divisible by 16"
    );
    let values = channels
        .checked_mul(input_width)
        .context("layer-three NVFP4 projection extent overflows usize")?;
    let expected_packed = values / 2;
    let expected_scales = values / 16;
    let prefix = format!("tensors/model.language_model.layers.3.mlp.{name}");
    let mut packed = Vec::with_capacity(expected_packed);
    let mut scales = Vec::with_capacity(expected_scales);
    artifact.copy_object(&format!("{prefix}.weight_packed"), &mut packed)?;
    artifact.copy_object(&format!("{prefix}.weight_scale"), &mut scales)?;
    ensure!(
        packed.len() == expected_packed && scales.len() == expected_scales,
        "layer-three {name} NVFP4 extent mismatch"
    );
    Ok(Nvfp4Projection {
        name: name.into(),
        packed,
        scales,
        channels,
        input_global: load_positive_global_scale(
            artifact,
            &format!("{prefix}.input_global_scale"),
        )?,
        weight_global: load_positive_global_scale(
            artifact,
            &format!("{prefix}.weight_global_scale"),
        )?,
    })
}

fn load_positive_global_scale(artifact: &mut VerifiedArtifact, key: &str) -> Result<f32> {
    let mut bytes = Vec::with_capacity(4);
    artifact.copy_object(key, &mut bytes)?;
    let raw: [u8; 4] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("layer-three NVFP4 global scale extent mismatch: {key}"))?;
    let scale = f32::from_le_bytes(raw);
    ensure!(
        scale.is_finite() && scale > 0.0,
        "layer-three NVFP4 global scale must be finite and positive: {key}"
    );
    Ok(scale)
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
