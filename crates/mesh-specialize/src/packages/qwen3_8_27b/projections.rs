//! Layer-zero FP8 QKV/Z and BF16 A/B projections from resident normalized activations.
use crate::{
    artifact::reader::VerifiedArtifact,
    kernels::{
        Bf16Projection, CausalConv4Weights, EmbeddingNormInput, Fp8Projection, ProjectionInput,
    },
};
use anyhow::Result;
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
    let entry = EmbeddingNormInput {
        table,
        weight,
        width: 5120,
        epsilon: 1e-6,
        batches: vec![vec![248044], (0..17).map(|i| (i * 7919) % 248320).collect()],
    };
    let mut projections = Vec::new();
    for (name, channels) in [("in_proj_qkv", 10240), ("in_proj_z", 6144)] {
        let prefix = format!("tensors/model.language_model.layers.0.linear_attn.{name}");
        let mut weights = Vec::with_capacity(channels * 5120);
        artifact.copy_object(&format!("{prefix}.weight"), &mut weights)?;
        let mut scales = Vec::new();
        artifact.copy_object(&format!("{prefix}.weight_scale"), &mut scales)?;
        projections.push(Fp8Projection {
            name: name.into(),
            weights,
            scales,
            channels,
        });
    }
    let mut bf16_projections = Vec::new();
    for name in ["in_proj_a", "in_proj_b"] {
        let mut weights = Vec::with_capacity(48 * 5120 * 2);
        artifact.copy_object(
            &format!("tensors/model.language_model.layers.0.linear_attn.{name}.weight"),
            &mut weights,
        )?;
        bf16_projections.push(Bf16Projection {
            name: name.into(),
            weights,
            channels: 48,
        });
    }
    let mut conv_weights = Vec::new();
    artifact.copy_object(
        "tensors/model.language_model.layers.0.linear_attn.conv1d.weight",
        &mut conv_weights,
    )?;
    let input = ProjectionInput {
        entry,
        projections,
        bf16_projections,
        convolution: Some(CausalConv4Weights {
            projection: 0,
            weights: conv_weights,
        }),
    };
    let mut report = crate::kernels::projection_check(ptx, device, &input)?;
    report["identity"] = json!(identity);
    report["compiled_inventory"] = json!(inventory);
    report["model_executable"] = json!(false);
    for key in [
        "model_prefill_tokens_per_second",
        "model_decode_tokens_per_second",
        "model_context_tokens",
    ] {
        report[key] = Value::Null;
    }
    Ok(report)
}
