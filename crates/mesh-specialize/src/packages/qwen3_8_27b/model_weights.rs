//! Bounded layer-at-a-time checkpoint loading for the independent model oracle.
use crate::{
    artifact::reader::VerifiedArtifact,
    decoder_mlp_reference as mlp, decoder_ops_reference as ops, fp8_mlp_reference,
    kernels::{Bf16Projection, Fp8Projection, Nvfp4Mlp, Nvfp4Projection},
};
use anyhow::{Result, ensure};

pub(super) fn bytes(artifact: &mut VerifiedArtifact, name: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    artifact.copy_object(name, &mut result)?;
    Ok(result)
}
pub(super) fn words(artifact: &mut VerifiedArtifact, name: &str) -> Result<Vec<u16>> {
    ops::words(&bytes(artifact, name)?)
}
pub(super) fn fp8(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    channels: usize,
) -> Result<Fp8Projection> {
    Ok(Fp8Projection {
        name: prefix.into(),
        weights: bytes(artifact, &format!("{prefix}.weight"))?,
        scales: bytes(artifact, &format!("{prefix}.weight_scale"))?,
        channels,
    })
}
pub(super) fn bf16(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    channels: usize,
) -> Result<Bf16Projection> {
    Ok(Bf16Projection {
        name: prefix.into(),
        weights: bytes(artifact, &format!("{prefix}.weight"))?,
        channels,
    })
}
fn scalar(artifact: &mut VerifiedArtifact, name: &str) -> Result<f32> {
    let data = bytes(artifact, name)?;
    let raw: [u8; 4] = data
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid scalar extent: {name}"))?;
    let result = f32::from_le_bytes(raw);
    ensure!(
        result.is_finite() && result > 0.0,
        "invalid model scale {name}"
    );
    Ok(result)
}
fn nvfp4(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    channels: usize,
) -> Result<Nvfp4Projection> {
    Ok(Nvfp4Projection {
        name: prefix.into(),
        packed: bytes(artifact, &format!("{prefix}.weight_packed"))?,
        scales: bytes(artifact, &format!("{prefix}.weight_scale"))?,
        channels,
        input_global: scalar(artifact, &format!("{prefix}.input_global_scale"))?,
        weight_global: scalar(artifact, &format!("{prefix}.weight_global_scale"))?,
    })
}
fn fp8_mlp_projection(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    channels: usize,
) -> Result<fp8_mlp_reference::Projection> {
    let value = fp8(artifact, prefix, channels)?;
    Ok(fp8_mlp_reference::Projection {
        weights: value.weights,
        scales: ops::words(&value.scales)?,
        channels,
    })
}
pub(super) fn mlp(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    index: usize,
) -> Result<mlp::Weights> {
    let gate = format!("{prefix}.mlp.gate_proj");
    let up = format!("{prefix}.mlp.up_proj");
    let down = format!("{prefix}.mlp.down_proj");
    Ok(if index < 56 {
        mlp::Weights::Nvfp4(Nvfp4Mlp {
            gate: nvfp4(artifact, &gate, 17408)?,
            up: nvfp4(artifact, &up, 17408)?,
            down: nvfp4(artifact, &down, 5120)?,
        })
    } else {
        mlp::Weights::Fp8(fp8_mlp_reference::Weights {
            gate: fp8_mlp_projection(artifact, &gate, 17408)?,
            up: fp8_mlp_projection(artifact, &up, 17408)?,
            down: fp8_mlp_projection(artifact, &down, 5120)?,
        })
    })
}

pub(super) fn gdn(
    artifact: &mut VerifiedArtifact,
    index: usize,
) -> Result<crate::decoder_gdn_reference::Weights> {
    let prefix = format!("tensors/model.language_model.layers.{index}");
    let attention = format!("{prefix}.linear_attn");
    Ok(crate::decoder_gdn_reference::Weights {
        input_norm: words(artifact, &format!("{prefix}.input_layernorm.weight"))?,
        post_norm: words(
            artifact,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        qkv: fp8(artifact, &format!("{attention}.in_proj_qkv"), 10240)?,
        z: fp8(artifact, &format!("{attention}.in_proj_z"), 6144)?,
        a: bf16(artifact, &format!("{attention}.in_proj_a"), 48)?,
        b: bf16(artifact, &format!("{attention}.in_proj_b"), 48)?,
        convolution: words(artifact, &format!("{attention}.conv1d.weight"))?,
        a_log: words(artifact, &format!("{attention}.A_log"))?,
        dt_bias: words(artifact, &format!("{attention}.dt_bias"))?,
        gated_norm: words(artifact, &format!("{attention}.norm.weight"))?,
        out: fp8(artifact, &format!("{attention}.out_proj"), 5120)?,
        mlp: mlp(artifact, &prefix, index)?,
    })
}
pub(super) fn attention(
    artifact: &mut VerifiedArtifact,
    index: usize,
) -> Result<crate::decoder_attention_reference::Weights> {
    let prefix = format!("tensors/model.language_model.layers.{index}");
    let attention = format!("{prefix}.self_attn");
    Ok(crate::decoder_attention_reference::Weights {
        input_norm: words(artifact, &format!("{prefix}.input_layernorm.weight"))?,
        post_norm: words(
            artifact,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        q: fp8(artifact, &format!("{attention}.q_proj"), 12288)?,
        k: fp8(artifact, &format!("{attention}.k_proj"), 1024)?,
        v: fp8(artifact, &format!("{attention}.v_proj"), 1024)?,
        q_norm: words(artifact, &format!("{attention}.q_norm.weight"))?,
        k_norm: words(artifact, &format!("{attention}.k_norm.weight"))?,
        out: fp8(artifact, &format!("{attention}.o_proj"), 5120)?,
        mlp: mlp(artifact, &prefix, index)?,
    })
}
