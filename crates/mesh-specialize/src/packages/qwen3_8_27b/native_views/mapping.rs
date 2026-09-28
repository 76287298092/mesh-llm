//! Fixed logical mapping for the observed Qwen3.8 text component.
use super::{Builder, Transform, product};
use crate::artifact::schema::DType;
use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub(super) fn validate_config(config: &Value) -> Result<()> {
    for (key, expected) in [
        ("hidden_size", 5120),
        ("vocab_size", 248320),
        ("num_hidden_layers", 64),
        ("num_attention_heads", 24),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
        ("linear_num_key_heads", 16),
        ("linear_key_head_dim", 128),
        ("linear_num_value_heads", 48),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("intermediate_size", 17408),
    ] {
        ensure!(
            config[key].as_u64() == Some(expected),
            "unsupported native config {key}"
        );
    }
    let layers = config["layer_types"]
        .as_array()
        .context("missing native layer types")?;
    ensure!(layers.len() == 64, "native layer count mismatch");
    for (i, layer) in layers.iter().enumerate() {
        let expected = if i % 4 == 3 {
            "full_attention"
        } else {
            "linear_attention"
        };
        ensure!(layer == expected, "native layer kind mismatch at {i}");
    }
    ensure!(
        config["rms_norm_eps"]
            .as_f64()
            .is_some_and(|v| v as f32 == 1e-6_f32),
        "native norm epsilon mismatch"
    );
    ensure!(
        config["rope_parameters"]["rope_theta"].as_f64() == Some(10_000_000.0)
            && config["rope_parameters"]["partial_rotary_factor"].as_f64() == Some(0.25),
        "native RoPE mismatch"
    );
    Ok(())
}

pub(super) fn build(b: &mut Builder<'_>) -> Result<()> {
    b.fp8(
        &["text/token_embedding".into()],
        "model.language_model.embed_tokens",
        248320,
        5120,
        false,
    )?;
    b.fp8(&["text/output_head".into()], "lm_head", 248320, 5120, false)?;
    b.direct(
        "text/final_norm",
        "model.language_model.norm.weight",
        DType::Bf16,
        &[5120],
    )?;
    for i in 0..64 {
        let source = format!("text/layers/{i}");
        let dest = format!("model.language_model.layers.{i}");
        for (from, to) in [
            ("input_norm", "input_layernorm.weight"),
            ("post_attention_norm", "post_attention_layernorm.weight"),
        ] {
            b.direct(
                &format!("{source}/{from}"),
                &format!("{dest}.{to}"),
                DType::Bf16,
                &[5120],
            )?;
        }
        if i % 4 == 3 {
            attention(b, &source, &dest)?;
        } else {
            gdn(b, &source, &dest)?;
        }
        for (op, field, n, k, input) in [
            ("gate", "gate_proj", 17408, 5120, "ffn_input"),
            ("up", "up_proj", 17408, 5120, "ffn_input"),
            ("down", "down_proj", 5120, 17408, "mlp/product"),
        ] {
            let from = format!("{source}/mlp/{op}");
            let to = format!("{dest}.mlp.{field}");
            if i < 56 {
                b.nvfp4(&from, &to, n, k, &format!("{source}/{input}"))?;
            } else {
                b.fp8(&[from], &to, n, k, false)?;
            }
        }
    }
    Ok(())
}

fn gdn(b: &mut Builder<'_>, source: &str, dest: &str) -> Result<()> {
    let s = format!("{source}/gdn");
    let d = format!("{dest}.linear_attn");
    for (from, to) in [("a_log", "A_log"), ("dt_bias", "dt_bias")] {
        b.direct(
            &format!("{s}/{from}"),
            &format!("{d}.{to}"),
            DType::F32,
            &[48],
        )?;
    }
    for (from, to) in [
        ("a_projection", "in_proj_a.weight"),
        ("b_projection", "in_proj_b.weight"),
    ] {
        b.direct(
            &format!("{s}/{from}"),
            &format!("{d}.{to}"),
            DType::Bf16,
            &[48, 5120],
        )?;
    }
    b.direct(
        &format!("{s}/norm"),
        &format!("{d}.norm.weight"),
        DType::Bf16,
        &[128],
    )?;
    let conv = b.span(&format!("{s}/convolution"))?;
    ensure!(
        conv.object.format == "bf16"
            && conv.object.layout == "contiguous_le_v1"
            && conv.object.shape == [4, 10240]
            && conv.first == 0
            && conv.end == 40960,
        "native convolution geometry mismatch"
    );
    ensure!(
        conv.object.bytes == product(&conv.object.shape)? as u64 * 2,
        "native convolution size mismatch"
    );
    b.push(
        &format!("{d}.conv1d.weight"),
        DType::Bf16,
        vec![10240, 1, 4],
        &conv.object,
        0..81920,
        Transform::Conv { channels: 10240 },
    )?;
    b.fp8(
        &[
            format!("{s}/query"),
            format!("{s}/key"),
            format!("{s}/value"),
        ],
        &format!("{d}.in_proj_qkv"),
        10240,
        5120,
        false,
    )?;
    b.fp8(
        &[format!("{s}/z")],
        &format!("{d}.in_proj_z"),
        6144,
        5120,
        false,
    )?;
    b.fp8(
        &[format!("{s}/output")],
        &format!("{d}.out_proj"),
        5120,
        6144,
        false,
    )
}

fn attention(b: &mut Builder<'_>, source: &str, dest: &str) -> Result<()> {
    let s = format!("{source}/attention");
    let d = format!("{dest}.self_attn");
    for (from, to) in [
        ("query_norm", "q_norm.weight"),
        ("key_norm", "k_norm.weight"),
    ] {
        b.direct(
            &format!("{s}/{from}"),
            &format!("{d}.{to}"),
            DType::Bf16,
            &[256],
        )?;
    }
    b.fp8(
        &[format!("{s}/query"), format!("{s}/gate")],
        &format!("{d}.q_proj"),
        12288,
        5120,
        true,
    )?;
    for (from, to, n, k) in [
        ("key", "k_proj", 1024, 5120),
        ("value", "v_proj", 1024, 5120),
        ("output", "o_proj", 5120, 6144),
    ] {
        b.fp8(&[format!("{s}/{from}")], &format!("{d}.{to}"), n, k, false)?;
    }
    Ok(())
}
