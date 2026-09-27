//! Independent CPU composition of the pinned Qwen GDN layer-zero path.

use anyhow::{Context, Result, ensure};

use crate::{
    causal_conv4_reference, entry_reference, gated_norm_reference, gdn_prepare_reference,
    gdn_recurrent_reference, kernels, mlp_activation_reference, nvfp4_linear_reference,
    nvfp4_quantize_reference, projection_reference, residual_add_reference,
    residual_norm_reference,
};

#[derive(Debug, PartialEq)]
pub struct Layer {
    pub output: Vec<u16>,
    pub convolution_history: Vec<u16>,
    pub recurrent_state: Vec<f32>,
}

struct Components<'a> {
    convolution: &'a kernels::CausalConv4Weights,
    gdn: &'a kernels::GdnWeights,
    gdn_output: &'a kernels::GdnOutputWeights,
    post_attention_norm: &'a kernels::ResidualNormWeights,
    mlp: &'a kernels::Nvfp4Mlp,
    qkv: &'a kernels::Fp8Projection,
    z: &'a kernels::Fp8Projection,
    a: &'a kernels::Bf16Projection,
    b: &'a kernels::Bf16Projection,
    inner: usize,
}

/// Run the logical layer from tokens through post-MLP residual addition.
///
/// This composition consumes only checkpoint/input data and scalar references.
/// It does not read or substitute any device output.
pub fn run(input: &kernels::ProjectionInput, tokens: &[u32]) -> Result<Layer> {
    let components = bind(input)?;
    let rows = tokens.len();
    let hidden = input.entry.width;
    let entry = entry_reference::embedding_norm(
        &input.entry.table,
        tokens,
        &input.entry.weight,
        hidden,
        input.entry.epsilon,
    )?;

    let qkv = fp8_projection(&entry.normalized, rows, hidden, components.qkv)?;
    let z = fp8_projection(&entry.normalized, rows, hidden, components.z)?;
    let a = bf16_projection(&entry.normalized, rows, hidden, components.a)?;
    let b = bf16_projection(&entry.normalized, rows, hidden, components.b)?;
    let convolution = causal_conv4_reference::run(
        &qkv.normalized,
        &decode_bf16_bytes(&components.convolution.weights, "convolution weights")?,
        &vec![0; components.qkv.channels * 3],
        rows,
        components.qkv.channels,
    )?;

    let prepared = gdn_prepare_reference::run(
        &convolution.output,
        &a.normalized,
        &b.normalized,
        &decode_bf16_bytes(&components.gdn.a_log, "GDN A-log")?,
        &decode_bf16_bytes(&components.gdn.dt_bias, "GDN dt-bias")?,
        &gdn_prepare_reference::Shape {
            rows,
            key_heads: components.gdn.key_heads,
            value_heads: components.gdn.value_heads,
            width: components.gdn.width,
        },
    )?;
    let recurrent = gdn_recurrent_reference::run(
        &gdn_recurrent_reference::Input {
            q: &prepared.q,
            k: &prepared.k,
            qkv: &convolution.output,
            beta: &prepared.beta,
            decay: &prepared.decay,
        },
        &vec![0.0; recurrent_state_len(components.gdn)?],
        &gdn_recurrent_reference::Shape {
            rows,
            key_heads: components.gdn.key_heads,
            value_heads: components.gdn.value_heads,
            width: components.gdn.width,
        },
        gdn_recurrent_reference::Reduction::OrderedF32,
    )?;
    let gated = gated_norm_reference::run(
        &recurrent.output,
        &z.normalized,
        &decode_bf16_bytes(&components.gdn_output.norm, "GDN output norm")?,
        rows.checked_mul(components.gdn.value_heads)
            .context("GDN output group count overflows usize")?,
        components.gdn.width,
        components.gdn_output.epsilon,
    )?;
    let output_projection = fp8_projection(
        &gated.output,
        rows,
        components.inner,
        &components.gdn_output.projection,
    )?;
    let post_attention = residual_norm_reference::run(
        &entry.residual,
        &output_projection.normalized,
        &decode_bf16_bytes(
            &components.post_attention_norm.weight,
            "post-attention norm weight",
        )?,
        rows,
        hidden,
        components.post_attention_norm.epsilon,
    )?;

    let gate = nvfp4_projection(
        &post_attention.normalized,
        rows,
        hidden,
        &components.mlp.gate,
    )?;
    let up = nvfp4_projection(&post_attention.normalized, rows, hidden, &components.mlp.up)?;
    let activated = mlp_activation_reference::run(&gate.normalized, &up.normalized)?;
    let down = nvfp4_projection(
        &activated.output,
        rows,
        components.mlp.gate.channels,
        &components.mlp.down,
    )?;
    let output = residual_add_reference::run(&post_attention.residual, &down.normalized)?;

    Ok(Layer {
        output,
        convolution_history: convolution.next_history,
        recurrent_state: recurrent.state,
    })
}

fn bind(input: &kernels::ProjectionInput) -> Result<Components<'_>> {
    let convolution = input
        .convolution
        .as_ref()
        .context("Qwen layer requires causal convolution weights")?;
    let gdn = input
        .gdn
        .as_ref()
        .context("Qwen layer requires GDN weights")?;
    let gdn_output = input
        .gdn_output
        .as_ref()
        .context("Qwen layer requires GDN output weights")?;
    let post_attention_norm = input
        .post_attention_norm
        .as_ref()
        .context("Qwen layer requires post-attention norm weights")?;
    let mlp = input
        .mlp
        .as_ref()
        .context("Qwen layer requires NVFP4 MLP weights")?;
    let qkv = input
        .projections
        .get(convolution.projection)
        .context("Qwen QKV projection index is invalid")?;
    let z = input
        .projections
        .get(gdn_output.z_projection)
        .context("Qwen Z projection index is invalid")?;
    let a = input
        .bf16_projections
        .get(gdn.a_projection)
        .context("Qwen A projection index is invalid")?;
    let b = input
        .bf16_projections
        .get(gdn.b_projection)
        .context("Qwen B projection index is invalid")?;

    let qkv_heads = gdn
        .key_heads
        .checked_mul(2)
        .and_then(|heads| heads.checked_add(gdn.value_heads))
        .context("Qwen QKV head count overflows usize")?;
    let qkv_channels = qkv_heads
        .checked_mul(gdn.width)
        .context("Qwen QKV channel count overflows usize")?;
    let inner = gdn
        .value_heads
        .checked_mul(gdn.width)
        .context("Qwen GDN inner width overflows usize")?;
    ensure!(qkv.channels == qkv_channels, "Qwen QKV channel mismatch");
    ensure!(z.channels == inner, "Qwen Z channel mismatch");
    ensure!(a.channels == gdn.value_heads, "Qwen A channel mismatch");
    ensure!(b.channels == gdn.value_heads, "Qwen B channel mismatch");
    ensure!(
        gdn_output.projection.channels == input.entry.width,
        "Qwen GDN output channel mismatch"
    );
    ensure!(
        mlp.gate.channels == mlp.up.channels,
        "Qwen MLP gate/up channel mismatch"
    );
    ensure!(
        mlp.down.channels == input.entry.width,
        "Qwen MLP down channel mismatch"
    );
    Ok(Components {
        convolution,
        gdn,
        gdn_output,
        post_attention_norm,
        mlp,
        qkv,
        z,
        a,
        b,
        inner,
    })
}

fn fp8_projection(
    input: &[u16],
    rows: usize,
    width: usize,
    projection: &kernels::Fp8Projection,
) -> Result<projection_reference::LinearReference> {
    let quantized = projection_reference::quantize(input, rows, width)?;
    let scales = decode_bf16_bytes(&projection.scales, "FP8 projection scales")?;
    projection_reference::linear(&quantized, &projection.weights, &scales, width)
}

fn bf16_projection(
    input: &[u16],
    rows: usize,
    width: usize,
    projection: &kernels::Bf16Projection,
) -> Result<projection_reference::LinearReference> {
    let weights = decode_bf16_bytes(&projection.weights, "BF16 projection weights")?;
    projection_reference::linear_bf16(input, &weights, rows, width)
}

fn nvfp4_projection(
    input: &[u16],
    rows: usize,
    width: usize,
    projection: &kernels::Nvfp4Projection,
) -> Result<projection_reference::LinearReference> {
    let quantized = nvfp4_quantize_reference::run(input, rows, width, projection.input_global)?;
    nvfp4_linear_reference::run(
        nvfp4_linear_reference::Matrix {
            packed: &quantized.packed,
            scales: &quantized.scales,
            rows,
            global: projection.input_global,
        },
        nvfp4_linear_reference::Matrix {
            packed: &projection.packed,
            scales: &projection.scales,
            rows: projection.channels,
            global: projection.weight_global,
        },
        width,
    )
}

fn decode_bf16_bytes(bytes: &[u8], label: &str) -> Result<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    ensure!(
        remainder.is_empty(),
        "{label} byte extent is not BF16 aligned"
    );
    Ok(words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

fn recurrent_state_len(gdn: &kernels::GdnWeights) -> Result<usize> {
    gdn.value_heads
        .checked_mul(gdn.width)
        .and_then(|size| size.checked_mul(gdn.width))
        .context("Qwen GDN recurrent state extent overflows usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn repeated_bf16(value: u16, count: usize) -> Vec<u8> {
        bf16_bytes(&vec![value; count])
    }

    fn fp8_projection(name: &str, channels: usize, width: usize) -> kernels::Fp8Projection {
        kernels::Fp8Projection {
            name: name.to_owned(),
            weights: vec![0; channels * width],
            scales: repeated_bf16(0x3f80, channels),
            channels,
        }
    }

    fn nvfp4_projection(name: &str, channels: usize, width: usize) -> kernels::Nvfp4Projection {
        kernels::Nvfp4Projection {
            name: name.to_owned(),
            packed: vec![0; channels * width / 2],
            scales: vec![0; channels * width / 16],
            input_global: 1.0,
            weight_global: 1.0,
            channels,
        }
    }

    fn tiny_input() -> kernels::ProjectionInput {
        let hidden = 16;
        let vocab = 2;
        let embedding = vec![0x3f80; vocab * hidden];
        kernels::ProjectionInput {
            entry: kernels::EmbeddingNormInput {
                table: bf16_bytes(&embedding),
                weight: repeated_bf16(0, hidden),
                width: hidden,
                epsilon: 1e-6,
                batches: vec![vec![0]],
            },
            projections: vec![
                fp8_projection("qkv", 3, hidden),
                fp8_projection("z", 1, hidden),
            ],
            bf16_projections: vec![
                kernels::Bf16Projection {
                    name: "a".to_owned(),
                    weights: repeated_bf16(0, hidden),
                    channels: 1,
                },
                kernels::Bf16Projection {
                    name: "b".to_owned(),
                    weights: repeated_bf16(0, hidden),
                    channels: 1,
                },
            ],
            convolution: Some(kernels::CausalConv4Weights {
                projection: 0,
                weights: repeated_bf16(0, 3 * 4),
            }),
            gdn: Some(kernels::GdnWeights {
                a_projection: 0,
                b_projection: 1,
                key_heads: 1,
                value_heads: 1,
                width: 1,
                a_log: repeated_bf16(0, 1),
                dt_bias: repeated_bf16(0, 1),
            }),
            gdn_output: Some(kernels::GdnOutputWeights {
                z_projection: 1,
                norm: repeated_bf16(0, 1),
                epsilon: 1e-6,
                projection: fp8_projection("out", hidden, 1),
            }),
            post_attention_norm: Some(kernels::ResidualNormWeights {
                weight: repeated_bf16(0, hidden),
                epsilon: 1e-6,
            }),
            mlp: Some(kernels::Nvfp4Mlp {
                gate: nvfp4_projection("gate", hidden, hidden),
                up: nvfp4_projection("up", hidden, hidden),
                down: nvfp4_projection("down", hidden, hidden),
            }),
        }
    }

    #[test]
    fn zero_weight_attention_and_mlp_preserve_nonzero_embedding_residual() {
        let input = tiny_input();
        let result = run(&input, &[0]).unwrap();
        assert_eq!(result.output, [0x3f80; 16]);
        assert_eq!(result.convolution_history, [0; 9]);
        assert_eq!(result.recurrent_state, [0.0]);
    }

    #[test]
    fn missing_options_invalid_indices_and_channel_links_are_rejected() {
        let mut missing = tiny_input();
        missing.mlp = None;
        assert!(run(&missing, &[0]).is_err());

        let mut bad_index = tiny_input();
        bad_index.convolution.as_mut().unwrap().projection = 7;
        assert!(run(&bad_index, &[0]).is_err());

        let mut bad_shape = tiny_input();
        bad_shape.projections[0].channels = 2;
        assert!(run(&bad_shape, &[0]).is_err());
    }
}
