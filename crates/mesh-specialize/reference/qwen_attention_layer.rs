//! Independent CPU composition of the pinned Qwen attention layer path.

use anyhow::{Context, Result, ensure};

use crate::{
    attention_gate_reference, attention_prepare_reference, causal_attention_reference,
    entry_reference, kernels, nvfp4_linear_reference, nvfp4_quantize_reference,
    projection_reference, residual_add_reference, residual_norm_reference,
};

const MAX_ELEMENTS: usize = 67_108_864;

#[derive(Debug, PartialEq)]
pub struct Layer {
    pub output: Vec<u16>,
    /// Initialized, rotary-encoded K rows in `[rows, kv_heads, head_width]` order.
    pub k_cache: Vec<u16>,
    /// Projected V rows in `[rows, kv_heads, head_width]` order.
    pub v_cache: Vec<u16>,
}

struct Dimensions {
    rows: usize,
    hidden: usize,
    query_inner: usize,
    mlp_inner: usize,
}

struct AttentionStage {
    projection: Vec<u16>,
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
}

/// Run the logical attention layer from original weights, token IDs, and positions.
///
/// The composition does not consume GPU intermediates. It returns the final
/// post-attention/post-MLP BF16 rows and the K/V rows initialized for this chunk.
pub fn run(input: &kernels::AttentionInput, tokens: &[u32], positions: &[u32]) -> Result<Layer> {
    let dimensions = validate_input(input, tokens, positions)?;
    let (cos, sin) = attention_prepare_reference::text_rope_tables(
        positions,
        input.rotary_dim,
        input.rope_theta,
    )?;
    let entry = entry_reference::embedding_norm(
        &input.entry.table,
        tokens,
        &input.entry.weight,
        dimensions.hidden,
        input.entry.epsilon,
    )?;
    let attention = run_attention_stage(input, &entry.normalized, &dimensions, &cos, &sin)?;
    let post_attention = residual_norm_reference::run(
        &entry.residual,
        &attention.projection,
        &decode_bf16_bytes(
            &input.post_attention_norm.weight,
            "post-attention norm weight",
        )?,
        dimensions.rows,
        dimensions.hidden,
        input.post_attention_norm.epsilon,
    )?;
    let down = run_mlp(input, &post_attention.normalized, &dimensions)?;
    let output = residual_add_reference::run(&post_attention.residual, &down)?;
    Ok(Layer {
        output,
        k_cache: attention.k_cache,
        v_cache: attention.v_cache,
    })
}

fn validate_input(
    input: &kernels::AttentionInput,
    tokens: &[u32],
    positions: &[u32],
) -> Result<Dimensions> {
    let rows = tokens.len();
    ensure!(
        (1..=2048).contains(&rows),
        "invalid attention layer row count"
    );
    ensure!(
        positions.len() == rows,
        "attention token/position count mismatch"
    );
    ensure!(
        (1..=128).contains(&input.query_heads)
            && (1..=128).contains(&input.kv_heads)
            && input.query_heads.is_multiple_of(input.kv_heads),
        "attention head counts are invalid"
    );
    ensure!(
        (2..=256).contains(&input.head_width)
            && input.rotary_dim >= 2
            && input.rotary_dim <= input.head_width
            && input.rotary_dim.is_multiple_of(2),
        "attention head or rotary width is invalid"
    );
    ensure!(
        input.rope_theta.is_finite() && input.rope_theta > 1.0,
        "invalid attention RoPE theta"
    );
    ensure!(
        input.entry.width.is_multiple_of(16) && (16..=32768).contains(&input.entry.width),
        "attention hidden width must be a multiple of 16 in 16..=32768"
    );
    ensure!(
        input.entry.epsilon.is_finite() && input.entry.epsilon > 0.0,
        "invalid attention entry epsilon"
    );

    let hidden = input.entry.width;
    let query_inner = checked_product(
        &[input.query_heads, input.head_width],
        "attention query width",
    )?;
    let kv_inner = checked_product(&[input.kv_heads, input.head_width], "attention KV width")?;
    ensure!(
        (1..=32768).contains(&query_inner) && (1..=32768).contains(&kv_inner),
        "attention projection width is out of range"
    );
    let query_channels = checked_product(&[query_inner, 2], "attention Q/gate channels")?;
    validate_count(rows, query_channels, "Q/gate projection")?;
    validate_count(rows, kv_inner, "K/V projection")?;
    validate_count(rows, hidden, "attention hidden state")?;

    ensure!(
        input
            .entry
            .table
            .len()
            .is_multiple_of(checked_product(&[hidden, 2], "embedding row bytes")?)
            && !input.entry.table.is_empty(),
        "attention embedding table has an invalid BF16 extent"
    );
    ensure!(
        input.entry.weight.len() == checked_product(&[hidden, 2], "embedding norm bytes")?,
        "attention embedding norm byte extent mismatch"
    );
    ensure!(
        input.q_norm.len() == checked_product(&[input.head_width, 2], "Q norm bytes")?
            && input.k_norm.len() == checked_product(&[input.head_width, 2], "K norm bytes")?,
        "attention Q/K norm byte extent mismatch"
    );
    ensure!(
        input.post_attention_norm.weight.len()
            == checked_product(&[hidden, 2], "post-attention norm bytes")?,
        "post-attention norm byte extent mismatch"
    );
    ensure!(
        input.post_attention_norm.epsilon.is_finite() && input.post_attention_norm.epsilon > 0.0,
        "invalid post-attention norm epsilon"
    );

    validate_fp8_projection(&input.projections[0], hidden, query_channels, "Q")?;
    validate_fp8_projection(&input.projections[1], hidden, kv_inner, "K")?;
    validate_fp8_projection(&input.projections[2], hidden, kv_inner, "V")?;
    validate_fp8_projection(
        &input.output_projection,
        query_inner,
        hidden,
        "attention output",
    )?;

    let mlp_inner = input.mlp.gate.channels;
    ensure!(
        input.mlp.up.channels == mlp_inner
            && (16..=32768).contains(&mlp_inner)
            && mlp_inner.is_multiple_of(16),
        "MLP gate/up channel shape is invalid"
    );
    ensure!(
        input.mlp.down.channels == hidden,
        "MLP down channels must equal hidden width"
    );
    validate_count(rows, mlp_inner, "MLP gate/up projection")?;
    validate_nvfp4_projection(&input.mlp.gate, hidden, mlp_inner, "MLP gate")?;
    validate_nvfp4_projection(&input.mlp.up, hidden, mlp_inner, "MLP up")?;
    validate_nvfp4_projection(&input.mlp.down, mlp_inner, hidden, "MLP down")?;

    Ok(Dimensions {
        rows,
        hidden,
        query_inner,
        mlp_inner,
    })
}

fn run_attention_stage(
    input: &kernels::AttentionInput,
    normalized_entry: &[u16],
    dimensions: &Dimensions,
    cos: &[u16],
    sin: &[u16],
) -> Result<AttentionStage> {
    let entry_quantized =
        projection_reference::quantize(normalized_entry, dimensions.rows, dimensions.hidden)?;
    let q_linear = fp8_linear(&entry_quantized, &input.projections[0], dimensions.hidden)?;
    let k_linear = fp8_linear(&entry_quantized, &input.projections[1], dimensions.hidden)?;
    let v_linear = fp8_linear(&entry_quantized, &input.projections[2], dimensions.hidden)?;
    let q_prepared = attention_prepare_reference::run(
        &q_linear.normalized,
        &decode_bf16_bytes(&input.q_norm, "Q norm weight")?,
        cos,
        sin,
        &attention_prepare_reference::Shape {
            rows: dimensions.rows,
            heads: input.query_heads,
            width: input.head_width,
            rotary_dim: input.rotary_dim,
            with_gate: true,
        },
        1e-6,
    )?;
    let k_prepared = attention_prepare_reference::run(
        &k_linear.normalized,
        &decode_bf16_bytes(&input.k_norm, "K norm weight")?,
        cos,
        sin,
        &attention_prepare_reference::Shape {
            rows: dimensions.rows,
            heads: input.kv_heads,
            width: input.head_width,
            rotary_dim: input.rotary_dim,
            with_gate: false,
        },
        1e-6,
    )?;
    let causal = causal_attention_reference::run(
        &q_prepared.output,
        &k_prepared.output,
        &v_linear.normalized,
        &causal_attention_reference::Shape {
            rows: dimensions.rows,
            query_heads: input.query_heads,
            kv_heads: input.kv_heads,
            width: input.head_width,
            past: 0,
            capacity: dimensions.rows,
            scale: 1.0 / (input.head_width as f32).sqrt(),
        },
    )?;
    let gated = attention_gate_reference::run(&causal.output, &q_prepared.gate)?;
    let gated_quantized =
        projection_reference::quantize(&gated.output, dimensions.rows, dimensions.query_inner)?;
    let projected = fp8_linear(
        &gated_quantized,
        &input.output_projection,
        dimensions.query_inner,
    )?;
    Ok(AttentionStage {
        projection: projected.normalized,
        k_cache: k_prepared.output,
        v_cache: v_linear.normalized,
    })
}

fn run_mlp(
    input: &kernels::AttentionInput,
    normalized: &[u16],
    dimensions: &Dimensions,
) -> Result<Vec<u16>> {
    let gate = nvfp4_linear(
        normalized,
        dimensions.rows,
        dimensions.hidden,
        &input.mlp.gate,
    )?;
    let up = nvfp4_linear(
        normalized,
        dimensions.rows,
        dimensions.hidden,
        &input.mlp.up,
    )?;
    let activation = crate::mlp_activation_reference::run(&gate.normalized, &up.normalized)?;
    let down = nvfp4_linear(
        &activation.output,
        dimensions.rows,
        dimensions.mlp_inner,
        &input.mlp.down,
    )?;
    Ok(down.normalized)
}

fn fp8_linear(
    activation: &projection_reference::QuantizedRows,
    projection: &kernels::Fp8Projection,
    width: usize,
) -> Result<projection_reference::LinearReference> {
    let scales = decode_bf16_bytes(&projection.scales, "FP8 projection scales")?;
    projection_reference::linear(activation, &projection.weights, &scales, width)
}

fn nvfp4_linear(
    activation: &[u16],
    rows: usize,
    width: usize,
    projection: &kernels::Nvfp4Projection,
) -> Result<projection_reference::LinearReference> {
    let quantized =
        nvfp4_quantize_reference::run(activation, rows, width, projection.input_global)?;
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

fn validate_fp8_projection(
    projection: &kernels::Fp8Projection,
    input_width: usize,
    channels: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        projection.channels == channels,
        "{label} projection channel mismatch"
    );
    let weight_elements = checked_product(&[channels, input_width], label)?;
    ensure!(
        weight_elements <= MAX_ELEMENTS,
        "{label} projection exceeds element limit"
    );
    ensure!(
        projection.weights.len() == weight_elements,
        "{label} FP8 weight extent mismatch"
    );
    ensure!(
        projection.scales.len() == checked_product(&[channels, 2], label)?,
        "{label} FP8 scale byte extent mismatch"
    );
    Ok(())
}

fn validate_nvfp4_projection(
    projection: &kernels::Nvfp4Projection,
    input_width: usize,
    channels: usize,
    label: &str,
) -> Result<()> {
    ensure!(projection.channels == channels, "{label} channel mismatch");
    ensure!(
        (16..=32768).contains(&input_width) && input_width.is_multiple_of(16),
        "{label} input width is invalid"
    );
    ensure!(
        projection.input_global.is_finite()
            && projection.input_global > 0.0
            && projection.weight_global.is_finite()
            && projection.weight_global > 0.0,
        "{label} global scales must be positive and finite"
    );
    let global_product = projection.input_global * projection.weight_global;
    ensure!(
        global_product.is_finite()
            && global_product > 0.0
            && (1.0 / global_product).is_finite()
            && (1.0 / global_product) > 0.0,
        "{label} combined global scale is not representable"
    );
    let packed_len = checked_product(&[channels, input_width / 2], label)?;
    let scale_len = checked_product(&[channels, input_width / 16], label)?;
    ensure!(
        packed_len <= MAX_ELEMENTS,
        "{label} packed weights exceed element limit"
    );
    ensure!(
        projection.packed.len() == packed_len,
        "{label} packed weight extent mismatch"
    );
    ensure!(
        projection.scales.len() == scale_len,
        "{label} scale extent mismatch"
    );
    ensure!(
        projection.scales.iter().all(|&scale| scale <= 126),
        "{label} scales contain nonfinite E4M3 codes"
    );
    Ok(())
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

fn validate_count(rows: usize, channels: usize, label: &str) -> Result<()> {
    ensure!(
        checked_product(&[rows, channels], label)? <= MAX_ELEMENTS,
        "{label} exceeds element limit"
    );
    Ok(())
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .with_context(|| format!("{label} extent overflows usize"))
    })
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

    fn tiny_input(rows: usize) -> kernels::AttentionInput {
        let hidden = 16;
        let query_inner = 2 * 8;
        let kv_inner = 8;
        kernels::AttentionInput {
            output_projection: fp8_projection("attention-output", hidden, query_inner),
            post_attention_norm: kernels::ResidualNormWeights {
                weight: repeated_bf16(0, hidden),
                epsilon: 1e-6,
            },
            mlp: kernels::Nvfp4Mlp {
                gate: nvfp4_projection("gate", hidden, hidden),
                up: nvfp4_projection("up", hidden, hidden),
                down: nvfp4_projection("down", hidden, hidden),
            },
            entry: kernels::EmbeddingNormInput {
                table: bf16_bytes(
                    &[0x3f80; 16]
                        .into_iter()
                        .chain([0x4000; 16])
                        .collect::<Vec<_>>(),
                ),
                weight: repeated_bf16(0, hidden),
                width: hidden,
                epsilon: 1e-6,
                batches: vec![vec![0; rows]],
            },
            projections: [
                fp8_projection("q", 2 * query_inner, hidden),
                fp8_projection("k", kv_inner, hidden),
                fp8_projection("v", kv_inner, hidden),
            ],
            q_norm: repeated_bf16(0, 8),
            k_norm: repeated_bf16(0, 8),
            query_heads: 2,
            kv_heads: 1,
            head_width: 8,
            rotary_dim: 4,
            rope_theta: 10_000.0,
            positions: vec![vec![0; rows]],
        }
    }

    #[test]
    fn zero_branches_preserve_nonzero_embedding_residual_for_one_and_two_rows() {
        for tokens in [&[0][..], &[1, 0][..]] {
            let input = tiny_input(tokens.len());
            let positions = (0..tokens.len()).map(|row| row as u32).collect::<Vec<_>>();
            let result = run(&input, tokens, &positions).unwrap();
            let expected = tokens
                .iter()
                .flat_map(|&token| vec![if token == 0 { 0x3f80 } else { 0x4000 }; 16])
                .collect::<Vec<_>>();
            assert_eq!(result.output, expected);
            assert_eq!(result.k_cache, vec![0; tokens.len() * 8]);
            assert_eq!(result.v_cache, vec![0; tokens.len() * 8]);
        }
    }

    #[test]
    fn invalid_links_positions_tokens_and_bf16_extents_are_rejected() {
        let tokens = [0];
        let positions = [0];
        assert!(run(&tiny_input(1), &tokens, &[]).is_err());
        assert!(run(&tiny_input(1), &[2], &positions).is_err());

        let mut bad_channels = tiny_input(1);
        bad_channels.projections[0].channels -= 1;
        assert!(run(&bad_channels, &tokens, &positions).is_err());

        let mut bad_heads = tiny_input(1);
        bad_heads.query_heads = 3;
        assert!(run(&bad_heads, &tokens, &positions).is_err());

        let mut odd_norm = tiny_input(1);
        odd_norm.q_norm.push(0);
        assert!(run(&odd_norm, &tokens, &positions).is_err());

        let mut bad_mlp = tiny_input(1);
        bad_mlp.mlp.down.channels -= 1;
        assert!(run(&bad_mlp, &tokens, &positions).is_err());
    }
}
