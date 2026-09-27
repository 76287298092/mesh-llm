//! Independent CPU composition of the Qwen MTP continuation block.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};

use crate::{
    attention_gate_reference, attention_prepare_reference, causal_attention_reference,
    decoder_ops_reference, entry_reference, kernels, mlp_activation_reference,
    projection_reference, residual_add_reference, residual_norm_reference,
};

const MAX_ELEMENTS: usize = 67_108_864;
const MAX_PROJECTION_CHANNELS: usize = 262_144;
const MAX_WEIGHT_ELEMENTS: usize = 2_000_000_000;

pub struct Weights {
    pub target_norm: Vec<u16>,
    /// BF16 embedding rows selected for the shifted MTP token IDs.
    pub embedding_rows: Vec<u16>,
    pub pre_embedding_norm: Vec<u16>,
    pub pre_hidden_norm: Vec<u16>,
    pub fc: kernels::Bf16Projection,
    pub input_norm: Vec<u16>,
    pub post_norm: Vec<u16>,
    pub q: kernels::Bf16Projection,
    pub k: kernels::Bf16Projection,
    pub v: kernels::Bf16Projection,
    pub out: kernels::Bf16Projection,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub gate: kernels::Bf16Projection,
    pub up: kernels::Bf16Projection,
    pub down: kernels::Bf16Projection,
    pub final_norm: Vec<u16>,
    pub head: kernels::Fp8Projection,
}

#[derive(Debug, PartialEq)]
pub struct Output {
    /// BF16 values at each named MTP composition boundary.
    pub stages: BTreeMap<String, Vec<u16>>,
    /// Full `[rows, hidden]` output after the MTP final norm.
    pub hidden: Vec<u16>,
    /// Final vocabulary logits for the last row only.
    pub logits: Vec<u16>,
    /// Initialized, RoPE-applied `[rows, kv_heads, head_width]` key cache.
    pub key: Vec<u16>,
    /// Initialized `[rows, kv_heads, head_width]` value cache.
    pub value: Vec<u16>,
}

struct Dimensions {
    hidden: usize,
    intermediate: usize,
    query_inner: usize,
    fc_input: usize,
}

struct BlockOutput {
    hidden: Vec<u16>,
    key: Vec<u16>,
    value: Vec<u16>,
}

/// Run the MTP head from the target model's raw final hidden rows and selected
/// shifted-token embeddings. The implementation uses only these inputs and
/// original weights; it does not consume GPU intermediates.
pub fn run(
    raw_target_hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
) -> Result<Output> {
    let dimensions = validate(raw_target_hidden, rows, shape, weights)?;
    let mut stages = BTreeMap::new();

    let target_hidden = decoder_ops_reference::normalize(
        raw_target_hidden,
        &weights.target_norm,
        rows,
        dimensions.hidden,
    )?;
    capture(&mut stages, "target_norm", &target_hidden)?;
    let embedding = decoder_ops_reference::normalize(
        &weights.embedding_rows,
        &weights.pre_embedding_norm,
        rows,
        dimensions.hidden,
    )?;
    capture(&mut stages, "pre_embedding_norm", &embedding)?;
    let hidden = decoder_ops_reference::normalize(
        &target_hidden,
        &weights.pre_hidden_norm,
        rows,
        dimensions.hidden,
    )?;
    capture(&mut stages, "pre_hidden_norm", &hidden)?;

    let combined = concatenate_embedding_then_hidden(&embedding, &hidden, rows, dimensions.hidden)?;
    let fc = decoder_ops_reference::bf16(&combined, &weights.fc, rows, dimensions.fc_input)?;
    capture(&mut stages, "fc", &fc)?;

    let block = run_block(&fc, rows, shape, weights, &dimensions, &mut stages)?;
    capture(&mut stages, "decoder_hidden", &block.hidden)?;
    let final_hidden = decoder_ops_reference::normalize(
        &block.hidden,
        &weights.final_norm,
        rows,
        dimensions.hidden,
    )?;
    capture(&mut stages, "final_norm", &final_hidden)?;

    let last_row = (rows - 1) * dimensions.hidden;
    let logits = decoder_ops_reference::fp8(
        &final_hidden[last_row..last_row + dimensions.hidden],
        &weights.head,
        1,
        dimensions.hidden,
    )?;
    capture(&mut stages, "logits", &logits)?;
    capture(&mut stages, "key", &block.key)?;
    capture(&mut stages, "value", &block.value)?;

    Ok(Output {
        stages,
        hidden: final_hidden,
        logits,
        key: block.key,
        value: block.value,
    })
}

fn run_block(
    residual: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
    dimensions: &Dimensions,
    stages: &mut BTreeMap<String, Vec<u16>>,
) -> Result<BlockOutput> {
    let normalized =
        decoder_ops_reference::normalize(residual, &weights.input_norm, rows, dimensions.hidden)?;
    capture(stages, "input_norm", &normalized)?;
    let q_linear = decoder_ops_reference::bf16(&normalized, &weights.q, rows, dimensions.hidden)?;
    capture(stages, "q_projection", &q_linear)?;
    let k_linear = decoder_ops_reference::bf16(&normalized, &weights.k, rows, dimensions.hidden)?;
    capture(stages, "k_projection", &k_linear)?;
    let value = decoder_ops_reference::bf16(&normalized, &weights.v, rows, dimensions.hidden)?;
    capture(stages, "v_projection", &value)?;

    let positions = (0..u32::try_from(rows)?).collect::<Vec<_>>();
    let (cos, sin) = attention_prepare_reference::text_rope_tables(
        &positions,
        shape.rotary_dim,
        shape.rope_theta,
    )?;
    let q = attention_prepare_reference::run(
        &q_linear,
        &weights.q_norm,
        &cos,
        &sin,
        &attention_prepare_reference::Shape {
            rows,
            heads: shape.query_heads,
            width: shape.head_width,
            rotary_dim: shape.rotary_dim,
            with_gate: true,
        },
        1e-6,
    )?;
    capture(stages, "q_prepared", &q.output)?;
    capture(stages, "q_gate", &q.gate)?;
    let k = attention_prepare_reference::run(
        &k_linear,
        &weights.k_norm,
        &cos,
        &sin,
        &attention_prepare_reference::Shape {
            rows,
            heads: shape.kv_heads,
            width: shape.head_width,
            rotary_dim: shape.rotary_dim,
            with_gate: false,
        },
        1e-6,
    )?;
    capture(stages, "k_prepared", &k.output)?;

    let attended = causal_attention_reference::run(
        &q.output,
        &k.output,
        &value,
        &causal_attention_reference::Shape {
            rows,
            query_heads: shape.query_heads,
            kv_heads: shape.kv_heads,
            width: shape.head_width,
            past: 0,
            capacity: rows,
            scale: 1.0 / (shape.head_width as f32).sqrt(),
        },
    )?;
    capture(stages, "attended", &attended.output)?;
    let gated = attention_gate_reference::run(&attended.output, &q.gate)?;
    capture(stages, "attention_gate", &gated.output)?;
    let projected =
        decoder_ops_reference::bf16(&gated.output, &weights.out, rows, dimensions.query_inner)?;
    capture(stages, "out_projection", &projected)?;

    let post_attention = residual_norm_reference::run(
        residual,
        &projected,
        &weights.post_norm,
        rows,
        dimensions.hidden,
        1e-6,
    )?;
    capture(stages, "post_residual", &post_attention.residual)?;
    capture(stages, "post_norm", &post_attention.normalized)?;
    run_mlp(
        post_attention.residual,
        &post_attention.normalized,
        rows,
        weights,
        dimensions,
        stages,
    )
    .map(|hidden| BlockOutput {
        hidden,
        key: k.output,
        value,
    })
}

fn run_mlp(
    residual: Vec<u16>,
    normalized: &[u16],
    rows: usize,
    weights: &Weights,
    dimensions: &Dimensions,
    stages: &mut BTreeMap<String, Vec<u16>>,
) -> Result<Vec<u16>> {
    let gate = decoder_ops_reference::bf16(normalized, &weights.gate, rows, dimensions.hidden)?;
    capture(stages, "mlp_gate", &gate)?;
    let up = decoder_ops_reference::bf16(normalized, &weights.up, rows, dimensions.hidden)?;
    capture(stages, "mlp_up", &up)?;
    let activated = mlp_activation_reference::run(&gate, &up)?;
    capture(stages, "mlp_activation", &activated.output)?;
    let down = decoder_ops_reference::bf16(
        &activated.output,
        &weights.down,
        rows,
        dimensions.intermediate,
    )?;
    capture(stages, "mlp_down", &down)?;
    residual_add_reference::run(&residual, &down)
}

fn validate(
    raw_target_hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
) -> Result<Dimensions> {
    let dimensions = validate_shape(rows, shape)?;
    validate_inputs(raw_target_hidden, rows, shape, weights)?;
    validate_norms(shape, weights)?;
    validate_projections(shape, &dimensions, weights)?;
    Ok(dimensions)
}

fn validate_shape(rows: usize, shape: &kernels::ResidentAttentionShape) -> Result<Dimensions> {
    ensure!((1..=2048).contains(&rows), "invalid MTP row count");
    ensure!(
        (16..=32768).contains(&shape.hidden) && (16..=32768).contains(&shape.intermediate),
        "MTP hidden and intermediate widths must be in 16..=32768"
    );
    ensure!(
        (1..=128).contains(&shape.query_heads)
            && (1..=128).contains(&shape.kv_heads)
            && shape.query_heads.is_multiple_of(shape.kv_heads),
        "MTP attention head counts are invalid"
    );
    ensure!(
        (2..=256).contains(&shape.head_width)
            && (2..=shape.head_width).contains(&shape.rotary_dim)
            && shape.rotary_dim.is_multiple_of(2),
        "MTP head or rotary width is invalid"
    );
    ensure!(
        shape.rope_theta.is_finite() && shape.rope_theta > 1.0,
        "MTP RoPE theta must be finite and greater than one"
    );

    let hidden_extent = checked_product(&[rows, shape.hidden], "MTP hidden")?;
    ensure!(
        hidden_extent <= MAX_ELEMENTS,
        "MTP hidden exceeds element limit"
    );
    let fc_input = checked_product(&[shape.hidden, 2], "MTP FC input width")?;
    ensure!(
        fc_input <= 32768,
        "MTP concatenated FC input width exceeds BF16 reference limit"
    );
    ensure!(
        checked_product(&[rows, fc_input], "MTP concatenated input")? <= MAX_ELEMENTS,
        "MTP concatenated input exceeds element limit"
    );
    let query_inner = checked_product(&[shape.query_heads, shape.head_width], "MTP query width")?;
    let kv_inner = checked_product(&[shape.kv_heads, shape.head_width], "MTP KV width")?;
    let q_channels = checked_product(&[query_inner, 2], "MTP Q/gate width")?;
    let cache_extent = checked_product(&[rows, kv_inner], "MTP cache")?;
    ensure!(
        cache_extent <= MAX_ELEMENTS,
        "MTP cache exceeds element limit"
    );
    validate_activation_count(rows, q_channels, "MTP Q/gate output")?;
    validate_activation_count(rows, query_inner, "MTP attention output")?;
    validate_activation_count(rows, shape.intermediate, "MTP MLP output")?;
    Ok(Dimensions {
        hidden: shape.hidden,
        intermediate: shape.intermediate,
        query_inner,
        fc_input,
    })
}

fn validate_inputs(
    raw_target_hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
) -> Result<()> {
    let hidden_extent = checked_product(&[rows, shape.hidden], "MTP hidden")?;
    ensure!(
        raw_target_hidden.len() == hidden_extent,
        "MTP target hidden extent mismatch"
    );
    ensure!(
        weights.embedding_rows.len() == hidden_extent,
        "MTP selected embedding extent mismatch"
    );
    ensure!(
        raw_target_hidden
            .iter()
            .chain(&weights.embedding_rows)
            .all(|&bits| entry_reference::bf16_to_f32(bits).is_finite()),
        "MTP target hidden and embeddings must contain finite BF16 values"
    );
    Ok(())
}

fn validate_norms(shape: &kernels::ResidentAttentionShape, weights: &Weights) -> Result<()> {
    validate_norm(&weights.target_norm, shape.hidden, "MTP target norm")?;
    validate_norm(
        &weights.pre_embedding_norm,
        shape.hidden,
        "MTP embedding norm",
    )?;
    validate_norm(&weights.pre_hidden_norm, shape.hidden, "MTP hidden norm")?;
    validate_norm(&weights.input_norm, shape.hidden, "MTP input norm")?;
    validate_norm(&weights.post_norm, shape.hidden, "MTP post norm")?;
    validate_norm(&weights.q_norm, shape.head_width, "MTP Q norm")?;
    validate_norm(&weights.k_norm, shape.head_width, "MTP K norm")?;
    validate_norm(&weights.final_norm, shape.hidden, "MTP final norm")?;
    Ok(())
}

fn validate_projections(
    shape: &kernels::ResidentAttentionShape,
    dimensions: &Dimensions,
    weights: &Weights,
) -> Result<()> {
    validate_bf16(&weights.fc, dimensions.fc_input, shape.hidden, "MTP FC")?;
    validate_bf16(
        &weights.q,
        shape.hidden,
        dimensions.query_inner * 2,
        "MTP Q",
    )?;
    let kv_inner = checked_product(&[shape.kv_heads, shape.head_width], "MTP KV width")?;
    validate_bf16(&weights.k, shape.hidden, kv_inner, "MTP K")?;
    validate_bf16(&weights.v, shape.hidden, kv_inner, "MTP V")?;
    validate_bf16(
        &weights.out,
        dimensions.query_inner,
        shape.hidden,
        "MTP output",
    )?;
    validate_bf16(
        &weights.gate,
        shape.hidden,
        dimensions.intermediate,
        "MTP gate",
    )?;
    validate_bf16(&weights.up, shape.hidden, dimensions.intermediate, "MTP up")?;
    validate_bf16(
        &weights.down,
        dimensions.intermediate,
        shape.hidden,
        "MTP down",
    )?;
    validate_fp8(&weights.head, shape.hidden, "MTP head")?;
    Ok(())
}

fn validate_activation_count(rows: usize, width: usize, label: &str) -> Result<()> {
    ensure!(
        checked_product(&[rows, width], label)? <= MAX_ELEMENTS,
        "{label} exceeds element limit"
    );
    Ok(())
}

fn validate_norm(weight: &[u16], width: usize, label: &str) -> Result<()> {
    ensure!(weight.len() == width, "{label} extent mismatch");
    ensure!(
        weight
            .iter()
            .all(|&bits| entry_reference::bf16_to_f32(bits).is_finite()),
        "{label} must contain finite BF16 values"
    );
    Ok(())
}

fn validate_bf16(
    projection: &kernels::Bf16Projection,
    input_width: usize,
    channels: usize,
    label: &str,
) -> Result<()> {
    ensure!(projection.channels == channels, "{label} channel mismatch");
    let elements = checked_product(&[channels, input_width], label)?;
    ensure!(
        elements <= MAX_WEIGHT_ELEMENTS,
        "{label} weights exceed element limit"
    );
    let byte_len = elements
        .checked_mul(2)
        .with_context(|| format!("{label} byte length overflows usize"))?;
    ensure!(
        projection.weights.len() == byte_len,
        "{label} weight extent mismatch"
    );
    ensure!(
        projection
            .weights
            .as_chunks::<2>()
            .0
            .iter()
            .all(|pair| entry_reference::bf16_to_f32(u16::from_le_bytes(*pair)).is_finite()),
        "{label} weights must contain finite BF16 values"
    );
    Ok(())
}

fn validate_fp8(
    projection: &kernels::Fp8Projection,
    input_width: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        (1..=MAX_PROJECTION_CHANNELS).contains(&projection.channels),
        "{label} channel count is invalid"
    );
    let weight_len = checked_product(&[projection.channels, input_width], label)?;
    ensure!(
        weight_len <= MAX_WEIGHT_ELEMENTS,
        "{label} weights exceed element limit"
    );
    let scale_len = projection
        .channels
        .checked_mul(2)
        .with_context(|| format!("{label} scale length overflows usize"))?;
    ensure!(
        projection.weights.len() == weight_len,
        "{label} weight extent mismatch"
    );
    ensure!(
        projection.scales.len() == scale_len,
        "{label} scale extent mismatch"
    );
    ensure!(
        projection
            .weights
            .iter()
            .all(|&code| projection_reference::decode(code).is_finite()),
        "{label} weights must use finite E4M3FN codes"
    );
    ensure!(
        projection.scales.as_chunks::<2>().0.iter().all(|pair| {
            let scale = entry_reference::bf16_to_f32(u16::from_le_bytes(*pair));
            scale.is_finite() && scale > 0.0
        }),
        "{label} scales must be positive finite BF16 values"
    );
    Ok(())
}

fn concatenate_embedding_then_hidden(
    embedding: &[u16],
    hidden: &[u16],
    rows: usize,
    width: usize,
) -> Result<Vec<u16>> {
    let row_len = checked_product(&[width, 2], "MTP concatenated row")?;
    let input_len = checked_product(&[rows, width], "MTP concatenation input")?;
    let combined_len = checked_product(&[rows, row_len], "MTP concatenated input")?;
    ensure!(
        embedding.len() == input_len && hidden.len() == input_len,
        "MTP concatenation input extent mismatch"
    );
    ensure!(
        combined_len <= MAX_ELEMENTS,
        "MTP concatenated input exceeds element limit"
    );
    let mut combined = Vec::new();
    combined
        .try_reserve_exact(combined_len)
        .context("cannot reserve MTP concatenated input")?;
    for row in 0..rows {
        let start = row * width;
        let end = start + width;
        combined.extend_from_slice(&embedding[start..end]);
        combined.extend_from_slice(&hidden[start..end]);
    }
    Ok(combined)
}

fn capture(stages: &mut BTreeMap<String, Vec<u16>>, name: &str, values: &[u16]) -> Result<()> {
    ensure!(
        !stages.contains_key(name),
        "duplicate MTP stage name: {name}"
    );
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .with_context(|| format!("cannot reserve MTP stage {name}"))?;
    copy.extend_from_slice(values);
    stages.insert(name.to_owned(), copy);
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
    use super::{Output, Weights, concatenate_embedding_then_hidden, run};
    use crate::kernels::{Bf16Projection, Fp8Projection, ResidentAttentionShape};

    fn bf16_projection(name: &str, channels: usize, input_width: usize) -> Bf16Projection {
        Bf16Projection {
            name: name.to_owned(),
            weights: vec![0; channels * input_width * 2],
            channels,
        }
    }

    fn fp8_projection(name: &str, channels: usize, input_width: usize) -> Fp8Projection {
        let mut scales = Vec::with_capacity(channels * 2);
        for _ in 0..channels {
            scales.extend_from_slice(&0x3f80_u16.to_le_bytes());
        }
        Fp8Projection {
            name: name.to_owned(),
            weights: vec![0; channels * input_width],
            scales,
            channels,
        }
    }

    fn shape() -> ResidentAttentionShape {
        ResidentAttentionShape {
            hidden: 16,
            intermediate: 16,
            query_heads: 2,
            kv_heads: 1,
            head_width: 8,
            rotary_dim: 4,
            rope_theta: 10_000.0,
        }
    }

    fn weights(shape: &ResidentAttentionShape) -> Weights {
        let hidden = shape.hidden;
        let intermediate = shape.intermediate;
        let query_inner = shape.query_heads * shape.head_width;
        let kv_inner = shape.kv_heads * shape.head_width;
        Weights {
            target_norm: vec![0; hidden],
            embedding_rows: vec![0x3f80; hidden * 2],
            pre_embedding_norm: vec![0; hidden],
            pre_hidden_norm: vec![0; hidden],
            fc: bf16_projection("fc", hidden, hidden * 2),
            input_norm: vec![0; hidden],
            post_norm: vec![0; hidden],
            q: bf16_projection("q", query_inner * 2, hidden),
            k: bf16_projection("k", kv_inner, hidden),
            v: bf16_projection("v", kv_inner, hidden),
            out: bf16_projection("out", hidden, query_inner),
            q_norm: vec![0; shape.head_width],
            k_norm: vec![0; shape.head_width],
            gate: bf16_projection("gate", intermediate, hidden),
            up: bf16_projection("up", intermediate, hidden),
            down: bf16_projection("down", hidden, intermediate),
            final_norm: vec![0; hidden],
            head: fp8_projection("head", 4, hidden),
        }
    }

    #[test]
    fn concatenation_places_each_shifted_embedding_before_its_target_hidden() {
        let actual = concatenate_embedding_then_hidden(&[1, 2, 3, 4], &[5, 6, 7, 8], 2, 2).unwrap();
        assert_eq!(actual, [1, 2, 5, 6, 3, 4, 7, 8]);
    }

    #[test]
    fn zero_projection_fixture_returns_post_norm_rows_and_initialized_cache() {
        let model_shape = shape();
        let result: Output = run(&[0x3f80; 32], 2, &model_shape, &weights(&model_shape)).unwrap();
        assert_eq!(result.hidden, [0; 32]);
        assert_eq!(result.logits, [0; 4]);
        assert_eq!(result.key, [0; 16]);
        assert_eq!(result.value, [0; 16]);
        assert_eq!(result.stages["target_norm"].len(), 32);
        assert_eq!(result.stages["pre_embedding_norm"].len(), 32);
        assert_eq!(result.stages["fc"].len(), 32);
        assert_eq!(result.stages["q_projection"].len(), 64);
        assert_eq!(result.stages["k_prepared"].len(), 16);
        assert_eq!(result.stages["logits"], result.logits);
        assert_eq!(result.stages["key"], result.key);
        assert_eq!(result.stages["value"], result.value);
    }

    #[test]
    fn rejects_mismatched_rows_shape_and_projection_extents() {
        let model_shape = shape();
        let model_weights = weights(&model_shape);
        assert!(run(&[0x3f80; 16], 2, &model_shape, &model_weights).is_err());
        assert!(run(&[0x3f80; 32], 0, &model_shape, &model_weights).is_err());

        let mut invalid_shape = shape();
        invalid_shape.query_heads = 3;
        assert!(run(&[0x3f80; 32], 2, &invalid_shape, &model_weights).is_err());

        let mut invalid_weights = weights(&model_shape);
        invalid_weights.fc.channels += 1;
        assert!(run(&[0x3f80; 32], 2, &model_shape, &invalid_weights).is_err());
    }
}
