//! Independent CPU composition for the decoder's full-batch attention block.

use anyhow::{Context, Result, ensure};

use crate::{
    attention_gate_reference, attention_prepare_reference, causal_attention_reference,
    decoder_mlp_reference, decoder_ops_reference, kernels, residual_add_reference,
    residual_norm_reference,
};

const MAX_ELEMENTS: usize = 67_108_864;

pub struct Weights {
    pub input_norm: Vec<u16>,
    pub post_norm: Vec<u16>,
    pub q: kernels::Fp8Projection,
    pub k: kernels::Fp8Projection,
    pub v: kernels::Fp8Projection,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub out: kernels::Fp8Projection,
    pub mlp: decoder_mlp_reference::Weights,
}

struct Dimensions {
    hidden: usize,
    query_inner: usize,
}

/// Compute the complete attention-plus-MLP decoder block from BF16 hidden rows.
///
/// Rows are treated as consecutive text positions starting at zero, with no
/// initialized prefix cache. This CPU oracle consumes the original hidden
/// values and weights only; it does not use GPU intermediates.
pub fn run(
    hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
) -> Result<Vec<u16>> {
    run_observed(hidden, rows, shape, weights, &mut |_, _| Ok(()))
}

/// Run the decoder block and observe its BF16 boundaries without changing computation.
pub fn run_observed(
    hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
    observer: &mut dyn FnMut(&str, &[u16]) -> Result<()>,
) -> Result<Vec<u16>> {
    let dimensions = validate(hidden, rows, shape, weights)?;
    let normalized =
        decoder_ops_reference::normalize(hidden, &weights.input_norm, rows, dimensions.hidden)?;
    observer("normalized", &normalized)?;
    let q_linear = decoder_ops_reference::fp8(&normalized, &weights.q, rows, dimensions.hidden)?;
    observer("q_linear", &q_linear)?;
    let k_linear = decoder_ops_reference::fp8(&normalized, &weights.k, rows, dimensions.hidden)?;
    observer("k_linear", &k_linear)?;
    let v_linear = decoder_ops_reference::fp8(&normalized, &weights.v, rows, dimensions.hidden)?;
    observer("v_linear", &v_linear)?;

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
    observer("q_prepared", &q.output)?;
    observer("q_gate", &q.gate)?;
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
    observer("k_prepared", &k.output)?;
    let attended = causal_attention_reference::run(
        &q.output,
        &k.output,
        &v_linear,
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
    observer("attended", &attended.output)?;
    let gated = attention_gate_reference::run(&attended.output, &q.gate)?;
    observer("gated", &gated.output)?;
    let projected =
        decoder_ops_reference::fp8(&gated.output, &weights.out, rows, dimensions.query_inner)?;
    observer("out", &projected)?;
    let post_attention = residual_norm_reference::run(
        hidden,
        &projected,
        &weights.post_norm,
        rows,
        dimensions.hidden,
        1e-6,
    )?;
    observer("post_residual", &post_attention.residual)?;
    observer("post_norm", &post_attention.normalized)?;
    let mlp = decoder_mlp_reference::run_observed(
        &post_attention.normalized,
        rows,
        dimensions.hidden,
        &weights.mlp,
        observer,
    )?;
    let output = residual_add_reference::run(&post_attention.residual, &mlp)?;
    observer("hidden", &output)?;
    Ok(output)
}

fn validate(
    hidden: &[u16],
    rows: usize,
    shape: &kernels::ResidentAttentionShape,
    weights: &Weights,
) -> Result<Dimensions> {
    ensure!(
        (1..=2048).contains(&rows),
        "invalid decoder attention row count"
    );
    ensure!(
        (16..=32768).contains(&shape.hidden) && (16..=32768).contains(&shape.intermediate),
        "decoder hidden and intermediate widths must be in 16..=32768"
    );
    ensure!(
        (1..=128).contains(&shape.query_heads)
            && (1..=128).contains(&shape.kv_heads)
            && shape.query_heads.is_multiple_of(shape.kv_heads),
        "decoder attention head counts are invalid"
    );
    ensure!(
        (2..=256).contains(&shape.head_width)
            && (2..=shape.head_width).contains(&shape.rotary_dim)
            && shape.rotary_dim.is_multiple_of(2),
        "decoder head or rotary width is invalid"
    );
    ensure!(
        shape.rope_theta.is_finite() && shape.rope_theta > 1.0,
        "decoder RoPE theta must be finite and greater than one"
    );

    let hidden_elements = checked_product(rows, shape.hidden, "decoder hidden")?;
    ensure!(
        hidden.len() == hidden_elements,
        "decoder hidden extent mismatch"
    );
    ensure!(
        hidden
            .iter()
            .all(|&bits| crate::entry_reference::bf16_to_f32(bits).is_finite()),
        "decoder hidden rows must contain finite BF16 values"
    );
    ensure!(
        weights.input_norm.len() == shape.hidden,
        "decoder input norm extent mismatch"
    );
    ensure!(
        weights.post_norm.len() == shape.hidden,
        "decoder post norm extent mismatch"
    );
    ensure!(
        weights.q_norm.len() == shape.head_width,
        "decoder Q norm extent mismatch"
    );
    ensure!(
        weights.k_norm.len() == shape.head_width,
        "decoder K norm extent mismatch"
    );

    let query_inner = checked_product(shape.query_heads, shape.head_width, "decoder Q width")?;
    let kv_inner = checked_product(shape.kv_heads, shape.head_width, "decoder KV width")?;
    let q_channels = checked_product(query_inner, 2, "decoder Q/gate channels")?;
    validate_count(rows, q_channels, "decoder Q/gate output")?;
    validate_count(rows, kv_inner, "decoder KV output")?;
    validate_count(rows, query_inner, "decoder attention output")?;
    validate_fp8(&weights.q, shape.hidden, q_channels, "Q")?;
    validate_fp8(&weights.k, shape.hidden, kv_inner, "K")?;
    validate_fp8(&weights.v, shape.hidden, kv_inner, "V")?;
    validate_fp8(&weights.out, query_inner, shape.hidden, "attention output")?;

    let (intermediate, up_channels, down_channels) = mlp_channels(&weights.mlp);
    ensure!(
        intermediate == shape.intermediate && up_channels == intermediate,
        "decoder MLP gate/up widths do not match the model shape"
    );
    ensure!(
        down_channels == shape.hidden,
        "decoder MLP output width must equal hidden width"
    );
    validate_count(rows, intermediate, "decoder MLP output")?;

    Ok(Dimensions {
        hidden: shape.hidden,
        query_inner,
    })
}

fn mlp_channels(weights: &decoder_mlp_reference::Weights) -> (usize, usize, usize) {
    match weights {
        decoder_mlp_reference::Weights::Nvfp4(weights) => (
            weights.gate.channels,
            weights.up.channels,
            weights.down.channels,
        ),
        decoder_mlp_reference::Weights::Fp8(weights) => (
            weights.gate.channels,
            weights.up.channels,
            weights.down.channels,
        ),
    }
}

fn validate_fp8(
    projection: &kernels::Fp8Projection,
    input_width: usize,
    channels: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        projection.channels == channels,
        "decoder {label} channel mismatch"
    );
    let elements = checked_product(channels, input_width, label)?;
    ensure!(
        elements <= MAX_ELEMENTS,
        "decoder {label} weights exceed element limit"
    );
    ensure!(
        projection.weights.len() == elements,
        "decoder {label} weight extent mismatch"
    );
    let scale_bytes = checked_product(channels, 2, label)?;
    ensure!(
        projection.scales.len() == scale_bytes,
        "decoder {label} scale extent mismatch"
    );
    Ok(())
}

fn validate_count(rows: usize, channels: usize, label: &str) -> Result<()> {
    ensure!(
        checked_product(rows, channels, label)? <= MAX_ELEMENTS,
        "decoder {label} exceeds element limit"
    );
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("decoder {label} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{Weights, run, run_observed};
    use crate::{
        decoder_mlp_reference,
        fp8_mlp_reference::{self, Projection},
        kernels::{Fp8Projection, ResidentAttentionShape},
    };

    fn fp8_projection(input_width: usize, channels: usize) -> Fp8Projection {
        let mut scales = Vec::with_capacity(channels * 2);
        for _ in 0..channels {
            scales.extend_from_slice(&0x3f80_u16.to_le_bytes());
        }
        Fp8Projection {
            name: "zero".to_owned(),
            weights: vec![0; input_width * channels],
            scales,
            channels,
        }
    }

    fn mlp_projection(input_width: usize, channels: usize) -> Projection {
        Projection {
            weights: vec![0; input_width * channels],
            scales: vec![0x3f80; channels],
            channels,
        }
    }

    fn zero_branch_fixture() -> (Vec<u16>, ResidentAttentionShape, Weights) {
        let rows = 2;
        let hidden_width = 16;
        let query_inner = 2 * 8;
        let kv_inner = 8;
        let intermediate = 16;
        let shape = ResidentAttentionShape {
            hidden: hidden_width,
            intermediate,
            query_heads: 2,
            kv_heads: 1,
            head_width: 8,
            rotary_dim: 4,
            rope_theta: 10_000.0,
        };
        let mlp = decoder_mlp_reference::Weights::Fp8(fp8_mlp_reference::Weights {
            gate: mlp_projection(hidden_width, intermediate),
            up: mlp_projection(hidden_width, intermediate),
            down: mlp_projection(intermediate, hidden_width),
        });
        let weights = Weights {
            input_norm: vec![0xbf80; hidden_width],
            post_norm: vec![0xbf80; hidden_width],
            q: fp8_projection(hidden_width, query_inner * 2),
            k: fp8_projection(hidden_width, kv_inner),
            v: fp8_projection(hidden_width, kv_inner),
            q_norm: vec![0; 8],
            k_norm: vec![0; 8],
            out: fp8_projection(query_inner, hidden_width),
            mlp,
        };
        let hidden = vec![0x3f80; rows * hidden_width];
        (hidden, shape, weights)
    }

    #[test]
    fn zero_branches_preserve_each_input_residual_row() {
        let (hidden, shape, weights) = zero_branch_fixture();
        assert_eq!(run(&hidden, 2, &shape, &weights).unwrap(), hidden);
    }

    #[test]
    fn observer_receives_decoder_boundaries_in_contract_order() {
        let (hidden, shape, weights) = zero_branch_fixture();
        let expected = run(&hidden, 2, &shape, &weights).unwrap();
        let mut stages = Vec::new();
        let actual = run_observed(&hidden, 2, &shape, &weights, &mut |stage, _| {
            stages.push(stage.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            stages,
            [
                "normalized",
                "q_linear",
                "k_linear",
                "v_linear",
                "q_prepared",
                "q_gate",
                "k_prepared",
                "attended",
                "gated",
                "out",
                "post_residual",
                "post_norm",
                "mlp_gate",
                "mlp_up",
                "mlp_activation",
                "mlp_down",
                "hidden",
            ]
        );
    }

    #[test]
    fn rejects_invalid_rows_dimensions_channels_and_extents() {
        let (hidden, shape, weights) = zero_branch_fixture();
        assert!(run(&hidden, 0, &shape, &weights).is_err());
        assert!(run(&hidden[..15], 1, &shape, &weights).is_err());

        let mut bad_shape = ResidentAttentionShape {
            hidden: shape.hidden,
            intermediate: shape.intermediate,
            query_heads: 3,
            kv_heads: 2,
            head_width: shape.head_width,
            rotary_dim: shape.rotary_dim,
            rope_theta: shape.rope_theta,
        };
        assert!(run(&hidden, 2, &bad_shape, &weights).is_err());
        bad_shape.query_heads = shape.query_heads;
        bad_shape.rotary_dim = 3;
        assert!(run(&hidden, 2, &bad_shape, &weights).is_err());

        let mut bad_weights = zero_branch_fixture().2;
        bad_weights.q.channels += 1;
        assert!(run(&hidden, 2, &shape, &bad_weights).is_err());
        let mut bad_weights = zero_branch_fixture().2;
        bad_weights.input_norm.pop();
        assert!(run(&hidden, 2, &shape, &bad_weights).is_err());
    }
}
