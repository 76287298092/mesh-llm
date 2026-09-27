//! Independent CPU reference for attention Q/K normalization, RoPE, and Q gates.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

const MAX_ELEMENTS: usize = 67_108_864;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub rows: usize,
    pub heads: usize,
    pub width: usize,
    pub rotary_dim: usize,
    pub with_gate: bool,
}

#[derive(Debug, PartialEq)]
pub struct Prepared {
    /// BF16 Q/K after rotary embedding, laid out `[rows, heads, width]`.
    pub output: Vec<u16>,
    /// BF16 zero-centered RMSNorm output before rotary embedding.
    pub normalized: Vec<u16>,
    /// FP32 zero-centered RMSNorm output before BF16 rounding.
    pub unrounded: Vec<f32>,
    /// Raw BF16 Q gate values, laid out `[rows, heads, width]` (empty for K).
    pub gate: Vec<u16>,
}

/// Normalize each head independently, preserve optional Q gates, then apply RoPE.
///
/// Input is row-major `[rows, heads, width * (with_gate ? 2 : 1)]`; for each
/// head its first `width` values are Q/K and its next `width` values are the Q
/// gate. `weight` is the shared zero-centered BF16 RMSNorm parameter. The
/// compact cosine and sine tables contain `[rows, rotary_dim / 2]` BF16 values.
pub fn run(
    input: &[u16],
    weight: &[u16],
    cos: &[u16],
    sin: &[u16],
    shape: &Shape,
    epsilon: f32,
) -> Result<Prepared> {
    let lengths = validate_inputs(input, weight, cos, sin, shape, Some(epsilon))?;
    let mut prepared = PreparedBuffers {
        normalized: Vec::with_capacity(lengths.output_len),
        unrounded: Vec::with_capacity(lengths.output_len),
        gate: Vec::with_capacity(if shape.with_gate {
            lengths.output_len
        } else {
            0
        }),
    };

    for row in 0..shape.rows {
        for head in 0..shape.heads {
            normalize_head(input, weight, row, head, shape, epsilon, &mut prepared)?;
        }
    }
    let output = rotate(&prepared.normalized, cos, sin, shape)?;
    Ok(Prepared {
        output,
        normalized: prepared.normalized,
        unrounded: prepared.unrounded,
        gate: prepared.gate,
    })
}

/// Apply split-half rotary embedding to compact BF16-normalized heads.
///
/// Products are each rounded to BF16 before their FP32 sum is rounded to BF16,
/// matching the explicit intermediate boundaries in the reference profile.
pub fn rotate(normalized: &[u16], cos: &[u16], sin: &[u16], shape: &Shape) -> Result<Vec<u16>> {
    let lengths = validate_shape(shape, None)?;
    ensure!(
        normalized.len() == lengths.output_len,
        "attention normalized extent mismatch"
    );
    validate_tables(cos, sin, lengths.table_len)?;
    ensure!(
        normalized.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention normalized values must be finite BF16 values"
    );

    let mut output = Vec::with_capacity(lengths.output_len);
    for row in 0..shape.rows {
        for head in 0..shape.heads {
            rotate_head(normalized, cos, sin, row, head, shape, &mut output)?;
        }
    }
    debug_assert_eq!(output.len(), lengths.output_len);
    Ok(output)
}

/// Build compact BF16 text RoPE tables using the documented CPU reference profile.
///
/// This calculation is intentionally independent of GPU table generation and
/// does not claim bitwise agreement with framework-specific trigonometric kernels.
pub fn text_rope_tables(
    positions: &[u32],
    rotary_dim: usize,
    theta: f32,
) -> Result<(Vec<u16>, Vec<u16>)> {
    ensure!(
        !positions.is_empty() && positions.len() <= 2048,
        "invalid RoPE row count"
    );
    ensure!(
        positions.iter().all(|&position| position <= 262_143),
        "RoPE position exceeds supported range"
    );
    ensure!(
        (2..=1024).contains(&rotary_dim) && rotary_dim.is_multiple_of(2),
        "rotary dimension must be even and in 2..=1024"
    );
    ensure!(theta.is_finite() && theta > 1.0, "invalid RoPE theta");

    let half = rotary_dim / 2;
    let table_len = positions
        .len()
        .checked_mul(half)
        .context("RoPE table length overflows usize")?;
    let mut cos = Vec::with_capacity(table_len);
    let mut sin = Vec::with_capacity(table_len);
    for &position in positions {
        for frequency in 0..half {
            let exponent = (2 * frequency) as f64 / rotary_dim as f64;
            let denominator = f64::from(theta).powf(exponent) as f32;
            ensure!(
                denominator.is_finite() && denominator > 0.0,
                "RoPE frequency denominator is invalid"
            );
            let inverse = 1.0_f32 / denominator;
            let angle = position as f32 * inverse;
            ensure!(angle.is_finite(), "RoPE angle is nonfinite");
            cos.push(round_finite_bf16(
                (f64::from(angle)).cos() as f32,
                "RoPE cosine",
            )?);
            sin.push(round_finite_bf16(
                (f64::from(angle)).sin() as f32,
                "RoPE sine",
            )?);
        }
    }
    Ok((cos, sin))
}

struct Lengths {
    output_len: usize,
    input_len: usize,
    table_len: usize,
}

struct PreparedBuffers {
    normalized: Vec<u16>,
    unrounded: Vec<f32>,
    gate: Vec<u16>,
}

fn validate_inputs(
    input: &[u16],
    weight: &[u16],
    cos: &[u16],
    sin: &[u16],
    shape: &Shape,
    epsilon: Option<f32>,
) -> Result<Lengths> {
    let lengths = validate_shape(shape, epsilon)?;
    ensure!(
        input.len() == lengths.input_len,
        "attention input extent mismatch"
    );
    ensure!(
        weight.len() == shape.width,
        "attention norm weight extent mismatch"
    );
    validate_tables(cos, sin, lengths.table_len)?;
    ensure!(
        weight.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention norm weights must be finite BF16 values"
    );
    ensure!(
        input.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention input values must be finite BF16 values"
    );
    Ok(lengths)
}

fn validate_shape(shape: &Shape, epsilon: Option<f32>) -> Result<Lengths> {
    ensure!(
        (1..=2048).contains(&shape.rows),
        "invalid attention row count"
    );
    ensure!(
        (1..=128).contains(&shape.heads),
        "invalid attention head count"
    );
    ensure!(
        (2..=1024).contains(&shape.width),
        "invalid attention head width"
    );
    ensure!(
        shape.rotary_dim > 0
            && shape.rotary_dim <= shape.width
            && shape.rotary_dim.is_multiple_of(2),
        "rotary dimension must be positive, even, and no greater than head width"
    );
    if let Some(epsilon) = epsilon {
        ensure!(
            epsilon.is_finite() && epsilon > 0.0,
            "invalid RMSNorm epsilon"
        );
    }

    let output_len = checked_product(&[shape.rows, shape.heads, shape.width], "attention output")?;
    ensure!(
        output_len <= MAX_ELEMENTS,
        "attention output exceeds element limit"
    );
    let input_width = shape
        .width
        .checked_mul(if shape.with_gate { 2 } else { 1 })
        .context("attention input row width overflows usize")?;
    let input_len = checked_product(&[shape.rows, shape.heads, input_width], "attention input")?;
    ensure!(
        input_len <= MAX_ELEMENTS,
        "attention input exceeds element limit"
    );
    let table_len = checked_product(&[shape.rows, shape.rotary_dim / 2], "attention RoPE table")?;
    Ok(Lengths {
        output_len,
        input_len,
        table_len,
    })
}

fn validate_tables(cos: &[u16], sin: &[u16], table_len: usize) -> Result<()> {
    ensure!(cos.len() == table_len, "attention cosine extent mismatch");
    ensure!(sin.len() == table_len, "attention sine extent mismatch");
    ensure!(
        cos.iter()
            .chain(sin)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention RoPE values must be finite BF16 values"
    );
    Ok(())
}

fn normalize_head(
    input: &[u16],
    weight: &[u16],
    row: usize,
    head: usize,
    shape: &Shape,
    epsilon: f32,
    prepared: &mut PreparedBuffers,
) -> Result<()> {
    let input_width = shape.width * if shape.with_gate { 2 } else { 1 };
    let input_start = (row * shape.heads + head) * input_width;
    let head_values = &input[input_start..input_start + shape.width];
    let squares: f64 = head_values
        .iter()
        .map(|&bits| {
            let value = f64::from(bf16_to_f32(bits));
            value * value
        })
        .sum();
    let mean = squares / shape.width as f64;
    let variance = mean + f64::from(epsilon);
    ensure!(
        variance.is_finite() && variance > 0.0,
        "attention variance is invalid"
    );
    let inverse = 1.0_f64 / variance.sqrt();

    for (column, &bits) in head_values.iter().enumerate() {
        let value = f64::from(bf16_to_f32(bits));
        let gamma = f64::from(bf16_to_f32(weight[column]));
        let result = value * inverse * (1.0 + gamma);
        let result_f32 = result as f32;
        ensure!(result_f32.is_finite(), "attention norm output is nonfinite");
        let rounded = round_finite_bf16(result_f32, "attention norm output")?;
        prepared.unrounded.push(result_f32);
        prepared.normalized.push(rounded);
    }
    if shape.with_gate {
        let gate_start = input_start + shape.width;
        prepared
            .gate
            .extend_from_slice(&input[gate_start..gate_start + shape.width]);
    }
    Ok(())
}

fn rotate_head(
    normalized: &[u16],
    cos: &[u16],
    sin: &[u16],
    row: usize,
    head: usize,
    shape: &Shape,
    output: &mut Vec<u16>,
) -> Result<()> {
    let half = shape.rotary_dim / 2;
    let head_start = (row * shape.heads + head) * shape.width;
    let table_start = row * half;
    for column in 0..shape.width {
        if column >= shape.rotary_dim {
            output.push(normalized[head_start + column]);
            continue;
        }
        let pair = if column < half {
            column + half
        } else {
            column - half
        };
        let sign = if column < half { -1.0_f32 } else { 1.0_f32 };
        let value = bf16_to_f32(normalized[head_start + column]);
        let paired = bf16_to_f32(normalized[head_start + pair]) * sign;
        let cosine = bf16_to_f32(cos[table_start + column % half]);
        let sine = bf16_to_f32(sin[table_start + column % half]);
        let cosine_product = round_finite_bf16(value * cosine, "RoPE cosine product")?;
        let sine_product = round_finite_bf16(paired * sine, "RoPE sine product")?;
        let sum = bf16_to_f32(cosine_product) + bf16_to_f32(sine_product);
        output.push(round_finite_bf16(sum, "RoPE output")?);
    }
    Ok(())
}

fn round_finite_bf16(value: f32, label: &str) -> Result<u16> {
    ensure!(value.is_finite(), "{label} is nonfinite");
    let rounded = round_bf16(value);
    ensure!(bf16_to_f32(rounded).is_finite(), "{label} overflows BF16");
    Ok(rounded)
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .with_context(|| format!("{label} length overflows usize"))
    })
}

#[cfg(test)]
mod tests {
    use super::{Shape, rotate, round_finite_bf16, run, text_rope_tables};
    use crate::entry_reference::{bf16_to_f32, round_bf16};

    fn shape(rows: usize, heads: usize, width: usize, rotary_dim: usize, with_gate: bool) -> Shape {
        Shape {
            rows,
            heads,
            width,
            rotary_dim,
            with_gate,
        }
    }

    fn zero_position_tables(rows: usize, rotary_dim: usize) -> (Vec<u16>, Vec<u16>) {
        let half = rotary_dim / 2;
        (
            vec![round_bf16(1.0); rows * half],
            vec![round_bf16(0.0); rows * half],
        )
    }

    #[test]
    fn hand_computed_norm_uses_f64_and_zero_centered_weight() {
        let shape = shape(1, 1, 2, 2, false);
        let (cos, sin) = zero_position_tables(1, 2);
        let result = run(
            &[round_bf16(3.0), round_bf16(4.0)],
            &[round_bf16(0.0), round_bf16(-1.0)],
            &cos,
            &sin,
            &shape,
            3.5,
        )
        .unwrap();
        assert_eq!(result.unrounded, [0.75, 0.0]);
        assert_eq!(result.normalized, [round_bf16(0.75), round_bf16(0.0)]);
        assert_eq!(result.output, result.normalized);
        assert!(result.gate.is_empty());
    }

    #[test]
    fn q_gate_is_extracted_per_head_without_reordering() {
        let shape = shape(1, 2, 2, 2, true);
        let (cos, sin) = zero_position_tables(1, 2);
        let input = [
            round_bf16(1.0),
            round_bf16(2.0),
            round_bf16(11.0),
            round_bf16(12.0),
            round_bf16(3.0),
            round_bf16(4.0),
            round_bf16(13.0),
            round_bf16(14.0),
        ];
        let result = run(
            &input,
            &[round_bf16(0.0), round_bf16(0.0)],
            &cos,
            &sin,
            &shape,
            1.0,
        )
        .unwrap();
        assert_eq!(
            result.gate,
            input[2..4]
                .iter()
                .chain(&input[6..8])
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(result.output.len(), 4);
        assert_eq!(result.normalized.len(), 4);
    }

    #[test]
    fn split_half_rope_rotates_signs_and_preserves_pass_through() {
        let shape = shape(1, 1, 6, 4, false);
        let normalized = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(round_bf16);
        let cos = [0.0, 0.0].map(round_bf16);
        let sin = [1.0, 1.0].map(round_bf16);
        let output = rotate(&normalized, &cos, &sin, &shape).unwrap();
        assert_eq!(output, [-3.0, -4.0, 1.0, 2.0, 5.0, 6.0].map(round_bf16));
    }

    #[test]
    fn rotary_products_round_before_their_sum() {
        let shape = shape(1, 1, 2, 2, false);
        let normalized = [round_bf16(1.0078125), round_bf16(0.75)];
        let cos = [round_bf16(0.75)];
        let sin = [round_bf16(1.0)];
        let output = rotate(&normalized, &cos, &sin, &shape).unwrap();
        let cosine_product = round_bf16(bf16_to_f32(normalized[0]) * bf16_to_f32(cos[0]));
        let sine_product = round_bf16(-bf16_to_f32(normalized[1]) * bf16_to_f32(sin[0]));
        let expected = round_bf16(0.0078125);
        assert_eq!(
            output[0],
            round_bf16(bf16_to_f32(cosine_product) + bf16_to_f32(sine_product))
        );
        assert_eq!(output[0], expected);
        assert_ne!(output[0], round_bf16(0.005859375));
    }

    #[test]
    fn text_tables_anchor_first_frequency_and_keep_high_positions_finite() {
        let (cos, sin) = text_rope_tables(&[1, 262_143], 4, 10_000.0).unwrap();
        assert_eq!(cos[0], round_bf16(1.0_f64.cos() as f32));
        assert_eq!(sin[0], round_bf16(1.0_f64.sin() as f32));
        let second_angle = f64::from(0.01_f32);
        assert_eq!(cos[1], round_bf16(second_angle.cos() as f32));
        assert_eq!(sin[1], round_bf16(second_angle.sin() as f32));
        assert!(
            cos.iter()
                .chain(&sin)
                .all(|&bits| bf16_to_f32(bits).is_finite())
        );
    }

    #[test]
    fn rejects_invalid_shapes_extents_and_nonfinite_values() {
        let valid_shape = shape(1, 1, 2, 2, false);
        let (cos, sin) = zero_position_tables(1, 2);
        let valid = run(
            &[round_bf16(1.0), round_bf16(2.0)],
            &[0, 0],
            &cos,
            &sin,
            &valid_shape,
            1e-6,
        );
        assert!(valid.is_ok());
        assert!(run(&[0], &[0, 0], &cos, &sin, &valid_shape, 1e-6).is_err());
        assert!(run(&[0, 0], &[0], &cos, &sin, &valid_shape, 1e-6).is_err());
        assert!(run(&[0, 0], &[0, 0], &[], &sin, &valid_shape, 1e-6).is_err());
        assert!(run(&[0, 0], &[0, 0], &cos, &sin, &valid_shape, f32::NAN).is_err());
        assert!(run(&[0x7f80, 0], &[0, 0], &cos, &sin, &valid_shape, 1e-6).is_err());
        assert!(run(&[0, 0], &[0x7f80, 0], &cos, &sin, &valid_shape, 1e-6).is_err());
        assert!(run(&[0, 0], &[0, 0], &[0x7f80], &sin, &valid_shape, 1e-6).is_err());
        assert!(round_finite_bf16(f32::MAX, "test BF16 overflow").is_err());
        assert!(
            run(
                &[0, 0],
                &[0, 0],
                &cos,
                &sin,
                &shape(1, 1, 2, 0, false),
                1e-6
            )
            .is_err()
        );
        assert!(text_rope_tables(&[], 2, 10_000.0).is_err());
        assert!(text_rope_tables(&[1], 3, 10_000.0).is_err());
        assert!(text_rope_tables(&[262_144], 2, 10_000.0).is_err());
    }
}
