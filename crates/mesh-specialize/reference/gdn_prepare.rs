//! Independent high-precision reference for GDN Q/K normalization and gate preparation.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
}

#[derive(Debug)]
pub struct Prepared {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub beta: Vec<u16>,
    pub g: Vec<f32>,
    pub decay: Vec<f32>,
}

/// Prepare row-major Q/K vectors and per-value-head GDN gates from BF16 inputs.
///
/// QKV rows are ordered as all Q heads, then all K heads, then all V heads.
/// Returned Q/K vectors keep only the key-head count; no key repetition or GPU
/// packing is applied. Norm reductions use FP64 while returned arithmetic is FP32.
pub fn run(
    qkv: &[u16],
    a: &[u16],
    b: &[u16],
    a_log: &[u16],
    dt_bias: &[u16],
    shape: &Shape,
) -> Result<Prepared> {
    validate_shape(shape)?;
    let key_width = shape
        .key_heads
        .checked_mul(shape.width)
        .context("GDN key width overflows usize")?;
    let doubled_key_heads = shape
        .key_heads
        .checked_mul(2)
        .context("GDN Q/K head count overflows usize")?;
    let row_heads = doubled_key_heads
        .checked_add(shape.value_heads)
        .context("GDN row head count overflows usize")?;
    let row_width = row_heads
        .checked_mul(shape.width)
        .context("GDN QKV row width overflows usize")?;
    let qkv_len = shape
        .rows
        .checked_mul(row_width)
        .context("GDN QKV length overflows usize")?;
    let gate_len = shape
        .rows
        .checked_mul(shape.value_heads)
        .context("GDN gate length overflows usize")?;
    let key_len = shape
        .rows
        .checked_mul(key_width)
        .context("GDN Q/K output length overflows usize")?;

    ensure!(qkv.len() == qkv_len, "GDN QKV extent mismatch");
    ensure!(a.len() == gate_len, "GDN A extent mismatch");
    ensure!(b.len() == gate_len, "GDN B extent mismatch");
    ensure!(
        a_log.len() == shape.value_heads,
        "GDN A-log extent mismatch"
    );
    ensure!(
        dt_bias.len() == shape.value_heads,
        "GDN dt-bias extent mismatch"
    );
    validate_finite(qkv, "QKV")?;
    validate_finite(a, "A")?;
    validate_finite(b, "B")?;
    validate_finite(a_log, "A-log")?;
    validate_finite(dt_bias, "dt-bias")?;

    let log_values: Vec<_> = a_log.iter().map(|&bits| bf16_to_f32(bits)).collect();
    ensure!(
        log_values
            .iter()
            .all(|value| (-80.0..=80.0).contains(value)),
        "GDN A-log values must be in [-80, 80]"
    );

    let mut prepared = Prepared {
        q: Vec::with_capacity(key_len),
        k: Vec::with_capacity(key_len),
        beta: Vec::with_capacity(gate_len),
        g: Vec::with_capacity(gate_len),
        decay: Vec::with_capacity(gate_len),
    };
    for row in 0..shape.rows {
        let row_start = row * row_width;
        let qkv_row = &qkv[row_start..row_start + row_width];
        for head in 0..shape.key_heads {
            let start = head * shape.width;
            normalize_head(qkv_row, start, shape.width, true, &mut prepared.q)?;
            normalize_head(
                qkv_row,
                key_width + start,
                shape.width,
                false,
                &mut prepared.k,
            )?;
        }
        for head in 0..shape.value_heads {
            let index = row * shape.value_heads + head;
            let a_value = bf16_to_f32(a[index]);
            let b_value = bf16_to_f32(b[index]);
            let (beta, g, decay) = prepare_gate(
                a_value,
                b_value,
                log_values[head],
                bf16_to_f32(dt_bias[head]),
            )?;
            prepared.beta.push(beta);
            prepared.g.push(g);
            prepared.decay.push(decay);
        }
    }
    Ok(prepared)
}

fn validate_shape(shape: &Shape) -> Result<()> {
    ensure!((1..=2048).contains(&shape.rows), "invalid GDN row count");
    ensure!(
        (1..=64).contains(&shape.key_heads),
        "invalid GDN key head count"
    );
    ensure!(
        (1..=256).contains(&shape.value_heads),
        "invalid GDN value head count"
    );
    ensure!(
        shape.value_heads.is_multiple_of(shape.key_heads),
        "GDN value heads must be divisible by key heads"
    );
    ensure!(
        (1..=256).contains(&shape.width) && shape.width.is_power_of_two(),
        "GDN width must be a power of two in 1..=256"
    );
    Ok(())
}

fn validate_finite(values: &[u16], label: &str) -> Result<()> {
    ensure!(
        values.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "GDN {label} values must be finite BF16 values"
    );
    Ok(())
}

fn normalize_head(
    qkv_row: &[u16],
    start: usize,
    width: usize,
    query: bool,
    output: &mut Vec<f32>,
) -> Result<()> {
    let values = &qkv_row[start..start + width];
    let norm_sum = values.iter().try_fold(0.0_f64, |sum, &bits| {
        let value = f64::from(bf16_to_f32(bits));
        let next = sum + value * value;
        if next.is_finite() {
            Ok(next)
        } else {
            anyhow::bail!("GDN Q/K norm sum overflows FP64")
        }
    })?;
    let denominator = (norm_sum + 1e-6_f64) as f32;
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "GDN Q/K norm denominator overflows FP32"
    );
    let inverse_norm = 1.0_f32 / denominator.sqrt();
    let width_sqrt = (width as f32).sqrt();
    ensure!(
        inverse_norm.is_finite() && width_sqrt.is_finite() && width_sqrt > 0.0,
        "GDN Q/K normalization factor is invalid"
    );
    for &bits in values {
        let mut normalized = bf16_to_f32(bits) * inverse_norm;
        if query {
            normalized /= width_sqrt;
        }
        ensure!(
            normalized.is_finite(),
            "GDN normalized Q/K value overflows FP32"
        );
        output.push(normalized);
    }
    Ok(())
}

pub(crate) fn prepare_gate(a: f32, b: f32, a_log: f32, dt_bias: f32) -> Result<(u16, f32, f32)> {
    let beta = round_bf16(sigmoid(b));
    let beta_value = bf16_to_f32(beta);
    ensure!(
        beta_value.is_finite() && (0.0..=1.0).contains(&beta_value),
        "GDN beta is outside [0, 1]"
    );

    let x = a + dt_bias;
    ensure!(x.is_finite(), "GDN A plus dt-bias overflows FP32");
    let x64 = f64::from(x);
    let softplus64 = if x > 20.0_f32 {
        x64
    } else {
        x64.max(0.0) + (-x64.abs()).exp().ln_1p()
    };
    let softplus = softplus64 as f32;
    ensure!(softplus.is_finite(), "GDN softplus overflows FP32");

    let g = (-f64::from(a_log).exp() * f64::from(softplus)) as f32;
    ensure!(g.is_finite() && g <= 0.0, "GDN decay gate is invalid");
    let decay = f64::from(g).exp() as f32;
    ensure!(
        decay.is_finite() && (0.0..=1.0).contains(&decay),
        "GDN decay is outside [0, 1]"
    );
    Ok((beta, g, decay))
}

fn sigmoid(value: f32) -> f32 {
    let value64 = f64::from(value);
    let exponential = (-value64.abs()).exp();
    let result = if value64 >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    result as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    fn shape(rows: usize, key_heads: usize, value_heads: usize, width: usize) -> Shape {
        Shape {
            rows,
            key_heads,
            value_heads,
            width,
        }
    }

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn normalizes_q_and_k_independently_and_maps_value_head_gates() {
        let qkv = [
            bf16(3.0),
            bf16(4.0),
            bf16(0.0),
            bf16(0.0),
            bf16(3.0),
            bf16(4.0),
            bf16(1.0),
            bf16(0.0),
            bf16(11.0),
            bf16(12.0),
            bf16(21.0),
            bf16(22.0),
            bf16(31.0),
            bf16(32.0),
            bf16(41.0),
            bf16(42.0),
        ];
        let a = [bf16(0.0), bf16(-1.0), bf16(1.0), bf16(2.0)];
        let b = [bf16(-2.0), bf16(-1.0), bf16(1.0), bf16(2.0)];
        let zeros = [bf16(0.0); 4];
        let prepared = run(&qkv, &a, &b, &zeros, &zeros, &shape(1, 2, 4, 2)).unwrap();

        let query_scale = 1.0 / (25.0_f32 + 1e-6).sqrt() / 2.0_f32.sqrt();
        let key_scale = 1.0 / (25.0_f32 + 1e-6).sqrt();
        close(prepared.q[0], 3.0 * query_scale, 1e-6);
        close(prepared.q[1], 4.0 * query_scale, 1e-6);
        assert_eq!(&prepared.q[2..], &[0.0, 0.0]);
        close(prepared.k[0], 3.0 * key_scale, 1e-6);
        close(prepared.k[1], 4.0 * key_scale, 1e-6);
        close(prepared.k[2], 1.0 / (1.0_f32 + 1e-6).sqrt(), 1e-6);
        assert_eq!(prepared.k[3], 0.0);
        assert_eq!(
            prepared.beta,
            b.map(|bits| round_bf16(sigmoid(bf16_to_f32(bits))))
        );
        assert_eq!(prepared.g.len(), 4);
        assert!(prepared.g.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn zero_vectors_and_neutral_gate_have_expected_values() {
        let qkv = [0_u16; 6];
        let one = [bf16(0.0)];
        let prepared = run(&qkv, &one, &one, &one, &one, &shape(1, 1, 1, 2)).unwrap();
        assert_eq!(prepared.q, [0.0, 0.0]);
        assert_eq!(prepared.k, [0.0, 0.0]);
        assert_eq!(prepared.beta, [bf16(0.5)]);
        close(prepared.g[0], -std::f32::consts::LN_2, 1e-6);
        close(prepared.decay[0], 0.5, 1e-6);
    }

    #[test]
    fn gate_threshold_extremes_and_f64_underflow_remain_bounded() {
        let qkv = [0_u16; 6];
        let a = [bf16(20.0), bf16(21.0), 0xff7f, bf16(1.0)];
        let b = [0x7f7f, 0xff7f, 0x7f7f, 0xff7f];
        let a_log = [bf16(0.0), bf16(0.0), bf16(0.0), bf16(-80.0)];
        let dt_bias = [bf16(0.0); 4];
        let prepared = run(&qkv, &a, &b, &a_log, &dt_bias, &shape(1, 1, 4, 1)).unwrap();

        assert_eq!(prepared.beta, [bf16(1.0), bf16(0.0), bf16(1.0), bf16(0.0)]);
        close(prepared.g[0], -20.0, 1e-5);
        close(prepared.g[1], -21.0, 1e-5);
        assert_eq!(prepared.g[2], 0.0);
        assert_eq!(prepared.decay[2], 1.0);
        assert_eq!(prepared.decay[3], 1.0);
        assert!(
            prepared
                .g
                .iter()
                .all(|value| value.is_finite() && *value <= 0.0)
        );
        assert!(
            prepared
                .decay
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn rejects_invalid_shapes_extents_nonfinite_values_and_overflow() {
        let valid_shape = shape(1, 1, 1, 1);
        let qkv = [0_u16; 3];
        let one = [bf16(1.0)];
        let zero = [bf16(0.0)];
        assert!(run(&qkv, &one, &one, &one, &one, &shape(0, 1, 1, 1)).is_err());
        assert!(run(&qkv, &one, &one, &one, &one, &shape(1, 0, 1, 1)).is_err());
        assert!(run(&qkv, &one, &one, &one, &one, &shape(1, 2, 3, 1)).is_err());
        assert!(run(&qkv, &one, &one, &one, &one, &shape(1, 1, 1, 3)).is_err());
        assert!(run(&qkv[..2], &one, &one, &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &[], &one, &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &[], &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &one, &[], &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &one, &zero, &[], &valid_shape).is_err());
        assert!(run(&[0x7f80, 0, 0], &one, &one, &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &[0x7fc0], &one, &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &[0x7f80], &zero, &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &one, &[0x7f80], &zero, &valid_shape).is_err());
        assert!(run(&qkv, &one, &one, &zero, &[0x7fc0], &valid_shape).is_err());
        assert!(run(&qkv, &one, &one, &[bf16(81.0)], &zero, &valid_shape).is_err());

        let q_overflow = [0x7f7f, 0x7f7f, 0];
        assert!(run(&q_overflow, &one, &one, &zero, &zero, &shape(1, 1, 1, 1)).is_err());
        let gate_overflow = [bf16(0.0), bf16(0.0), bf16(0.0)];
        assert!(
            run(
                &gate_overflow,
                &[0x7f7f],
                &one,
                &zero,
                &[0x7f7f],
                &valid_shape
            )
            .is_err()
        );
    }
}
