//! Independent logical reference for Qwen3.5's BF16 gated RMS normalization.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

#[derive(Debug, PartialEq)]
pub struct GatedNorm {
    pub normalized: Vec<f32>,
    pub weighted: Vec<u16>,
    pub silu: Vec<f32>,
    pub unrounded: Vec<f32>,
    pub output: Vec<u16>,
}

/// Normalize each logical group, round to BF16 before applying gamma, then gate.
///
/// This mirrors the pinned Transformers operation order: RMS norm precedes the
/// direct learned weight, the normalized activation is BF16-rounded before that
/// multiply, and the final gated result is BF16-rounded once more.
pub fn run(
    x: &[u16],
    z: &[u16],
    weight: &[u16],
    groups: usize,
    width: usize,
    epsilon: f32,
) -> Result<GatedNorm> {
    let extent = validate(x, z, weight, groups, width, epsilon)?;
    let mut result = GatedNorm {
        normalized: Vec::with_capacity(extent),
        weighted: Vec::with_capacity(extent),
        silu: Vec::with_capacity(extent),
        unrounded: Vec::with_capacity(extent),
        output: Vec::with_capacity(extent),
    };

    for group in 0..groups {
        let start = group * width;
        let end = start + width;
        let inverse = inverse_rms(&x[start..end], width, epsilon)?;
        for index in start..end {
            let normalized = checked_mul(bf16_to_f32(x[index]), inverse, "normalized activation")?;
            let normalized_bf16 = round_bf16(normalized);
            ensure!(
                bf16_to_f32(normalized_bf16).is_finite(),
                "normalized activation overflows BF16"
            );
            let weighted = checked_mul(
                bf16_to_f32(normalized_bf16),
                bf16_to_f32(weight[index - start]),
                "normalized activation times gamma",
            )?;
            let weighted_bf16 = round_bf16(weighted);
            let weighted_value = bf16_to_f32(weighted_bf16);
            ensure!(
                weighted_value.is_finite(),
                "weighted activation overflows BF16"
            );
            let gate = silu(bf16_to_f32(z[index]))?;
            let unrounded = checked_mul(weighted_value, gate, "weighted activation times SiLU")?;
            let output = round_bf16(unrounded);
            ensure!(
                bf16_to_f32(output).is_finite(),
                "gated output overflows BF16"
            );

            result.normalized.push(normalized);
            result.weighted.push(weighted_bf16);
            result.silu.push(gate);
            result.unrounded.push(unrounded);
            result.output.push(output);
        }
    }
    Ok(result)
}

/// Compute stable SiLU in FP64, returning one checked FP32 result.
pub fn silu(value: f32) -> Result<f32> {
    ensure!(value.is_finite(), "SiLU input must be finite");
    let wide = f64::from(value);
    let exponential = (-wide.abs()).exp();
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    checked_f32((wide * sigmoid) as f32, "SiLU output")
}

fn validate(
    x: &[u16],
    z: &[u16],
    weight: &[u16],
    groups: usize,
    width: usize,
    epsilon: f32,
) -> Result<usize> {
    ensure!(
        (1..=131072).contains(&groups),
        "invalid gated norm group count"
    );
    ensure!(
        (1..=256).contains(&width) && width.is_power_of_two(),
        "gated norm width must be a power of two in 1..=256"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "invalid RMS norm epsilon"
    );
    let extent = groups
        .checked_mul(width)
        .context("gated norm extent overflows usize")?;
    ensure!(x.len() == extent, "gated norm X extent mismatch");
    ensure!(z.len() == extent, "gated norm Z extent mismatch");
    ensure!(weight.len() == width, "gated norm weight extent mismatch");
    ensure!(
        x.iter()
            .chain(z)
            .chain(weight)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "gated norm inputs and gamma must be finite BF16"
    );
    Ok(extent)
}

fn inverse_rms(values: &[u16], width: usize, epsilon: f32) -> Result<f32> {
    let mut square_sum = 0.0_f64;
    for &bits in values {
        let value = f64::from(bf16_to_f32(bits));
        let square = value * value;
        ensure!(
            square.is_finite() && square <= f64::from(f32::MAX),
            "RMS norm square overflows FP32"
        );
        square_sum += square;
        ensure!(
            square_sum.is_finite() && square_sum <= f64::from(f32::MAX),
            "RMS norm sum overflows FP32"
        );
    }
    let mean = checked_f32((square_sum / width as f64) as f32, "RMS norm mean")?;
    let variance = checked_add(mean, epsilon, "RMS norm epsilon addition")?;
    ensure!(variance > 0.0, "RMS norm variance must be positive");
    let root = checked_f32(variance.sqrt(), "RMS norm square root")?;
    ensure!(root > 0.0, "RMS norm square root must be positive");
    checked_f32(1.0 / root, "RMS norm reciprocal")
}

fn checked_mul(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left * right, label)
}

fn checked_add(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left + right, label)
}

fn checked_f32(value: f32, label: &str) -> Result<f32> {
    ensure!(value.is_finite(), "{label} overflows FP32");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn width_one_zero_gate_keeps_direct_signed_gamma_result() {
        let result = run(&[bf16(1.0)], &[bf16(0.0)], &[bf16(-2.0)], 1, 1, 1e-6).unwrap();
        close(result.normalized[0], 1.0, 1e-6);
        assert_eq!(result.weighted, [bf16(-2.0)]);
        assert_eq!(result.silu, [0.0]);
        assert_eq!(result.unrounded, [-0.0]);
        assert_eq!(result.output, [0x8000]);
    }

    #[test]
    fn rms_norm_precedes_gate_and_gamma_uses_bf16_rounded_norm() {
        let x = bf16(1.0);
        let z = bf16(1.0);
        let gamma = bf16(0.50390625);
        let result = run(&[x], &[z], &[gamma], 1, 1, 0.00392).unwrap();
        let normalized_bf16 = bf16_to_f32(bf16(result.normalized[0]));
        let staged_weighted = bf16(normalized_bf16 * bf16_to_f32(gamma));
        let direct_weighted = bf16(result.normalized[0] * bf16_to_f32(gamma));
        close(result.normalized[0], 0.99804574, 2e-7);
        assert_eq!(normalized_bf16, 0.99609375);
        assert_eq!(result.weighted, [bf16(0.5)]);
        assert_eq!(result.weighted[0], staged_weighted);
        assert_ne!(result.weighted[0], direct_weighted);
        close(result.silu[0], silu(1.0).unwrap(), 0.0);
        close(result.unrounded[0], 0.5 * result.silu[0], 0.0);
        assert_eq!(result.output, [bf16(result.unrounded[0])]);
    }

    #[test]
    fn zero_group_and_width_256_groups_are_independent() {
        let groups = 2;
        let width = 256;
        let mut x = vec![bf16(0.0); width];
        x.extend(vec![bf16(1.0); width]);
        let mut z = vec![bf16(0.0); width];
        z.extend(vec![bf16(1.0); width]);
        let mut weight = vec![bf16(1.0); width];
        weight[0] = bf16(2.0);
        weight[1] = bf16(-1.0);
        let result = run(&x, &z, &weight, groups, width, 1e-6).unwrap();
        assert!(result.normalized[..width].iter().all(|&value| value == 0.0));
        let expected_weighted_zero: Vec<_> = weight
            .iter()
            .map(|&gamma| round_bf16(0.0_f32 * bf16_to_f32(gamma)))
            .collect();
        let gate_zero = silu(0.0).unwrap();
        let expected_output_zero: Vec<_> = expected_weighted_zero
            .iter()
            .map(|&value| round_bf16(bf16_to_f32(value) * gate_zero))
            .collect();
        assert_eq!(&result.weighted[..width], expected_weighted_zero.as_slice());
        assert_eq!(&result.output[..width], expected_output_zero.as_slice());
        assert!(
            result.normalized[width..]
                .iter()
                .all(|&value| (value - 1.0).abs() < 1e-6)
        );
        assert_eq!(result.weighted[width], bf16(2.0));
        assert_eq!(result.weighted[width + 1], bf16(-1.0));
        assert!(
            result.weighted[width + 2..]
                .iter()
                .all(|&value| value == bf16(1.0))
        );
        assert_eq!(result.output[width], bf16(2.0 * silu(1.0).unwrap()));
        assert_eq!(result.output[width + 1], bf16(-silu(1.0).unwrap()));
        assert!(
            result.output[width + 2..]
                .iter()
                .all(|&value| value == bf16(silu(1.0).unwrap()))
        );
    }

    #[test]
    fn silu_is_stable_for_extremes_and_rejects_nonfinite_input() {
        assert_eq!(silu(0.0).unwrap(), 0.0);
        assert_eq!(silu(100.0).unwrap(), 100.0);
        let negative = silu(-100.0).unwrap();
        let exponential = (-100.0_f64).exp();
        let expected = (-100.0_f64 * exponential / (1.0 + exponential)) as f32;
        assert_eq!(negative, expected);
        assert!(negative < 0.0 && negative.abs() < f32::MIN_POSITIVE);
        assert!(silu(f32::INFINITY).is_err());
        assert!(silu(f32::NAN).is_err());
    }

    #[test]
    fn rejects_invalid_shapes_extents_inputs_epsilon_and_overflow() {
        let valid_x = [bf16(1.0)];
        let valid_z = [bf16(0.0)];
        let valid_weight = [bf16(1.0)];
        assert!(run(&valid_x, &valid_z, &valid_weight, 0, 1, 1e-6).is_err());
        assert!(run(&valid_x, &valid_z, &valid_weight, 1, 3, 1e-6).is_err());
        assert!(run(&[], &valid_z, &valid_weight, 1, 1, 1e-6).is_err());
        assert!(run(&[bf16(1.0); 2], &[bf16(0.0); 2], &[bf16(1.0)], 1, 2, 1e-6).is_err());
        assert!(run(&valid_x, &valid_z, &valid_weight, 1, 1, 0.0).is_err());
        assert!(run(&valid_x, &valid_z, &valid_weight, 1, 1, f32::NAN).is_err());

        assert!(run(&[0x7f80], &valid_z, &valid_weight, 1, 1, 1e-6).is_err());
        assert!(run(&valid_x, &[0x7f80], &valid_weight, 1, 1, 1e-6).is_err());
        assert!(run(&valid_x, &valid_z, &[0x7f80], 1, 1, 1e-6).is_err());
        assert!(run(&[0x7f7f], &valid_z, &valid_weight, 1, 1, 1e-6).is_err());
        let large_group = [bf16(1.5e19), bf16(1.5e19)];
        assert!(run(&large_group, &[bf16(0.0); 2], &[bf16(1.0); 2], 1, 2, 1e-6).is_err());

        let x = [bf16(1.0), bf16(0.0)];
        let z = [bf16(0.0), bf16(0.0)];
        let gamma_overflow = [0x7f7f, bf16(1.0)];
        assert!(run(&x, &z, &gamma_overflow, 1, 2, 1e-6).is_err());

        let gamma = [bf16(2.0_f32.powi(100)), bf16(1.0)];
        let large_gate = [0x7f7f, bf16(0.0)];
        assert!(run(&x, &large_gate, &gamma, 1, 2, 1e-6).is_err());
    }
}
