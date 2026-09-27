//! Independent scalar reference for BF16 residual addition and zero-centered RMSNorm.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

#[derive(Debug, PartialEq)]
pub struct ResidualNorm {
    pub residual: Vec<u16>,
    pub normalized: Vec<u16>,
    pub unrounded: Vec<f32>,
}

/// Add BF16 residual inputs, round, then RMS-normalize and apply `1 + weight`.
pub fn run(
    residual: &[u16],
    branch: &[u16],
    weight: &[u16],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<ResidualNorm> {
    let extent = validate(residual, branch, weight, rows, width, epsilon)?;
    let mut rounded_residual = Vec::with_capacity(extent);
    for (&left, &right) in residual.iter().zip(branch) {
        let sum = checked_add(bf16_to_f32(left), bf16_to_f32(right), "residual addition")?;
        let rounded = round_bf16(sum);
        ensure!(
            bf16_to_f32(rounded).is_finite(),
            "residual addition overflows BF16"
        );
        rounded_residual.push(rounded);
    }

    let mut result = ResidualNorm {
        residual: rounded_residual,
        normalized: Vec::with_capacity(extent),
        unrounded: Vec::with_capacity(extent),
    };
    for row in 0..rows {
        let start = row * width;
        let end = start + width;
        let factor = inverse_rms(&result.residual[start..end], width, epsilon)?;
        for (feature, &gamma) in weight.iter().enumerate() {
            let index = start + feature;
            let value = bf16_to_f32(result.residual[index]);
            let scaled = checked_mul(value, factor, "normalized residual")?;
            let centered_weight = checked_add(1.0, bf16_to_f32(gamma), "1 plus weight")?;
            let normalized = checked_mul(scaled, centered_weight, "zero-centered RMSNorm weight")?;
            let rounded = round_bf16(normalized);
            ensure!(
                bf16_to_f32(rounded).is_finite(),
                "normalized residual overflows BF16"
            );
            result.unrounded.push(normalized);
            result.normalized.push(rounded);
        }
    }
    Ok(result)
}

fn validate(
    residual: &[u16],
    branch: &[u16],
    weight: &[u16],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<usize> {
    ensure!(
        (1..=2048).contains(&rows),
        "invalid residual norm row count"
    );
    ensure!((1..=32768).contains(&width), "invalid residual norm width");
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "invalid residual norm epsilon"
    );
    let extent = rows
        .checked_mul(width)
        .context("residual norm extent overflows usize")?;
    ensure!(residual.len() == extent, "residual input extent mismatch");
    ensure!(branch.len() == extent, "residual branch extent mismatch");
    ensure!(
        weight.len() == width,
        "residual norm weight extent mismatch"
    );
    ensure!(
        residual
            .iter()
            .chain(branch)
            .chain(weight)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "residual norm inputs and weight must be finite BF16"
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
            "residual norm square overflows FP32"
        );
        square_sum += square;
        ensure!(
            square_sum.is_finite() && square_sum <= f64::from(f32::MAX),
            "residual norm sum overflows FP32"
        );
    }
    let mean = checked_f32((square_sum / width as f64) as f32, "residual norm mean")?;
    let variance = checked_add(mean, epsilon, "residual norm epsilon addition")?;
    ensure!(variance > 0.0, "residual norm variance must be positive");
    let root = checked_f32(variance.sqrt(), "residual norm square root")?;
    ensure!(root > 0.0, "residual norm square root must be positive");
    checked_f32(1.0 / root, "residual norm reciprocal")
}

fn checked_add(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left + right, label)
}

fn checked_mul(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left * right, label)
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
    fn width_one_residual_add_tie_rounds_before_normalization() {
        let result = run(&[bf16(1.0)], &[bf16(1.0 / 256.0)], &[bf16(0.0)], 1, 1, 1e-6).unwrap();
        assert_eq!(result.residual, [bf16(1.0)]);
        close(result.unrounded[0], 1.0, 1e-6);
        assert_eq!(result.normalized, [bf16(1.0)]);
    }

    #[test]
    fn cancellation_and_signed_zero_are_preserved() {
        let result = run(
            &[bf16(1.0), 0x8000],
            &[bf16(-1.0), 0x8000],
            &[bf16(0.0), bf16(-1.0)],
            1,
            2,
            1e-6,
        )
        .unwrap();
        assert_eq!(result.residual, [0x0000, 0x8000]);
        assert_eq!(result.normalized, [0x0000, 0x8000]);
        assert_eq!(result.unrounded[0].to_bits(), 0x0000_0000);
        assert_eq!(result.unrounded[1].to_bits(), 0x8000_0000);
    }

    #[test]
    fn shared_zero_centered_weights_cover_negative_one_and_multiple_rows() {
        let width = 71;
        let mut residual = vec![bf16(1.0); width];
        residual.extend(vec![bf16(2.0); width]);
        let branch = vec![bf16(0.0); width * 2];
        let mut weight = vec![bf16(0.0); width];
        weight[0] = bf16(0.0);
        weight[1] = bf16(-1.0);
        weight[2] = bf16(-0.5);
        let result = run(&residual, &branch, &weight, 2, width, 1e-6).unwrap();
        assert_eq!(result.residual[..width], vec![bf16(1.0); width]);
        assert_eq!(result.residual[width..], vec![bf16(2.0); width]);
        close(result.unrounded[0], 1.0, 1e-6);
        assert_eq!(result.normalized[1], bf16(0.0));
        close(result.unrounded[2], 0.5, 1e-6);
        close(result.unrounded[width], 1.0, 1e-6);
        assert_eq!(result.normalized[width + 1], bf16(0.0));
        close(result.unrounded[width + 2], 0.5, 1e-6);
    }

    #[test]
    fn width_one_and_zero_centered_weight_minus_one_produce_zero() {
        let result = run(&[bf16(-3.0)], &[bf16(0.0)], &[bf16(-1.0)], 1, 1, 1e-6).unwrap();
        assert_eq!(result.unrounded[0].to_bits(), 0x8000_0000);
        assert_eq!(result.normalized, [0x8000]);
    }

    #[test]
    fn rejects_invalid_dimensions_extents_inputs_epsilon_and_overflow() {
        let x = [bf16(1.0)];
        let branch = [bf16(0.0)];
        let weight = [bf16(0.0)];
        assert!(run(&x, &branch, &weight, 0, 1, 1e-6).is_err());
        assert!(run(&x, &branch, &weight, 1, 0, 1e-6).is_err());
        assert!(run(&[], &branch, &weight, 1, 1, 1e-6).is_err());
        assert!(run(&x, &[], &weight, 1, 1, 1e-6).is_err());
        assert!(run(&x, &branch, &[], 1, 1, 1e-6).is_err());
        assert!(run(&x, &branch, &weight, 1, 1, 0.0).is_err());
        assert!(run(&x, &branch, &weight, 1, 1, f32::INFINITY).is_err());
        assert!(run(&[0x7f80], &branch, &weight, 1, 1, 1e-6).is_err());
        assert!(run(&x, &[0x7f80], &weight, 1, 1, 1e-6).is_err());
        assert!(run(&x, &branch, &[0x7f80], 1, 1, 1e-6).is_err());

        let add_overflow = [0x7f7f];
        assert!(run(&add_overflow, &add_overflow, &weight, 1, 1, 1e-6).is_err());
        let square_overflow = [0x7f7f];
        assert!(run(&square_overflow, &branch, &weight, 1, 1, 1e-6).is_err());
        let sum_overflow = [bf16(1.5e19), bf16(1.5e19)];
        assert!(run(&sum_overflow, &[bf16(0.0); 2], &[bf16(0.0); 2], 1, 2, 1e-6).is_err());

        let norm_overflow = [bf16(1.0), bf16(0.0)];
        let gamma_overflow = [0x7f7f, bf16(0.0)];
        assert!(run(&norm_overflow, &[bf16(0.0); 2], &gamma_overflow, 1, 2, 1e-6).is_err());
    }
}
