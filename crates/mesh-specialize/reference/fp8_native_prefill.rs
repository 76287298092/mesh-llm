//! Independent logical E4M3 reference and error reporting for native FP8 prefill.

use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionReference {
    /// FP32 result after row and BF16 channel scale multiplication.
    pub unrounded: Vec<f32>,
    /// The same result rounded to BF16 with round-to-nearest-even.
    pub bf16: Vec<u16>,
    /// Sum of absolute scaled products for each output, useful for error analysis.
    pub absolute_sums: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErrorReport {
    pub outputs: usize,
    pub nonfinite_unrounded: usize,
    pub nonfinite_bf16: usize,
    pub bf16_differences: usize,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub root_mean_square_error: f64,
}

/// Compute `out[m,n] = (sum_k A[m,k] * W[n,k]) * row_scale[m] * weight_scale[n]`.
///
/// E4M3 values are decoded logically and products are accumulated in FP64. The
/// reduction is cast once to FP32, then the two scale multiplications are rounded
/// in the declared order before BF16 round-to-nearest-even conversion. This is an
/// independent diagnostic reference for the kernel's experimental FP32 MMA profile;
/// its result is not proof that the kernel agrees.
pub fn fp8_native_prefill_reference(
    codes_a: &[u8],
    codes_w: &[u8],
    row_scales: &[f32],
    weight_scales_bf16: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<ProjectionReference> {
    validate_inputs(codes_a, codes_w, row_scales, weight_scales_bf16, m, n, k)?;
    let count = checked_product(m, n, "output")?;
    let mut output = ProjectionReference {
        unrounded: Vec::with_capacity(count),
        bf16: Vec::with_capacity(count),
        absolute_sums: Vec::with_capacity(count),
    };

    for (row, &row_scale) in row_scales.iter().enumerate().take(m) {
        let row_start = row * k;
        for (channel, &weight_scale) in weight_scales_bf16.iter().enumerate().take(n) {
            let weight_start = channel * k;
            let mut dot = 0.0_f64;
            let mut absolute_sum = 0.0_f64;
            for inner in 0..k {
                let product = decode_e4m3_f64(codes_a[row_start + inner])
                    * decode_e4m3_f64(codes_w[weight_start + inner]);
                dot += product;
                absolute_sum += product.abs();
            }

            let scaled_row = (dot as f32) * row_scale;
            let value = scaled_row * decode_bf16(weight_scale);
            let rounded = encode_bf16_rne(value);
            ensure!(
                value.is_finite() && decode_bf16(rounded).is_finite(),
                "native FP8 prefill reference overflow"
            );
            let scale_product = f64::from(row_scale) * f64::from(decode_bf16(weight_scale));
            output.unrounded.push(value);
            output.bf16.push(rounded);
            output.absolute_sums.push(absolute_sum * scale_product);
        }
    }
    Ok(output)
}

/// Summarize output differences without applying a pass threshold.
///
/// The caller owns qualification budgets. This helper only reports finite counts,
/// BF16 mismatches, and FP32 absolute, relative, and RMS errors.
pub fn compare_outputs(
    actual_unrounded: &[f32],
    actual_bf16: &[u16],
    reference: &ProjectionReference,
) -> Result<ErrorReport> {
    let outputs = reference.unrounded.len();
    ensure!(
        outputs > 0
            && reference.bf16.len() == outputs
            && reference.absolute_sums.len() == outputs
            && actual_unrounded.len() == outputs
            && actual_bf16.len() == outputs,
        "native FP8 prefill comparison extent mismatch"
    );

    let mut report = ErrorReport {
        outputs,
        nonfinite_unrounded: 0,
        nonfinite_bf16: 0,
        bf16_differences: 0,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
        root_mean_square_error: 0.0,
    };
    let mut squared_error = 0.0_f64;
    for ((&actual, &actual_bits), (&expected, &expected_bits)) in actual_unrounded
        .iter()
        .zip(actual_bf16)
        .zip(reference.unrounded.iter().zip(&reference.bf16))
    {
        if !actual.is_finite() {
            report.nonfinite_unrounded += 1;
            report.max_absolute_error = f64::INFINITY;
            report.max_relative_error = f64::INFINITY;
        } else {
            let error = (f64::from(actual) - f64::from(expected)).abs();
            report.max_absolute_error = report.max_absolute_error.max(error);
            let denominator = f64::from(expected).abs().max(f64::MIN_POSITIVE);
            report.max_relative_error = report.max_relative_error.max(error / denominator);
            squared_error += error * error;
        }
        if !decode_bf16(actual_bits).is_finite() {
            report.nonfinite_bf16 += 1;
        }
        if actual_bits != expected_bits {
            report.bf16_differences += 1;
        }
    }
    report.root_mean_square_error = (squared_error / outputs as f64).sqrt();
    Ok(report)
}

fn validate_inputs(
    codes_a: &[u8],
    codes_w: &[u8],
    row_scales: &[f32],
    weight_scales_bf16: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        (1..=2048).contains(&m),
        "invalid native FP8 prefill row count"
    );
    ensure!(
        (1..=262_144).contains(&n),
        "invalid native FP8 channel count"
    );
    ensure!(
        (1..=32_768).contains(&k),
        "invalid native FP8 reduction width"
    );
    ensure!(
        codes_a.len() == checked_product(m, k, "A")?,
        "native FP8 A extent mismatch"
    );
    ensure!(
        codes_w.len() == checked_product(n, k, "W")?,
        "native FP8 W extent mismatch"
    );
    ensure!(
        row_scales.len() == m,
        "native FP8 row-scale extent mismatch"
    );
    ensure!(
        weight_scales_bf16.len() == n,
        "native FP8 weight-scale extent mismatch"
    );
    ensure!(
        codes_a
            .iter()
            .chain(codes_w)
            .all(|&code| is_finite_e4m3(code)),
        "native FP8 matrices contain an E4M3 NaN code"
    );
    ensure!(
        row_scales
            .iter()
            .all(|scale| scale.is_finite() && *scale > 0.0),
        "native FP8 row scales must be finite and positive"
    );
    ensure!(
        weight_scales_bf16.iter().all(|&bits| {
            let scale = decode_bf16(bits);
            scale.is_finite() && scale > 0.0
        }),
        "native FP8 weight scales must be finite and positive BF16 values"
    );
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("native FP8 {label} extent overflows usize"))
}

fn is_finite_e4m3(code: u8) -> bool {
    code & 0x7f != 0x7f
}

fn decode_e4m3_f64(code: u8) -> f64 {
    let magnitude = code & 0x7f;
    if magnitude == 0 {
        return 0.0_f64.copysign(if code & 0x80 == 0 { 1.0 } else { -1.0 });
    }
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from(magnitude >> 3);
    let fraction = f64::from(magnitude & 0x07);
    let value = if exponent == 0 {
        fraction / 512.0
    } else {
        (1.0 + fraction / 8.0) * 2.0_f64.powi(exponent - 7)
    };
    sign * value
}

fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{ProjectionReference, compare_outputs, fp8_native_prefill_reference};

    #[test]
    fn signed_nonuniform_scales_and_odd_m_n_k_tails() {
        let m = 3;
        let n = 5;
        let k = 35;
        let row_signs = [0x38_u8, 0xb8, 0x38];
        let column_signs = [0x38_u8, 0xb8, 0x38, 0xb8, 0x38];
        let codes_a = row_signs
            .into_iter()
            .flat_map(|code| std::iter::repeat_n(code, k))
            .collect::<Vec<_>>();
        let codes_w = column_signs
            .into_iter()
            .flat_map(|code| std::iter::repeat_n(code, k))
            .collect::<Vec<_>>();
        let row_scales = [0.5_f32, 1.0, 1.5];
        let weight_scales = [0x3f80_u16, 0x3f00, 0x3fa0, 0x4000, 0x3e80];

        let result =
            fp8_native_prefill_reference(&codes_a, &codes_w, &row_scales, &weight_scales, m, n, k)
                .unwrap();

        for row in 0..m {
            for channel in 0..n {
                let sign = if row_signs[row] == column_signs[channel] {
                    1.0
                } else {
                    -1.0
                };
                let expected = sign
                    * k as f32
                    * row_scales[row]
                    * f32::from_bits(u32::from(weight_scales[channel]) << 16);
                let index = row * n + channel;
                assert_eq!(result.unrounded[index], expected);
                assert_eq!(result.absolute_sums[index], f64::from(expected.abs()));
            }
        }
        assert_eq!(result.unrounded.len(), 15);
        assert_eq!(result.bf16.len(), 15);
    }

    #[test]
    fn fp64_reference_retains_small_residual_after_large_cancellation() {
        let result = fp8_native_prefill_reference(
            &[0x7e, 0x38, 0xfe],
            &[0x7e, 0x38, 0x7e],
            &[1.0],
            &[0x3f80],
            1,
            1,
            3,
        )
        .unwrap();

        assert_eq!(result.unrounded, [1.0]);
        assert_eq!(result.bf16, [0x3f80]);
        assert_eq!(result.absolute_sums, [401_409.0]);
    }

    #[test]
    fn error_comparison_reports_metrics_without_a_pass_threshold() {
        let reference = ProjectionReference {
            unrounded: vec![1.0, -2.0],
            bf16: vec![0x3f80, 0xc000],
            absolute_sums: vec![1.0, 2.0],
        };
        let report = compare_outputs(&[1.25, -2.0], &[0x3f81, 0xc000], &reference).unwrap();

        assert_eq!(report.outputs, 2);
        assert_eq!(report.nonfinite_unrounded, 0);
        assert_eq!(report.nonfinite_bf16, 0);
        assert_eq!(report.bf16_differences, 1);
        assert_eq!(report.max_absolute_error, 0.25);
        assert_eq!(report.max_relative_error, 0.25);
        assert!((report.root_mean_square_error - 0.25 / 2.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn rejects_bad_extents_nan_codes_and_invalid_scales() {
        assert!(fp8_native_prefill_reference(&[], &[], &[], &[], 1, 1, 1).is_err());
        assert!(
            fp8_native_prefill_reference(&[0x7f], &[0x38], &[1.0], &[0x3f80], 1, 1, 1).is_err()
        );
        assert!(
            fp8_native_prefill_reference(&[0x38], &[0x38], &[0.0], &[0x3f80], 1, 1, 1).is_err()
        );
        assert!(fp8_native_prefill_reference(&[0x38], &[0x38], &[1.0], &[0], 1, 1, 1).is_err());
    }
}
