const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode one signed FP4 E2M1 value.
pub fn decode_e2m1(code: u8) -> Result<f32, String> {
    if code > 0x0f {
        return Err("E2M1 code must fit in four bits".to_string());
    }

    let magnitude = E2M1_MAGNITUDES[(code & 0x07) as usize];
    Ok(if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// Decode one unsigned positive FP8 E4M3 scale value.
pub fn decode_ue4m3(code: u8) -> Result<f32, String> {
    if code & 0x80 != 0 {
        return Err("UE4M3 scale code must not set the sign bit".to_string());
    }
    if code == 0x7f {
        return Err("UE4M3 scale code 0x7f is NaN".to_string());
    }

    let exponent = ((code >> 3) & 0x0f) as i32;
    let mantissa = (code & 0x07) as f32;
    if exponent == 0 {
        return Ok(mantissa * 2.0_f32.powi(-9));
    }

    Ok((1.0 + mantissa / 8.0) * 2.0_f32.powi(exponent - 7))
}

/// Compute a dense dot product using f64 accumulation.
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.is_empty() || a.len() != b.len() {
        return Err("dot product requires equal, nonempty inputs".to_string());
    }
    validate_finite(a, "dot input")?;
    validate_finite(b, "dot input")?;

    let sum = a
        .iter()
        .zip(b)
        .map(|(&left, &right)| f64::from(left) * f64::from(right))
        .sum();
    cast_finite(sum, "dot product")
}

/// Normalize one vector by its root mean square and apply elementwise weights.
pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, String> {
    if input.is_empty() || input.len() != weight.len() {
        return Err("RMSNorm requires equal, nonempty input and weight".to_string());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("RMSNorm epsilon must be finite and positive".to_string());
    }
    validate_finite(input, "RMSNorm input")?;
    validate_finite(weight, "RMSNorm weight")?;

    let sum_squares: f64 = input
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum();
    let mean_square = sum_squares / input.len() as f64;
    let denominator = (mean_square + f64::from(epsilon)).sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("RMSNorm denominator is not finite and positive".to_string());
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| "RMSNorm output allocation failed".to_string())?;
    for (&value, &scale) in input.iter().zip(weight) {
        output.push(cast_finite(
            f64::from(value) / denominator * f64::from(scale),
            "RMSNorm output",
        )?);
    }
    Ok(output)
}

/// Multiply row-major A (m by k) and B (k by n), accumulating in f64.
pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("matrix dimensions must be nonzero".to_string());
    }
    let a_len = checked_matrix_len(m, k, "A")?;
    let b_len = checked_matrix_len(k, n, "B")?;
    let output_len = checked_matrix_len(m, n, "output")?;
    if a.len() != a_len || b.len() != b_len {
        return Err("matrix input lengths do not match their dimensions".to_string());
    }
    validate_finite(a, "matrix input")?;
    validate_finite(b, "matrix input")?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| "matrix output allocation failed".to_string())?;
    output.resize(output_len, 0.0);
    for (a_row, output_row) in a.chunks_exact(k).zip(output.chunks_exact_mut(n)) {
        for (column, output_cell) in output_row.iter_mut().enumerate() {
            let sum: f64 = a_row
                .iter()
                .zip(b.chunks_exact(n))
                .map(|(&a_value, b_row)| f64::from(a_value) * f64::from(b_row[column]))
                .sum();
            *output_cell = cast_finite(sum, "matrix product")?;
        }
    }
    Ok(output)
}

fn checked_matrix_len(rows: usize, columns: usize, label: &str) -> Result<usize, String> {
    rows.checked_mul(columns)
        .ok_or_else(|| format!("{label} matrix size overflowed"))
}

fn validate_finite(values: &[f32], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a nonfinite value"));
    }
    Ok(())
}

fn cast_finite(value: f64, label: &str) -> Result<f32, String> {
    let result = value as f32;
    if !result.is_finite() {
        return Err(format!("{label} is not finite as f32"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{decode_e2m1, decode_ue4m3, dot, matmul, rms_norm};

    #[test]
    fn e2m1_decodes_every_four_bit_code() {
        let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, expected) in magnitudes.into_iter().enumerate() {
            assert_eq!(decode_e2m1(code as u8).unwrap(), expected);
            let negative = decode_e2m1(code as u8 + 8).unwrap();
            assert_eq!(negative, -expected);
            if code == 0 {
                assert!(negative.is_sign_negative());
            }
        }
        assert!(decode_e2m1(0x10).is_err());
        assert!(decode_e2m1(u8::MAX).is_err());
    }

    #[test]
    fn ue4m3_decodes_subnormal_normal_and_maximum_boundaries() {
        assert_eq!(decode_ue4m3(0x00).unwrap(), 0.0);
        assert_eq!(decode_ue4m3(0x01).unwrap(), 1.0 / 512.0);
        assert_eq!(decode_ue4m3(0x07).unwrap(), 7.0 / 512.0);
        assert_eq!(decode_ue4m3(0x08).unwrap(), 1.0 / 64.0);
        assert_eq!(decode_ue4m3(0x37).unwrap(), 0.9375);
        assert_eq!(decode_ue4m3(0x38).unwrap(), 1.0);
        assert_eq!(decode_ue4m3(0x40).unwrap(), 2.0);
        assert_eq!(decode_ue4m3(0x7e).unwrap(), 448.0);
        assert!(decode_ue4m3(0x7f).is_err());
        assert!(decode_ue4m3(0x80).is_err());
        assert!(decode_ue4m3(0xff).is_err());
    }

    #[test]
    fn dot_uses_finite_inputs_and_rejects_invalid_shapes_or_results() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(), 32.0);
        assert!(dot(&[], &[]).is_err());
        assert!(dot(&[1.0], &[]).is_err());
        assert!(dot(&[f32::NAN], &[1.0]).is_err());
        assert!(dot(&[f32::INFINITY], &[1.0]).is_err());
        assert!(dot(&[f32::MAX], &[f32::MAX]).is_err());
    }

    #[test]
    fn rms_norm_matches_hand_worked_epsilon_case() {
        let result = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.5).unwrap();
        let denominator = 13.0_f32.sqrt();

        assert_close(result[0], 3.0 / denominator);
        assert_close(result[1], 8.0 / denominator);
    }

    #[test]
    fn rms_norm_rejects_invalid_shapes_values_and_epsilon() {
        assert!(rms_norm(&[], &[], 1.0).is_err());
        assert!(rms_norm(&[1.0], &[], 1.0).is_err());
        assert!(rms_norm(&[f32::NAN], &[1.0], 1.0).is_err());
        assert!(rms_norm(&[1.0], &[f32::INFINITY], 1.0).is_err());
        assert!(rms_norm(&[1.0], &[1.0], 0.0).is_err());
        assert!(rms_norm(&[1.0], &[1.0], f32::NAN).is_err());
        assert!(
            rms_norm(
                &[1.0, 0.0, 0.0, 0.0],
                &[f32::MAX, 1.0, 1.0, 1.0],
                f32::MIN_POSITIVE
            )
            .is_err()
        );
    }

    #[test]
    fn matmul_matches_nonsymmetric_hand_worked_product() {
        let result = matmul(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(result, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_invalid_dimensions_shapes_inputs_and_results() {
        assert!(matmul(&[], &[], 0, 1, 1).is_err());
        assert!(matmul(&[], &[], usize::MAX, 1, 2).is_err());
        assert!(matmul(&[], &[], 1, 1, 1).is_err());
        assert!(matmul(&[1.0], &[1.0], 1, 2, 1).is_err());
        assert!(matmul(&[f32::NAN], &[1.0], 1, 1, 1).is_err());
        assert!(matmul(&[1.0], &[f32::INFINITY], 1, 1, 1).is_err());
        assert!(matmul(&[f32::MAX], &[f32::MAX], 1, 1, 1).is_err());
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}
