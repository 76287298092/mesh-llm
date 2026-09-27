//! Independent logical FP8/BF16 quantization and dense projection arithmetic.
use crate::entry_reference::{bf16_to_f32, round_bf16};
use anyhow::{Result, ensure};

pub struct QuantizedRows {
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
}

pub struct LinearReference {
    pub unrounded: Vec<f32>,
    pub normalized: Vec<u16>,
    pub absolute_sums: Vec<f64>,
}

/// Logical BF16 X[m,k] times W[n,k]^T, with no activation quantization or scales.
pub fn linear_bf16(
    input: &[u16],
    weights: &[u16],
    rows: usize,
    width: usize,
) -> Result<LinearReference> {
    ensure!(
        (1..=2048).contains(&rows) && (1..=32768).contains(&width),
        "invalid BF16 projection dimensions"
    );
    ensure!(input.len() == rows * width, "BF16 input extent mismatch");
    ensure!(
        !weights.is_empty() && weights.len().is_multiple_of(width),
        "BF16 weight extent mismatch"
    );
    let channels = weights.len() / width;
    ensure!((1..=262144).contains(&channels), "invalid BF16 channels");
    ensure!(
        input
            .iter()
            .chain(weights)
            .all(|&v| bf16_to_f32(v).is_finite()),
        "nonfinite BF16 projection input"
    );
    let count = rows * channels;
    let mut output = LinearReference {
        unrounded: Vec::with_capacity(count),
        normalized: Vec::with_capacity(count),
        absolute_sums: Vec::with_capacity(count),
    };
    for row in input.chunks_exact(width) {
        for weight in weights.chunks_exact(width) {
            let mut sum = 0.0_f64;
            let mut absolute = 0.0_f64;
            for (&a, &b) in row.iter().zip(weight) {
                let product = f64::from(bf16_to_f32(a)) * f64::from(bf16_to_f32(b));
                sum += product;
                absolute += product.abs();
            }
            let value = sum as f32;
            let rounded = round_bf16(value);
            ensure!(
                value.is_finite() && bf16_to_f32(rounded).is_finite(),
                "BF16 projection overflow"
            );
            output.unrounded.push(value);
            output.normalized.push(rounded);
            output.absolute_sums.push(absolute);
        }
    }
    Ok(output)
}

/// E4M3FN, including subnormals and its two NaN bit patterns.
pub fn decode(code: u8) -> f32 {
    let magnitude = code & 127;
    let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
    if magnitude == 127 {
        return f32::NAN;
    }
    let exponent = i32::from(magnitude >> 3);
    let fraction = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        fraction / 512.0
    } else {
        (1.0 + fraction / 8.0) * 2.0_f32.powi(exponent - 7)
    };
    sign * value
}

/// Search the finite representable set. Independent of the device encoder.
pub fn encode(value: f32) -> Result<u8> {
    ensure!(value.is_finite(), "nonfinite FP8 input");
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    let magnitude = value.abs().min(448.0);
    let mut best = 0_u8;
    let mut error = f32::INFINITY;
    for code in 0..=126 {
        let distance = (magnitude - decode(code)).abs();
        if distance < error || (distance == error && code & 1 == 0) {
            error = distance;
            best = code;
        }
    }
    Ok(sign | best)
}

pub fn quantize(input: &[u16], rows: usize, width: usize) -> Result<QuantizedRows> {
    ensure!(
        (1..=2048).contains(&rows) && (1..=32768).contains(&width),
        "invalid FP8 dimensions"
    );
    ensure!(input.len() == rows * width, "invalid FP8 input extent");
    let mut output = QuantizedRows {
        codes: Vec::with_capacity(input.len()),
        scales: Vec::with_capacity(rows),
    };
    for row in input.chunks_exact(width) {
        let values: Vec<_> = row.iter().map(|&v| bf16_to_f32(v)).collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite BF16 activation"
        );
        let amax = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = amax / 448.0;
        let scale = if scale == 0.0 { 1.0 } else { scale };
        output.scales.push(scale);
        for value in values {
            output.codes.push(encode(value / scale)?);
        }
    }
    Ok(output)
}

/// Logical X[m,k] times W[n,k]^T, followed by row and channel scales.
pub fn linear(
    input: &QuantizedRows,
    weights: &[u8],
    scales: &[u16],
    width: usize,
) -> Result<LinearReference> {
    let rows = input.scales.len();
    let channels = scales.len();
    ensure!(rows > 0 && channels > 0 && width > 0, "empty projection");
    ensure!(
        input.codes.len()
            == rows
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("input size overflow"))?,
        "input extent mismatch"
    );
    ensure!(
        weights.len()
            == channels
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("weight size overflow"))?,
        "weight extent mismatch"
    );
    let values: [f64; 256] = std::array::from_fn(|code| f64::from(decode(code as u8)));
    ensure!(
        input
            .codes
            .iter()
            .chain(weights)
            .all(|&code| values[usize::from(code)].is_finite()),
        "nonfinite FP8 code"
    );
    ensure!(
        input.scales.iter().all(|s| s.is_finite() && *s > 0.0),
        "invalid activation scale"
    );
    ensure!(
        scales
            .iter()
            .all(|&s| bf16_to_f32(s).is_finite() && bf16_to_f32(s) > 0.0),
        "invalid channel scale"
    );
    let count = rows
        .checked_mul(channels)
        .ok_or_else(|| anyhow::anyhow!("projection size overflow"))?;
    let mut output = LinearReference {
        unrounded: Vec::with_capacity(count),
        normalized: Vec::with_capacity(count),
        absolute_sums: Vec::with_capacity(count),
    };
    for (row, &sx) in input.codes.chunks_exact(width).zip(&input.scales) {
        for (weight, &sw) in weights.chunks_exact(width).zip(scales) {
            let mut sum = 0.0_f64;
            let mut absolute = 0.0_f64;
            for (&a, &b) in row.iter().zip(weight) {
                let product = values[usize::from(a)] * values[usize::from(b)];
                sum += product;
                absolute += product.abs();
            }
            let sw = bf16_to_f32(sw);
            let scaled = (sum as f32 * sx) * sw;
            ensure!(scaled.is_finite(), "nonfinite projected value");
            output.unrounded.push(scaled);
            output.normalized.push(round_bf16(scaled));
            output
                .absolute_sums
                .push(absolute * f64::from(sx) * f64::from(sw));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_projection_preserves_fractional_inputs_and_rejects_invalid_data() {
        let input = [1.5, -2.0, 0.25, 3.0].map(round_bf16);
        let weights = [2.0, -0.5, 1.25, 2.0, -1.0, 1.0].map(round_bf16);
        let result = linear_bf16(&input, &weights, 2, 2).unwrap();
        assert_eq!(result.unrounded, [4.0, -2.125, -3.5, -1.0, 6.3125, 2.75]);
        assert_eq!(result.absolute_sums, [4.0, 5.875, 3.5, 2.0, 6.3125, 3.25]);
        assert_eq!(
            result.normalized,
            result
                .unrounded
                .iter()
                .map(|&x| round_bf16(x))
                .collect::<Vec<_>>()
        );
        assert!(linear_bf16(&input, &weights, 0, 2).is_err());
        assert!(linear_bf16(&input, &weights, 2, 0).is_err());
        assert!(linear_bf16(&input[..3], &weights, 2, 2).is_err());
        assert!(linear_bf16(&input, &weights[..5], 2, 2).is_err());
        assert!(linear_bf16(&[0x7fc0], &[0x3f80], 1, 1).is_err());
        assert!(linear_bf16(&[0x3f80], &[0x7f80], 1, 1).is_err());
        assert!(linear_bf16(&[0x7f7f], &[0x7f7f], 1, 1).is_err());
    }
    #[test]
    fn fp8_values_ties_saturation_and_signed_zero() {
        assert_eq!(decode(1), 1.0 / 512.0);
        assert_eq!(decode(8), 1.0 / 64.0);
        assert_eq!(decode(0x38), 1.0);
        assert_eq!(decode(0x7e), 448.0);
        assert!(decode(0x7f).is_nan());
        assert!(decode(0xff).is_nan());
        assert_eq!(encode(1.0625).unwrap(), 0x38);
        assert_eq!(encode(1.1875).unwrap(), 0x3a);
        assert_eq!(encode(-1.1875).unwrap(), 0xba);
        assert_eq!(encode(900.0).unwrap(), 0x7e);
        assert_eq!(encode(-0.0).unwrap(), 0x80);
        for code in 0..=255 {
            if code & 127 != 127 {
                assert_eq!(encode(decode(code)).unwrap(), code);
            }
        }
    }
    #[test]
    fn independent_row_scales_and_zero_row() {
        let input = [0, 0x8000, round_bf16(448.0), round_bf16(-224.0)];
        let q = quantize(&input, 2, 2).unwrap();
        assert_eq!(q.scales, [1.0, 1.0]);
        assert_eq!(q.codes, [0, 128, 126, 246]);
        assert!(quantize(&[0x7fc0], 1, 1).is_err());
    }
    #[test]
    fn logical_matrix_product_and_scale_order() {
        let input = QuantizedRows {
            codes: vec![0x38, 0x40, 0xb8, 0x38],
            scales: vec![0.5, 2.0],
        };
        let weights = [0x40, 0xb8, 0x38, 0x38];
        let result = linear(&input, &weights, &[round_bf16(2.0), round_bf16(0.5)], 2).unwrap();
        assert_eq!(result.unrounded, [0.0, 0.75, -12.0, 0.0]);
        assert_eq!(result.absolute_sums, [4.0, 0.75, 12.0, 2.0]);
        assert!(linear(&input, &weights[..3], &[0x3f80, 0x3f80], 2).is_err());
        assert!(linear(&input, &[127, 0, 0, 0], &[0x3f80, 0x3f80], 2).is_err());
    }
}
