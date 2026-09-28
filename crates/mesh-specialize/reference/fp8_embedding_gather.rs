//! Independent scalar E4M3FN lookup oracle. No device decoding or math helpers.

use anyhow::{Context, Result, ensure};

/// Decode the logical signed E4M3FN number, not an implementation's packed format.
pub fn decode_e4m3(code: u8) -> f32 {
    let exponent = i32::from((code >> 3) & 15);
    let fraction = code & 7;
    let magnitude = match (exponent, fraction) {
        (15, 7) => f32::NAN,
        (0, _) => f32::from(fraction) * 2.0_f32.powi(-9),
        _ => (1.0 + f32::from(fraction) / 8.0) * 2.0_f32.powi(exponent - 7),
    };
    if code & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Round FP32 to BF16 by comparing the discarded bits with half an ULP.
fn round_bf16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let high = (bits >> 16) as u16;
    let low = bits & 0xffff;
    let increment = low > 0x8000 || (low == 0x8000 && high & 1 != 0);
    high.wrapping_add(u16::from(increment))
}

/// Exactly the gather's pre-normalization scalar boundary.
pub fn dequantize(code: u8, scale: u16) -> u16 {
    let scale = f32::from_bits(u32::from(scale) << 16);
    round_bf16(decode_e4m3(code) * scale)
}

/// Gather requested rows without materializing a decoded vocabulary table.
pub fn gather(
    codes: &[u8],
    scales: &[u16],
    tokens: &[u32],
    vocabulary: usize,
    width: usize,
) -> Result<Vec<u16>> {
    ensure!((1..=1_048_576).contains(&vocabulary), "invalid vocabulary");
    ensure!((1..=32768).contains(&width), "invalid embedding width");
    ensure!((1..=2048).contains(&tokens.len()), "invalid row count");
    let count = vocabulary
        .checked_mul(width)
        .context("embedding extent overflow")?;
    ensure!(codes.len() == count, "E4M3 table extent mismatch");
    ensure!(scales.len() == vocabulary, "BF16 row-scale extent mismatch");
    ensure!(
        tokens.iter().all(|&t| u64::from(t) < vocabulary as u64),
        "token out of range"
    );
    let mut result = Vec::with_capacity(tokens.len() * width);
    for &token in tokens {
        let row = token as usize;
        for &code in &codes[row * width..(row + 1) * width] {
            result.push(dequantize(code, scales[row]));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_decoder_covers_extremes_subnormals_and_signed_zero() {
        assert_eq!(decode_e4m3(0).to_bits(), 0);
        assert_eq!(decode_e4m3(0x80).to_bits(), 0x8000_0000);
        for (code, value) in [
            (1, 1.0 / 512.0),
            (7, 7.0 / 512.0),
            (8, 1.0 / 64.0),
            (0x38, 1.0),
            (0x78, 256.0),
            (0x7e, 448.0),
        ] {
            assert_eq!(decode_e4m3(code), value);
            assert_eq!(decode_e4m3(code | 0x80), -value);
        }
        assert!(decode_e4m3(0x7f).is_nan());
        assert!(decode_e4m3(0xff).is_nan());
    }

    #[test]
    fn multiplication_rounds_ties_to_even_before_normalization() {
        // 1.5 * 1.0078125 = 1.51171875: odd low neighbor rounds upward.
        assert_eq!(dequantize(0x3c, 0x3f81), 0x3fc2);
        // 1.5 * 1.0234375 = 1.53515625: even low neighbor rounds downward.
        assert_eq!(dequantize(0x3c, 0x3f83), 0x3fc4);
        assert_eq!(dequantize(0xbc, 0x3f81), 0xbfc2);
        assert_eq!(dequantize(0xbc, 0x3f83), 0xbfc4);
        assert_eq!(dequantize(0x80, 0x3f80), 0x8000);
        assert_eq!(dequantize(0x80, 0xbf80), 0);
        assert_eq!(dequantize(0, 0xbf80), 0x8000);
        // BF16 subnormal and overflow boundaries also retain round-to-even.
        assert_eq!(dequantize(0x30, 1), 0);
        assert_eq!(dequantize(0x3c, 1), 2);
        assert_eq!(dequantize(0x40, 0x7f7f), 0x7f80);
    }

    #[test]
    fn existing_norm_sees_only_rounded_rows_and_keeps_zero_centered_weights() {
        let rows = gather(&[0x3c, 0x38], &[0x3f81], &[0, 0], 1, 2).unwrap();
        let bytes: Vec<_> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result =
            crate::entry_reference::embedding_norm(&bytes, &[0, 1], &[0, 0, 0x80, 0xbf], 2, 1e-6)
                .unwrap();
        assert_eq!(result.residual, rows);
        assert_eq!(result.residual[0], 0x3fc2);
        assert_eq!(result.normalized[1], 0); // weight -1 => 1 + weight == 0
        assert_eq!(result.normalized[..2], result.normalized[2..]);
        let x = crate::entry_reference::bf16_to_f32(rows[0]);
        let y = crate::entry_reference::bf16_to_f32(rows[1]);
        let factor =
            1.0 / (((f64::from(x).powi(2) + f64::from(y).powi(2)) / 2.0) as f32 + 1e-6).sqrt();
        assert_eq!(result.unrounded[0], x * factor);
    }

    #[test]
    fn repeated_and_endpoint_tokens_use_their_own_row_scale() {
        let codes = [0x38, 0xb8, 0x40, 0x80, 0x3c, 0x3c];
        let scales = [0x4000, 0x3f80, 0x3f81];
        assert_eq!(
            gather(&codes, &scales, &[2, 0, 2, 1], 3, 2).unwrap(),
            [
                0x3fc2, 0x3fc2, 0x4000, 0xc000, 0x3fc2, 0x3fc2, 0x4000, 0x8000
            ]
        );
        assert!(gather(&codes, &scales, &[3], 3, 2).is_err());
        assert!(gather(&codes[..5], &scales, &[0], 3, 2).is_err());
        assert!(gather(&codes, &scales[..2], &[0], 3, 2).is_err());
        assert!(gather(&codes, &scales, &[], 3, 2).is_err());
    }
}
