//! Independent scalar embedding and zero-centered RMSNorm reference.

use anyhow::{Result, ensure};

pub struct EntryReference {
    pub residual: Vec<u16>,
    pub normalized: Vec<u16>,
    pub unrounded: Vec<f32>,
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub fn round_bf16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn embedding_norm(
    table: &[u8],
    tokens: &[u32],
    weight: &[u8],
    width: usize,
    epsilon: f32,
) -> Result<EntryReference> {
    ensure!((1..=32768).contains(&width), "invalid embedding width");
    ensure!((1..=2048).contains(&tokens.len()), "invalid token count");
    ensure!(epsilon.is_finite() && epsilon > 0.0, "invalid epsilon");
    let row_bytes = width * 2;
    ensure!(weight.len() == row_bytes, "invalid norm weight length");
    ensure!(
        !table.is_empty() && table.len().is_multiple_of(row_bytes),
        "invalid embedding table length"
    );
    let vocabulary = table.len() / row_bytes;
    ensure!(
        tokens.iter().all(|id| (*id as usize) < vocabulary),
        "token outside vocabulary"
    );
    let scales: Vec<f32> = weight
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16_to_f32(u16::from_le_bytes(*bytes)))
        .collect();
    ensure!(
        scales.iter().all(|value| value.is_finite()),
        "nonfinite norm weight"
    );
    let count = tokens
        .len()
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("output overflow"))?;
    let mut result = EntryReference {
        residual: Vec::with_capacity(count),
        normalized: Vec::with_capacity(count),
        unrounded: Vec::with_capacity(count),
    };
    for &token in tokens {
        let start = token as usize * row_bytes;
        let words: Vec<u16> = table[start..start + row_bytes]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect();
        let values: Vec<f32> = words.iter().map(|&bits| bf16_to_f32(bits)).collect();
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "nonfinite embedding"
        );
        let sum: f64 = values.iter().map(|&value| f64::from(value).powi(2)).sum();
        let mean = (sum / width as f64) as f32;
        ensure!(
            (mean + epsilon).is_finite(),
            "normalization variance overflow"
        );
        let factor = 1.0 / (mean + epsilon).sqrt();
        for (&value, &weight) in values.iter().zip(&scales) {
            let normalized = value * factor;
            let output = normalized * (1.0 + weight);
            ensure!(output.is_finite(), "nonfinite reference output");
            result.unrounded.push(output);
            result.normalized.push(round_bf16(output));
        }
        result.residual.extend(words);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| round_bf16(value).to_le_bytes())
            .collect()
    }
    #[test]
    fn conversion_handles_ties_sign_and_special_values() {
        assert_eq!(round_bf16(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(round_bf16(f32::from_bits(0x3f818000)), 0x3f82);
        assert_eq!(round_bf16(f32::from_bits(0xbf818000)), 0xbf82);
        assert_eq!(round_bf16(-0.0), 0x8000);
        assert_eq!(round_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(round_bf16(f32::NEG_INFINITY), 0xff80);
        assert_eq!(round_bf16(f32::NAN), 0x7fc0);
        assert_eq!(bf16_to_f32(0x3fc0), 1.5);
    }
    #[test]
    fn lookup_and_zero_centered_norm_follow_hand_computed_rows() {
        let result = embedding_norm(
            &bytes(&[1.0, -1.0, 0.0, 0.0, 2.0, -2.0]),
            &[2, 0, 2, 1],
            &bytes(&[0.0, -1.0]),
            2,
            1e-6,
        )
        .unwrap();
        assert_eq!(
            result.residual,
            [0x4000, 0xc000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0, 0]
        );
        assert_eq!(
            result.normalized,
            [0x3f80, 0x8000, 0x3f80, 0x8000, 0x3f80, 0x8000, 0, 0]
        );
        assert_eq!(result.unrounded[0], 2.0 / (4.0_f32 + 1e-6).sqrt());
        assert_eq!(result.unrounded[2], 1.0 / (1.0_f32 + 1e-6).sqrt());
    }
    #[test]
    fn rejects_invalid_dimensions_tokens_and_nonfinite_values() {
        let table = bytes(&[1.0, 2.0]);
        let weight = bytes(&[0.0, 0.0]);
        assert!(embedding_norm(&table, &[1], &weight, 2, 1e-6).is_err());
        assert!(embedding_norm(&table, &[], &weight, 2, 1e-6).is_err());
        assert!(embedding_norm(&table, &[0], &weight, 0, 1e-6).is_err());
        assert!(embedding_norm(&table, &[0], &weight, 2, f32::NAN).is_err());
        assert!(embedding_norm(&table, &[0], &weight, 2, 0.0).is_err());
        assert!(embedding_norm(&table[..3], &[0], &weight, 2, 1e-6).is_err());
        assert!(embedding_norm(&table, &[0], &weight[..2], 2, 1e-6).is_err());
        assert!(embedding_norm(&bytes(&[f32::NAN, 0.0]), &[0], &weight, 2, 1e-6).is_err());
        assert!(embedding_norm(&table, &[0], &bytes(&[f32::INFINITY, 0.0]), 2, 1e-6).is_err());
    }
}
