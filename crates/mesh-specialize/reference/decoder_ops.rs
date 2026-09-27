//! Independent scalar operations shared by the whole-model CPU composition.
use crate::{
    entry_reference,
    kernels::{Bf16Projection, Fp8Projection},
    projection_reference,
};
use anyhow::{Result, ensure};

pub fn words(bytes: &[u8]) -> Result<Vec<u16>> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "unaligned BF16 reference bytes"
    );
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect())
}
pub fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub fn normalize(hidden: &[u16], weight: &[u16], rows: usize, width: usize) -> Result<Vec<u16>> {
    ensure!(
        (1..=2048).contains(&rows)
            && (1..=32768).contains(&width)
            && hidden.len() == rows * width
            && weight.len() == width,
        "invalid decoder norm extent"
    );
    let ids = (0..u32::try_from(rows)?).collect::<Vec<_>>();
    Ok(
        entry_reference::embedding_norm(&bytes(hidden), &ids, &bytes(weight), width, 1e-6)?
            .normalized,
    )
}
pub fn fp8(
    hidden: &[u16],
    projection: &Fp8Projection,
    rows: usize,
    width: usize,
) -> Result<Vec<u16>> {
    let scales = words(&projection.scales)?;
    ensure!(
        scales.len() == projection.channels,
        "decoder FP8 channel extent mismatch"
    );
    let input = projection_reference::quantize(hidden, rows, width)?;
    Ok(projection_reference::linear(&input, &projection.weights, &scales, width)?.normalized)
}
pub fn bf16(
    hidden: &[u16],
    projection: &Bf16Projection,
    rows: usize,
    width: usize,
) -> Result<Vec<u16>> {
    let weights = words(&projection.weights)?;
    ensure!(
        projection.channels.checked_mul(width) == Some(weights.len()),
        "decoder BF16 weight extent mismatch"
    );
    Ok(projection_reference::linear_bf16(hidden, &weights, rows, width)?.normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_norm_preserves_independent_rows_and_zero_centered_weights() {
        let input = [0x3f80, 0x4000, 0, 0];
        let actual = normalize(&input, &[0, 0], 2, 2).unwrap();
        assert_eq!(actual[..2], normalize(&input[..2], &[0, 0], 1, 2).unwrap());
        assert_eq!(actual[2..], [0, 0]);
        assert_eq!(normalize(&input, &[0xbf80; 2], 2, 2).unwrap(), [0; 4]);
        assert!(normalize(&input, &[0], 2, 2).is_err());
        assert!(words(&[1]).is_err());
    }
}
