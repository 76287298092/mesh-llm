use anyhow::{Context, Result, bail, ensure};

pub(super) struct PackedRow<'a> {
    pub codes: &'a [u8],
    pub scale_bytes: &'a [u8],
    pub logical_k: usize,
    pub padded_k: usize,
}

pub(super) struct Q8EncodedRow {
    pub signed_codes: Vec<i8>,
    pub scale_bits: Vec<u16>,
}

pub(super) fn q8_g32_fp16_encoded_row(row: PackedRow<'_>) -> Result<Q8EncodedRow> {
    validate_q8_row(&row)?;
    let mut signed_codes = Vec::new();
    signed_codes
        .try_reserve_exact(row.logical_k)
        .context("cannot reserve decoded Q8 codes")?;
    signed_codes.extend(
        row.codes[..row.logical_k]
            .iter()
            .map(|&code| i8::from_ne_bytes([code])),
    );
    let (scale_words, remainder) = row.scale_bytes.as_chunks::<2>();
    ensure!(remainder.is_empty(), "Q8 scale plane has partial FP16 bits");
    let scale_bits = scale_words
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect();
    Ok(Q8EncodedRow {
        signed_codes,
        scale_bits,
    })
}

pub(super) fn q8_g32_fp16_decode_row(_row: PackedRow<'_>) -> Result<Vec<f32>> {
    bail!("Q8 numeric decode is unsupported by the local source contract")
}

pub(super) fn q8_g32_fp16_dot(row: PackedRow<'_>, activations: &[f32]) -> Result<f64> {
    ensure!(
        activations.len() <= row.logical_k,
        "Q8 activation row exceeds logical K"
    );
    let _codes = q8_g32_fp16_encoded_row(PackedRow {
        logical_k: activations.len(),
        ..row
    })?;
    bail!("Q8 row dot is not available without a source-established decode equation")
}

pub(super) fn q4_g64_fp16_row(row: PackedRow<'_>) -> Result<Vec<f32>> {
    ensure!(
        row.padded_k >= row.logical_k,
        "Q4 padded K is shorter than logical K"
    );
    ensure!(
        row.padded_k.is_multiple_of(128),
        "Q4 padded K must be divisible by 128"
    );
    ensure!(
        row.codes.len() == row.padded_k / 2,
        "Q4 code row extent mismatch"
    );
    ensure!(
        row.scale_bytes.len() == row.padded_k / 32,
        "Q4 scale count mismatch"
    );
    bail!("Q4_g64_FP16 code and scale interpretation is not established by the local contract")
}

pub(super) fn q4_g64_fp16_dot(row: PackedRow<'_>, activations: &[f32]) -> Result<f64> {
    ensure!(
        activations.len() <= row.logical_k,
        "Q4 activation row exceeds logical K"
    );
    let _decoded = q4_g64_fp16_row(PackedRow {
        logical_k: activations.len(),
        ..row
    })?;
    bail!("Q4_g64_FP16 code and scale interpretation is not established by the local contract")
}

fn validate_q8_row(row: &PackedRow<'_>) -> Result<()> {
    ensure!(
        row.padded_k >= row.logical_k,
        "Q8 padded K is shorter than logical K"
    );
    ensure!(
        row.padded_k.is_multiple_of(128),
        "Q8 padded K must be divisible by 128"
    );
    ensure!(
        row.codes.len() == row.padded_k,
        "Q8 code row extent mismatch"
    );
    ensure!(
        row.scale_bytes.len() == row.padded_k / 16,
        "Q8 scale count mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PackedRow, q4_g64_fp16_dot, q4_g64_fp16_row, q8_g32_fp16_decode_row, q8_g32_fp16_dot,
        q8_g32_fp16_encoded_row,
    };

    fn scales(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect()
    }

    #[test]
    fn q8_reference_sign_extends_codes_and_preserves_scale_bits() {
        let mut codes = vec![0; 128];
        codes[0] = 0xff;
        codes[1] = 2;
        codes[2] = 0x80;
        codes[3] = 0x7f;
        let scale_bytes = scales(&[0x7bff, 0x0001, 0x3c00, 0xfc00]);
        let row = q8_g32_fp16_encoded_row(PackedRow {
            codes: &codes,
            scale_bytes: &scale_bytes,
            logical_k: 3,
            padded_k: 128,
        })
        .expect("well-shaped Q8 codes");
        assert_eq!(row.signed_codes, [-1, 2, i8::MIN]);
        assert_eq!(row.scale_bits, [0x7bff, 0x0001, 0x3c00, 0xfc00]);
    }

    #[test]
    fn q8_reference_ignores_padding_and_rejects_malformed_rows() {
        let mut codes = vec![0; 128];
        codes[127] = 0x7f;
        let scale_bytes = scales(&[0; 4]);
        let row = q8_g32_fp16_encoded_row(PackedRow {
            codes: &codes,
            scale_bytes: &scale_bytes,
            logical_k: 3,
            padded_k: 128,
        })
        .expect("padded Q8 row");
        assert_eq!(row.signed_codes, [0, 0, 0]);
        assert!(
            q8_g32_fp16_encoded_row(PackedRow {
                codes: &[0; 127],
                scale_bytes: &scale_bytes,
                logical_k: 1,
                padded_k: 128,
            })
            .is_err()
        );
        assert!(
            q8_g32_fp16_encoded_row(PackedRow {
                codes: &[0; 128],
                scale_bytes: &[0; 6],
                logical_k: 1,
                padded_k: 128,
            })
            .is_err()
        );
    }

    #[test]
    fn q8_reference_keeps_extreme_scale_bits_without_guessing_numeric_decode() {
        let mut codes = vec![0; 128];
        codes[0] = 0x7f;
        let maximum_scale = scales(&[0x7bff, 0x0001, 0x3c00, 0xfc00]);
        let encoded = q8_g32_fp16_encoded_row(PackedRow {
            codes: &codes,
            scale_bytes: &maximum_scale,
            logical_k: 1,
            padded_k: 128,
        })
        .expect("Q8 signed bytes parse");
        assert_eq!(encoded.scale_bits, [0x7bff, 0x0001, 0x3c00, 0xfc00]);
        assert!(
            q8_g32_fp16_decode_row(PackedRow {
                codes: &codes,
                scale_bytes: &maximum_scale,
                logical_k: 1,
                padded_k: 128,
            })
            .is_err()
        );
        assert!(
            q8_g32_fp16_dot(
                PackedRow {
                    codes: &codes,
                    scale_bytes: &maximum_scale,
                    logical_k: 1,
                    padded_k: 128,
                },
                &[1.0],
            )
            .is_err()
        );
    }

    #[test]
    fn q8_reference_preserves_all_fp16_bit_patterns_without_interpreting_scales() {
        let infinite_scale = scales(&[0x7c00; 4]);
        let encoded = q8_g32_fp16_encoded_row(PackedRow {
            codes: &[0; 128],
            scale_bytes: &infinite_scale,
            logical_k: 1,
            padded_k: 128,
        })
        .expect("raw scale bits are preserved");
        assert_eq!(encoded.scale_bits, [0x7c00; 4]);
        assert!(
            q8_g32_fp16_decode_row(PackedRow {
                codes: &[0; 128],
                scale_bytes: &infinite_scale,
                logical_k: 1,
                padded_k: 128,
            })
            .is_err()
        );
    }

    #[test]
    fn q4_reference_checks_geometry_then_rejects_unknown_code_equation() {
        let scale_bytes = scales(&[0x3c00; 2]);
        assert!(
            q4_g64_fp16_row(PackedRow {
                codes: &[0; 63],
                scale_bytes: &scale_bytes,
                logical_k: 1,
                padded_k: 128,
            })
            .is_err()
        );
        assert!(
            q4_g64_fp16_dot(
                PackedRow {
                    codes: &[0; 64],
                    scale_bytes: &scale_bytes,
                    logical_k: 1,
                    padded_k: 128,
                },
                &[1.0],
            )
            .is_err()
        );
    }
}
