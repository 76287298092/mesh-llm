use super::super::{
    error::NativeMtpDecodeError,
    row::{PackedRow, q4_g64_fp16_decode_row, q8_g32_fp16_decode_row},
};

fn scale_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn q4_code_bytes(code: u8, padded_k: usize) -> Vec<u8> {
    vec![(code & 0x0f) | ((code & 0x0f) << 4); padded_k / 2]
}

#[test]
fn q8_signed_edge_codes_decode_and_reject_the_forbidden_word() {
    let mut codes = vec![0; 128];
    codes[..4].copy_from_slice(&[0xff, 0x81, 0x00, 0x7f]);
    let mut scales = scale_bytes(&[0x3c00; 4]);
    let (scale_words, remainder) = scales.as_chunks_mut::<2>();
    assert!(remainder.is_empty());
    for (group, scale) in scale_words.iter_mut().enumerate().skip(1) {
        *scale = 0_u16.to_le_bytes();
        codes[group * 32..(group + 1) * 32].fill(0);
    }
    let decoded = q8_g32_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scales,
        logical_k: 4,
        padded_k: 128,
    })
    .expect("spec-valid signed edge codes decode");
    assert_eq!(decoded, [-1.0, -127.0, 0.0, 127.0]);

    codes[0] = 0x80;
    assert_eq!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &codes,
            scale_bytes: &scales,
            logical_k: 4,
            padded_k: 128,
        }),
        Err(NativeMtpDecodeError::InvalidQ8Code {
            index: 0,
            code: 0x80,
        })
    );
}

#[test]
fn q4_twos_complement_nibbles_decode_low_lane_before_high_lane() {
    let mut codes = vec![0; 64];
    codes[0] = 0x87;
    codes[1] = 0xf0;
    for index in 4..64 {
        let code_byte = &mut codes[index / 2];
        if index.is_multiple_of(2) {
            *code_byte &= 0xf0;
        } else {
            *code_byte &= 0x0f;
        }
    }
    let decoded = q4_g64_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[0x3c00, 0]),
        logical_k: 4,
        padded_k: 128,
    })
    .expect("Q4 edge nibbles decode");
    assert_eq!(decoded, [7.0, -8.0, 0.0, -1.0]);
}

#[test]
fn q4_short_logical_tail_requires_zero_codes() {
    let mut codes = q4_code_bytes(0, 128);
    codes[63] = 0x01;
    assert_eq!(
        q4_g64_fp16_decode_row(PackedRow {
            codes: &codes,
            scale_bytes: &scale_bytes(&[0x3c00, 0]),
            logical_k: 1,
            padded_k: 128,
        }),
        Err(NativeMtpDecodeError::NonZeroPaddingCode {
            index: 126,
            code: 1,
        })
    );
}

#[test]
fn q8_group_boundaries_at_k32_k64_and_k128_use_the_next_scale() {
    let mut codes = vec![1; 256];
    codes[127] = 0x7f;
    let decoded = q8_g32_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[
            0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800,
        ]),
        logical_k: 256,
        padded_k: 256,
    })
    .expect("group transitions decode");
    assert_eq!(&decoded[31..33], &[1.0, 2.0]);
    assert_eq!(&decoded[63..65], &[2.0, 3.0]);
    assert_eq!(&decoded[127..129], &[127.0 * 4.0, 5.0]);
}

#[test]
fn q4_group_boundaries_at_k64_and_k128_use_the_next_scale() {
    let mut codes = q4_code_bytes(1, 256);
    codes[32] = 0x12;
    codes[64] = 0x31;
    let decoded = q4_g64_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[0x3c00, 0x4000, 0x4200, 0x4400]),
        logical_k: 256,
        padded_k: 256,
    })
    .expect("group transitions decode");
    assert_eq!(&decoded[63..65], &[1.0, 4.0]);
    assert_eq!(&decoded[127..129], &[2.0, 3.0]);
    assert_eq!(decoded[128], 3.0);
}

#[test]
fn q8_full_row_decodes_across_the_k128_split() {
    let codes = vec![1; 256];
    let decoded = q8_g32_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[
            0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800,
        ]),
        logical_k: 256,
        padded_k: 256,
    })
    .expect("full Q8 row spans both K128 halves");
    let expected = [
        [1.0; 32], [2.0; 32], [3.0; 32], [4.0; 32], [5.0; 32], [6.0; 32], [7.0; 32], [8.0; 32],
    ]
    .concat();
    assert_eq!(decoded, expected);
}

#[test]
fn q4_full_row_decodes_across_the_k128_split() {
    let codes = q4_code_bytes(1, 256);
    let decoded = q4_g64_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[0x3c00, 0x4000, 0x4200, 0x4400]),
        logical_k: 256,
        padded_k: 256,
    })
    .expect("full Q4 row spans both K128 halves");
    let expected = [[1.0; 64], [2.0; 64], [3.0; 64], [4.0; 64]].concat();
    assert_eq!(decoded, expected);
}

#[test]
fn q8_spec_codes_and_scales_round_trip_through_row_layout() {
    let signed_codes = [-127_i8, 0, 127, -2];
    let mut codes = vec![0; 128];
    for (index, code) in signed_codes.into_iter().enumerate() {
        codes[index * 32] = code.to_ne_bytes()[0];
    }
    let scales = scale_bytes(&[0x3c00, 0x4000, 0x3c00, 0x3c00]);
    let decoded = q8_g32_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scales,
        logical_k: 128,
        padded_k: 128,
    })
    .expect("Ninfer Q8 logical codes round trip");
    assert_eq!(
        [decoded[0], decoded[32], decoded[64], decoded[96]],
        [-127.0, 0.0, 127.0, -2.0]
    );
}

#[test]
fn q4_spec_codes_and_scales_round_trip_through_row_layout() {
    let signed_codes = [-8_i8, -1, 0, 7, 3];
    let mut codes = vec![0; 64];
    for (index, code) in signed_codes.into_iter().enumerate() {
        let unsigned = code.to_ne_bytes()[0] & 0x0f;
        if index.is_multiple_of(2) {
            codes[index / 2] |= unsigned;
        } else {
            codes[index / 2] |= unsigned << 4;
        }
    }
    let scales = scale_bytes(&[0x3c00, 0x3c00]);
    let decoded = q4_g64_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scales,
        logical_k: 128,
        padded_k: 128,
    })
    .expect("Ninfer Q4 logical codes round trip");
    assert_eq!(&decoded[..5], &[-8.0, -1.0, 0.0, 7.0, 3.0]);
}

#[test]
fn positive_fp16_subnormal_scale_is_preserved_exactly() {
    let mut codes = vec![0; 128];
    codes[0] = 1;
    let decoded = q8_g32_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[0x0001, 0, 0, 0]),
        logical_k: 1,
        padded_k: 128,
    })
    .expect("minimum positive FP16 subnormal scale is supported");
    assert_eq!(decoded[0].to_bits(), (2.0_f32.powi(-24)).to_bits());
}

#[test]
fn zero_scale_requires_zero_codes_and_zero_group_decodes_to_zero() {
    let codes = vec![0; 64];
    let decoded = q4_g64_fp16_decode_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes(&[0x0000, 0]),
        logical_k: 64,
        padded_k: 128,
    })
    .expect("all-zero group accepts positive-zero scale");
    assert_eq!(decoded, vec![0.0; 64]);
    let mut invalid_codes = codes;
    invalid_codes[0] = 1;
    assert_eq!(
        q4_g64_fp16_decode_row(PackedRow {
            codes: &invalid_codes,
            scale_bytes: &scale_bytes(&[0x0000, 0]),
            logical_k: 64,
            padded_k: 128,
        }),
        Err(NativeMtpDecodeError::ZeroScaleNonZeroCode { group: 0, index: 0 })
    );
}

#[test]
fn invalid_fp16_scales_return_typed_boundary_errors() {
    for bits in [0x8000, 0xbc00, 0x7c00, 0xfc00, 0x7e00] {
        let mut scales = [0_u16; 4];
        scales[0] = bits;
        let scale_bytes = scale_bytes(&scales);
        assert_eq!(
            q8_g32_fp16_decode_row(PackedRow {
                codes: &[0; 128],
                scale_bytes: &scale_bytes,
                logical_k: 32,
                padded_k: 128,
            }),
            Err(NativeMtpDecodeError::InvalidScale { group: 0, bits })
        );
    }
}

#[test]
fn logical_tail_padding_requires_zero_code_and_scale() {
    assert_eq!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &[0; 128],
            scale_bytes: &scale_bytes(&[0x3c00, 0x3c00, 0, 0]),
            logical_k: 1,
            padded_k: 128,
        }),
        Err(NativeMtpDecodeError::NonZeroPaddingScale {
            group: 1,
            bits: 0x3c00,
        })
    );
    assert_eq!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &[0; 256],
            scale_bytes: &scale_bytes(&[0x3c00, 0, 0x3c00, 0, 0, 0, 0, 0]),
            logical_k: 1,
            padded_k: 256,
        }),
        Err(NativeMtpDecodeError::NonZeroPaddingScale {
            group: 2,
            bits: 0x3c00,
        })
    );
}
