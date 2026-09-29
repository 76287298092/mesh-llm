use super::{decode_q4_view_row, decode_q8_view_row};
use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView, Q8MatrixView};

fn put_scale(object: &mut [u8], offset: usize, value: u16) {
    object[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn q8_parent_row_mapping_and_k128_group_addressing_use_aligned_scale_plane() {
    let codes = BytePlane {
        offset: 0,
        bytes: 512,
    };
    let scales = BytePlane {
        offset: 512,
        bytes: 32,
    };
    let view = Q8MatrixView {
        object_id: "q8".into(),
        shape: [2, 256],
        padded_k: 256,
        group_size: 32,
        codes: codes.clone(),
        scale_bits: scales,
        scale_count: 16,
        source_rows: vec![1, 0],
    };
    let mut object = vec![0; 544];
    object[256] = 2;
    object[256 + 127] = 3;
    object[256 + 128] = 4;
    object[256 + 255] = 5;
    put_scale(&mut object, 512 + 16, 0x4000);
    put_scale(&mut object, 512 + 22, 0x4400);
    put_scale(&mut object, 512 + 24, 0x4800);
    put_scale(&mut object, 512 + 30, 0x4c00);

    let decoded = decode_q8_view_row(&object, &view, 0).expect("mapped Q8 row decodes");

    assert_eq!(decoded[0], 4.0);
    assert_eq!(decoded[127], 12.0);
    assert_eq!(decoded[128], 32.0);
    assert_eq!(decoded[255], 80.0);
}

#[test]
fn q8_plane_alignment_padding_must_be_zero() {
    let view = Q8MatrixView {
        object_id: "q8".into(),
        shape: [1, 128],
        padded_k: 128,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: 128,
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: 8,
        },
        scale_count: 4,
        source_rows: vec![0],
    };
    let mut object = vec![0; 264];
    object[0] = 1;
    put_scale(&mut object, 256, 0x3c00);

    let decoded = decode_q8_view_row(&object, &view, 0).expect("256-byte aligned scales decode");
    assert_eq!(decoded[0], 1.0);
    object[200] = 1;
    assert!(matches!(
        decode_q8_view_row(&object, &view, 0),
        Err(
            super::super::error::NativeMtpDecodeError::NonZeroPlanePadding {
                index: 200,
                byte: 1,
            }
        )
    ));
}

#[test]
fn q4_low_high_nibbles_and_group_crossing_k128_decode_from_row_split_planes() {
    let view = Q4MatrixView {
        object_id: "q4".into(),
        shape: [2, 128],
        padded_k: 128,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: 128,
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: 8,
        },
        scale_count: 4,
        source_rows: vec![1, 0],
    };
    let mut object = vec![0; 264];
    let row_offset = 64;
    object[row_offset] = 0x87;
    object[row_offset + 1] = 0xf8;
    object[row_offset + 32] = 0x23;
    put_scale(&mut object, 256 + 4, 0x3c00);
    put_scale(&mut object, 256 + 6, 0x4000);

    let decoded = decode_q4_view_row(&object, &view, 0).expect("mapped Q4 row decodes");

    assert_eq!(&decoded[..4], &[7.0, -8.0, -8.0, -1.0]);
    assert_eq!(&decoded[64..66], &[6.0, 4.0]);
}
