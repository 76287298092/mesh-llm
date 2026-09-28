use super::{
    BytePlane,
    cpu_reference::{
        PackedRow, q4_g64_fp16_dot, q4_g64_fp16_row, q8_g32_fp16_decode_row, q8_g32_fp16_dot,
        q8_g32_fp16_encoded_row,
    },
    selection::{parse_token_map, plan},
    test_fixtures::{exact_map, observed_directory, synthetic_directory},
};
#[test]
fn selected_q8_view_has_expected_parent_planes() {
    let directory = synthetic_directory();
    let selected = plan(&directory).expect("synthetic metadata matches contract");
    assert_eq!(selected.fc.shape, [5_120, 10_240]);
    assert_eq!(selected.fc.padded_k, 10_240);
    assert_eq!(
        selected.fc.codes,
        BytePlane {
            offset: 0,
            bytes: 52_428_800
        }
    );
    assert_eq!(
        selected.fc.scale_bits,
        BytePlane {
            offset: 52_428_800,
            bytes: 3_276_800
        }
    );
    assert_eq!(selected.fc.scale_count, 1_638_400);
    assert_eq!(selected.fc.scale_bits.bytes, 3_276_800);
}

#[test]
fn selected_qkv_views_retain_checked_parent_rows() {
    let selected = plan(&synthetic_directory()).expect("synthetic metadata matches contract");
    assert_eq!(selected.key.source_rows, (6_144..7_168).collect::<Vec<_>>());
    assert_eq!(
        selected.value.source_rows,
        (13_312..14_336).collect::<Vec<_>>()
    );
    assert_eq!(selected.query_gate.source_rows[0], 0);
    assert_eq!(selected.query_gate.source_rows[256], 7_168);
    assert_eq!(selected.query_gate.source_rows[512], 256);
    assert_eq!(selected.query_gate.source_rows[768], 7_424);
}

#[test]
fn proposal_head_has_shortlist_shape_and_checked_planes() {
    let selected = plan(&synthetic_directory()).expect("synthetic metadata matches contract");
    assert_eq!(selected.proposal_head.shape, [131_072, 5_120]);
    assert_eq!(selected.proposal_head.padded_k, 5_120);
    assert_eq!(selected.proposal_head.codes.bytes, 335_544_320);
    assert_eq!(selected.proposal_head.scale_bits.bytes, 20_971_520);
    assert_eq!(selected.proposal_head.scale_count, 10_485_760);
    assert_eq!(selected.norms.query.elements, 256);
}

#[test]
fn inspected_artifact_mtp_bindings_resolve_with_source_plane_metadata() {
    let selected = plan(&observed_directory()).expect("inspected artifact MTP records resolve");
    assert_eq!(selected.fc.object_id, "weight/000725");
    assert_eq!(selected.query_gate.object_id, "weight/000176");
    assert_eq!(selected.query_gate.source_rows[256], 7_168);
    assert_eq!(selected.mlp_gate.object_id, "weight/000177");
    assert_eq!(selected.mlp_up.source_rows[0], 17_408);
    assert_eq!(selected.proposal_head.object_id, "weight/001070");
    assert_eq!(selected.token_map_object, "weight/001071");
}

#[test]
fn inspected_artifact_token_map_metadata_is_signed_int32() {
    let directory = observed_directory();
    let token_map = directory
        .objects
        .iter()
        .find(|object| object.id == "weight/001071")
        .expect("proposal map object exists");
    assert_eq!(token_map.format.as_deref(), Some("int32"));
    assert_eq!(token_map.layout.as_deref(), Some("contiguous_le_v1"));
    assert_eq!(token_map.shape, [131_072]);
    assert_eq!(token_map.bytes, 524_288);
}

#[test]
fn malformed_quantization_and_binding_are_rejected() {
    let mut directory = synthetic_directory();
    directory
        .objects
        .iter_mut()
        .find(|object| object.id == "fc")
        .expect("fixture FC exists")
        .format = Some("unknown".into());
    assert!(plan(&directory).is_err());

    let mut directory = synthetic_directory();
    directory.bindings.remove("mtp/input_projection");
    assert!(plan(&directory).is_err());
}

#[test]
fn signed_token_map_rejects_negative_and_out_of_range_ids() {
    let valid = parse_token_map(&exact_map(247_999)).expect("valid target IDs");
    assert_eq!(valid.len(), 131_072);
    assert_eq!(
        valid.target_id(0).expect("row zero exists").value(),
        247_999
    );
    for invalid_id in [-1, 248_320, i32::MIN] {
        assert!(parse_token_map(&exact_map(invalid_id)).is_err());
    }
    assert!(parse_token_map(&[0; 3]).is_err());
}

#[test]
fn unsupported_row_split_codec_is_rejected_at_the_selected_projection() {
    let mut directory = synthetic_directory();
    directory
        .objects
        .iter_mut()
        .find(|object| object.id == "fc")
        .expect("fixture FC exists")
        .format = Some("q5_g64_fp16".into());
    assert!(plan(&directory).is_err());
}

#[test]
fn independent_q8_reference_decodes_signed_codes_scale_bits_and_dots() {
    let mut codes = vec![0; 128];
    codes[0] = 0xff;
    codes[1] = 2;
    codes[2] = 0x80;
    codes[3] = 0x7f;
    let scale_bytes = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
    let encoded = q8_g32_fp16_encoded_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes,
        logical_k: 4,
        padded_k: 128,
    })
    .expect("Q8 codes sign-extend");
    assert_eq!(encoded.signed_codes, [-1, 2, i8::MIN, i8::MAX]);
    assert_eq!(encoded.scale_bits, [0x3c00; 4]);
    assert!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &codes,
            scale_bytes: &scale_bytes,
            logical_k: 2,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q8_g32_fp16_dot(
            PackedRow {
                codes: &codes,
                scale_bytes: &scale_bytes,
                logical_k: 2,
                padded_k: 128,
            },
            &[2.0, 3.0],
        )
        .is_err()
    );
}

#[test]
fn independent_q8_reference_handles_extreme_scale_and_bad_buffers() {
    let mut codes = vec![0; 128];
    codes[0] = 0x7f;
    assert!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &codes,
            scale_bytes: &[0xff, 0x7b, 0xff, 0x7b, 0xff, 0x7b, 0xff, 0x7b],
            logical_k: 1,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &codes,
            scale_bytes: &[0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00],
            logical_k: 1,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &[0; 127],
            scale_bytes: &[0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c],
            logical_k: 1,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q8_g32_fp16_decode_row(PackedRow {
            codes: &[0; 128],
            scale_bytes: &[0x00, 0x7c, 0x00, 0x7c, 0x00, 0x7c, 0x00, 0x7c],
            logical_k: 1,
            padded_k: 128,
        })
        .is_err()
    );
}

#[test]
fn independent_q8_reference_preserves_scale_bits_without_numeric_assumptions() {
    let codes = [0x80; 128];
    let scale_bytes = [0xff, 0x7b, 0x01, 0x00, 0x00, 0x7c, 0x00, 0xfc];
    let encoded = q8_g32_fp16_encoded_row(PackedRow {
        codes: &codes,
        scale_bytes: &scale_bytes,
        logical_k: 1,
        padded_k: 128,
    })
    .expect("well-shaped packed bytes");
    assert_eq!(encoded.signed_codes, [i8::MIN]);
    assert_eq!(encoded.scale_bits, [0x7bff, 0x0001, 0x7c00, 0xfc00]);
}

#[test]
fn q4_reference_rejects_malformed_geometry_and_undocumented_codec() {
    let scales = [0x00, 0x3c, 0x00, 0x3c];
    assert!(
        q4_g64_fp16_row(PackedRow {
            codes: &[0; 63],
            scale_bytes: &scales,
            logical_k: 1,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q4_g64_fp16_row(PackedRow {
            codes: &[0; 64],
            scale_bytes: &scales,
            logical_k: 128,
            padded_k: 128,
        })
        .is_err()
    );
    assert!(
        q4_g64_fp16_dot(
            PackedRow {
                codes: &[0; 64],
                scale_bytes: &scales,
                logical_k: 1,
                padded_k: 128,
            },
            &[1.0],
        )
        .is_err()
    );
}
