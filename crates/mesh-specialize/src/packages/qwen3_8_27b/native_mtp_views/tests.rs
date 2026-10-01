use super::{
    BytePlane, NativeMtpViews, Q8MatrixView,
    selection::{parse_token_map, plan},
    test_fixtures::{exact_map, observed_directory, synthetic_directory},
};
use crate::artifact::ninfer::NinferArtifact;
use serde_json::json;
use std::io::{Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

#[test]
fn resolve_reads_exact_token_map_when_artifact_has_complete_native_mtp_metadata() {
    let mut directory = synthetic_directory();
    let mut payload_bytes = 0_u64;
    for object in &mut directory.objects {
        object.offset = payload_bytes.div_ceil(256) * 256;
        payload_bytes = object.offset + object.bytes;
    }
    let map_offset = directory
        .objects
        .iter()
        .find(|object| object.id == "token-map")
        .expect("fixture token map exists")
        .offset;
    let map_bytes: Vec<u8> = (70_000_i32..201_072).flat_map(i32::to_le_bytes).collect();
    let mut document = serde_json::to_value(directory).expect("serialize complete MTP fixture");
    document["files"] = json!([{"path":null,"payload_bytes":payload_bytes}]);
    let directory_bytes = serde_json::to_vec(&document).expect("serialize fixture directory");
    let directory_len = u64::try_from(directory_bytes.len()).expect("fixture directory length");
    let payload_offset = (32 + directory_len).div_ceil(4096) * 4096;
    let mut file = NamedTempFile::new().expect("create sparse MTP fixture");
    file.write_all(b"NINFER\0\x03").expect("write v3 magic");
    file.write_all(&directory_len.to_le_bytes())
        .expect("write directory length");
    file.write_all(&[0x19; 16]).expect("write artifact ID");
    file.write_all(&directory_bytes)
        .expect("write complete MTP directory");
    file.as_file()
        .set_len(payload_offset + payload_bytes)
        .expect("size sparse payload");
    file.seek(SeekFrom::Start(payload_offset + map_offset))
        .expect("seek to token map");
    file.write_all(&map_bytes).expect("write token-map payload");
    file.flush().expect("flush fixture");
    let mut artifact = NinferArtifact::open(file.path()).expect("open complete native MTP fixture");

    let views = NativeMtpViews::resolve(&mut artifact).expect("resolve streamed token map");

    assert_eq!(views.proposal_tokens.len(), 131_072);
    for (row, expected_id) in (70_000_u32..201_072).enumerate() {
        assert_eq!(
            views
                .proposal_tokens
                .target_id(row)
                .map(|token| token.value()),
            Some(expected_id)
        );
    }
    assert_eq!(views.proposal_tokens.target_id(131_072), None);
}

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
fn q8_view_decodes_rows_from_separate_aligned_parent_planes() {
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
    object[0] = 0xff;
    object[256..258].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let decoded = super::cpu_reference::decode_q8_view_row(&object, &view, 0)
        .expect("checked view decodes its mapped parent row");
    assert_eq!(decoded[0], -1.0);
    assert_eq!(decoded[1], 0.0);
}
