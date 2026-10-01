use super::fixture::{DensePattern, identity_view};
use super::{ProjectionKind, compare, reference};
use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView};
use serde_json::json;

#[test]
fn dense_fixtures_when_t1_and_t5_are_generated_cover_each_full_k_row() {
    for kind in [ProjectionKind::QueryKeyValue, ProjectionKind::MlpGateUp] {
        let [_, k] = kind.dimensions();
        for pattern in [DensePattern::AlternatingUnit, DensePattern::SignedMix] {
            let t1 = pattern.input(k, 1).expect("T1 fixture");
            let t5 = pattern.input(k, 5).expect("T5 fixture");
            assert_eq!(t1, t5[..k]);
            assert_eq!(t1.len(), k);
            assert_eq!(t5.len(), 5 * k);
            assert!(columns_are_distinct(&t5, k));
            assert!(
                t5.iter()
                    .all(|&word| super::fixture::input_value(word).abs() <= 2.0)
            );
            assert!(
                (0..5).all(|token| t5[token * k..(token + 1) * k].iter().any(|&word| word != 0))
            );
        }
    }
}

#[test]
fn reference_when_t1_and_t5_schedule_different_split_counts_matches_hand_dots() {
    let (parent, view, input) = one_row_parent(5_120, 1);
    let mut parent = parent;
    let mut input = input;
    for (split, value) in [
        0x3f80, 0x4000, 0xbf80, 0x3f00, 0x3f80, 0x4000, 0xbf80, 0x3f00,
    ]
    .into_iter()
    .enumerate()
    {
        parent[split * 64] = 1;
        input[split * 64] = value;
    }
    let t1 = reference::run(reference::ProjectionReferenceRequest {
        parent: &parent,
        view: &view,
        input_bf16: &input,
        split_warps: 8,
    })
    .expect("T1 reference");
    let t5_input = DensePattern::SignedMix.input(5_120, 5).expect("T5 input");
    let t5 = reference::run(reference::ProjectionReferenceRequest {
        parent: &parent,
        view: &view,
        input_bf16: &t5_input,
        split_warps: 4,
    })
    .expect("T5 reference");
    assert_eq!(t1.scheduled_f32, [5.0]);
    assert_eq!(t1.mathematical_f64, [5.0]);
    assert_eq!(t5.output_bf16.len(), 5);
}

#[test]
fn dense_reference_when_every_row_is_identical_code_reports_five_pairwise_columns() {
    let rows = 8;
    let k = 256;
    let (mut parent, view, _) = one_row_parent(k, rows);
    for row in 0..rows {
        parent[row * k] = 1;
    }
    let input = DensePattern::SignedMix.input(k, 5).expect("T5 fixture");
    let result = reference::run(reference::ProjectionReferenceRequest {
        parent: &parent,
        view: &view,
        input_bf16: &input,
        split_warps: 4,
    })
    .expect("reference");
    let words: Vec<u16> = result.output_bf16;
    assert!((0..5).all(|token| {
        words[token * rows..(token + 1) * rows]
            .iter()
            .all(|&word| word == words[token * rows])
    }));
    assert_eq!(
        words
            .chunks_exact(rows)
            .map(|column| column[0])
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5
    );
}

#[test]
fn exact_comparator_when_both_repeats_share_wrong_result_fails_independently() {
    let expected = reference::ProjectionReference {
        scheduled_f32: vec![1.0; 20],
        output_bf16: vec![0x3f80; 20],
        mathematical_f64: vec![1.0; 20],
        mathematical_error_bound: vec![0.0; 20],
    };
    let mut first = vec![0x3f80; 20];
    let mut second = first.clone();
    first[19] = 0x3f81;
    second[19] = 0x3f81;
    let report = compare::compare(&expected, &[first, second], 4, 5).expect("report");
    assert_eq!(report["all_passed"], json!(false));
    assert_eq!(report["repeat_mismatches"], json!(0));
    assert_eq!(report["exact_bf16_mismatches"], json!(2));
}

#[test]
fn identity_view_when_full_parent_shape_is_reused_preserves_saved_packed_planes() {
    let carrier = Q8MatrixView {
        object_id: "parent".into(),
        shape: [12_288, 5_120],
        padded_k: 5_120,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: 14_336 * 5_120,
        },
        scale_bits: BytePlane {
            offset: 14_336 * 5_120,
            bytes: 14_336 * 5_120 / 32 * 2,
        },
        scale_count: 14_336 * 5_120 / 32,
        source_rows: (0..12_288).collect(),
    };
    let identity = identity_view(&carrier, ProjectionKind::QueryKeyValue).expect("identity view");
    assert_eq!(identity.shape, [14_336, 5_120]);
    assert_eq!(identity.codes, carrier.codes);
    assert_eq!(identity.scale_bits, carrier.scale_bits);
    assert_eq!(identity.source_rows, (0..14_336).collect::<Vec<_>>());
}

#[test]
fn mlp_identity_view_when_saved_gate_is_a_parent_half_preserves_complete_planes() {
    let parent_rows = 34_816_usize;
    let k = 5_120_usize;
    let code_bytes = parent_rows * k;
    let scale_count = parent_rows * (k / 32);
    let scale_offset = code_bytes + (256 - code_bytes % 256) % 256;
    let gate_view = Q8MatrixView {
        object_id: "mtp-mlp-parent".into(),
        shape: [17_408, 5_120],
        padded_k: 5_120,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(code_bytes).expect("parent code extent"),
        },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset).expect("parent scale offset"),
            bytes: u64::try_from(scale_count * 2).expect("parent scale extent"),
        },
        scale_count,
        source_rows: (0..17_408).collect(),
    };

    let identity = identity_view(&gate_view, ProjectionKind::MlpGateUp).expect("full parent view");

    assert_eq!(identity.shape, [34_816, 5_120]);
    assert_eq!(
        identity.codes.bytes,
        u64::try_from(code_bytes).expect("parent code extent")
    );
    assert_eq!(identity.scale_count, scale_count);
    assert_eq!(identity.source_rows, (0..parent_rows).collect::<Vec<_>>());
}

fn one_row_parent(k: usize, rows: usize) -> (Vec<u8>, Q8MatrixView, Vec<u16>) {
    let code_bytes = rows * k;
    let scale_count = rows * (k / 32);
    let scale_offset = code_bytes + (256 - code_bytes % 256) % 256;
    let mut object = vec![0_u8; scale_offset + scale_count * 2];
    for scale in object[scale_offset..].as_chunks_mut::<2>().0 {
        scale.copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    let view = Q8MatrixView {
        object_id: "dense-q8-parent".into(),
        shape: [rows, k],
        padded_k: k,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(code_bytes).expect("code size"),
        },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset).expect("scale offset"),
            bytes: u64::try_from(scale_count * 2).expect("scale bytes"),
        },
        scale_count,
        source_rows: (0..rows).collect(),
    };
    (object, view, vec![0_u16; k])
}

fn columns_are_distinct(words: &[u16], rows: usize) -> bool {
    if rows == 0 || !words.len().is_multiple_of(rows) {
        return false;
    }
    words.chunks_exact(rows).enumerate().all(|(index, column)| {
        words
            .chunks_exact(rows)
            .skip(index + 1)
            .all(|other| column != other)
    })
}
