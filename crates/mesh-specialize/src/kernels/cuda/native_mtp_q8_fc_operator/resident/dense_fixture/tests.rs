use super::super::super::compare;
use super::super::super::fixture::{Candidate, Fixture, Kind};
use super::super::super::{FC_GROUP, FC_K, FC_ROWS, SparseInput, compare_sparse, sparse_input};
use super::{DensePattern, GROUPS, columns_are_pairwise_distinct, input};
use crate::{
    native_mtp_q8_sliced_k_fc_reference::{self, Q8SlicedKReference},
    packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView},
};
use serde_json::json;

const EXPECTED_FACTORS: [u16; 5] = [0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x4000];

#[test]
fn dense_inputs_when_five_tokens_cover_all_k_have_distinct_full_columns() {
    for pattern in [DensePattern::AlternatingUnit, DensePattern::SignedDyadic] {
        let words = input(pattern, 5).expect("dense input");
        let single = input(pattern, 1).expect("single-token dense input");
        assert_eq!(words.len(), 5 * FC_K);
        assert_eq!(single, words[..FC_K]);
        assert!(words.iter().all(|&word| word != 0));
        assert!(columns_are_pairwise_distinct(&words, FC_K));
    }
}

#[test]
fn alternating_when_groups_and_lanes_change_match_exact_sign_bits() {
    let words = input(DensePattern::AlternatingUnit, 5).expect("dense input");
    for token in 0..5 {
        for group in 0..GROUPS {
            for lane in 0..FC_GROUP {
                let negative = ((group >> token) & 1) ^ (lane & 1) != 0;
                let expected = if negative { 0xbf80 } else { 0x3f80 };
                assert_eq!(words[token * FC_K + group * FC_GROUP + lane], expected);
            }
        }
    }
    let expected = [
        (0, 0, 0, 0x3f80),
        (0, 0, 1, 0xbf80),
        (0, 1, 0, 0xbf80),
        (0, 1, 1, 0x3f80),
        (0, 32, 30, 0x3f80),
        (0, 319, 31, 0x3f80),
        (1, 0, 0, 0x3f80),
        (1, 1, 0, 0x3f80),
        (1, 2, 0, 0xbf80),
        (1, 319, 0, 0xbf80),
        (4, 16, 0, 0xbf80),
        (4, 16, 1, 0x3f80),
    ];
    for (token, group, lane, bits) in expected {
        assert_eq!(words[token * FC_K + group * FC_GROUP + lane], bits);
    }
}

#[test]
fn signed_dyadic_when_tokens_and_group_edges_change_follow_rotated_factors() {
    let words = input(DensePattern::SignedDyadic, 5).expect("dense input");
    for token in 0..5 {
        for k in 0..FC_K {
            assert_eq!(
                words[token * FC_K + k],
                EXPECTED_FACTORS[(k + token) % EXPECTED_FACTORS.len()]
            );
        }
    }
    let expected = [
        (0, 0, 0x3f80),
        (0, 1, 0xbf80),
        (0, 4, 0x4000),
        (0, 5, 0x3f80),
        (0, 31, 0xbf80),
        (0, 32, 0x3f00),
        (0, 10_239, 0x4000),
        (1, 0, 0xbf80),
        (1, 1, 0x3f00),
        (1, 4, 0x3f80),
        (1, 32, 0xbf00),
        (1, 10_239, 0x3f80),
        (4, 0, 0x4000),
        (4, 10_239, 0xbf00),
    ];
    for (token, k, bits) in expected {
        assert_eq!(words[token * FC_K + k], bits);
    }
    assert_eq!(EXPECTED_FACTORS, [0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x4000]);
}

#[test]
fn dense_patterns_when_packed_unit_scale_reference_runs_match_independent_dots() {
    let alternating_codes = [
        (32_usize, 1_i8),
        (64, 2),
        (128, 4),
        (256, 8),
        (512, 16),
        (1024, 32),
    ];
    let alternating = packed_reference(&alternating_codes, DensePattern::AlternatingUnit);
    assert_eq!(alternating.scheduled_f32, [61.0, 59.0, 55.0, 47.0, 31.0]);
    assert_eq!(
        alternating.output_bf16,
        [0x4274, 0x426c, 0x425c, 0x423c, 0x41f8]
    );
    assert!(columns_are_pairwise_distinct(&alternating.output_bf16, 1));

    let mix_codes = [(2_usize, 1_i8)];
    let mixed = packed_reference(&mix_codes, DensePattern::SignedDyadic);
    assert_eq!(mixed.scheduled_f32, [0.5, -0.5, 2.0, 1.0, -1.0]);
    assert_eq!(mixed.output_bf16, [0x3f00, 0xbf00, 0x4000, 0x3f80, 0xbf80]);
    assert!(columns_are_pairwise_distinct(&mixed.output_bf16, 1));
}

#[test]
fn group_sweep_when_t5_covers_every_group_and_factor() {
    let words = sparse_input(SparseInput::GroupSweep, 5).expect("input");
    assert_eq!(words.iter().filter(|&&word| word != 0).count(), 5 * GROUPS);
    for token in 0..5 {
        for group in 0..GROUPS {
            let lane = group % FC_GROUP;
            assert_eq!(
                words[token * FC_K + group * FC_GROUP + lane],
                EXPECTED_FACTORS[(group + token) % EXPECTED_FACTORS.len()]
            );
        }
    }
}

#[test]
fn last_k_when_t1_and_t5_target_group_319_second_lane_31() {
    for tokens in [1, 5] {
        let words = sparse_input(SparseInput::LastK, tokens).expect("input");
        assert_eq!(words.iter().filter(|&&word| word != 0).count(), tokens);
        for token in 0..tokens {
            assert_eq!(words[token * FC_K + FC_K - 1], EXPECTED_FACTORS[token % 5]);
        }
    }
}

#[test]
fn cancellation_when_five_columns_are_distinct_matches_simple_oracle() {
    let fixture = Fixture::new(Candidate::C8, Kind::Cancellation).expect("fixture");
    let expected = reference(&fixture.object, &fixture.view, &fixture.input);
    let actual: Vec<u16> = (0..5 * FC_ROWS)
        .map(|index| {
            fixture
                .simple_expected(index / FC_ROWS, index % FC_ROWS)
                .expect("simple oracle")
                .expect("simple case")
        })
        .collect();
    let report = compare::run(&fixture, &expected, &[actual.clone(), actual]).expect("comparison");
    assert_eq!(report["all_passed"], true);
    let first_row: Vec<u16> = expected
        .output_bf16
        .chunks(fixture.view.source_rows.len())
        .map(|column| column[0])
        .collect();
    assert_eq!(
        first_row
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5
    );
}

#[test]
fn comparison_fails_when_last_row_is_poisoned_or_repeat_changes() {
    let fixture = Fixture::new(Candidate::C4, Kind::LastK).expect("fixture");
    let expected = reference(&fixture.object, &fixture.view, &fixture.input);
    let actual: Vec<u16> = (0..FC_ROWS)
        .map(|row| {
            fixture
                .simple_expected(0, row)
                .expect("simple oracle")
                .expect("simple case")
        })
        .collect();
    let mut changed = actual.clone();
    changed[FC_ROWS - 1] = 0x7fc1;
    let report = compare::run(&fixture, &expected, &[actual, changed]).expect("comparison");
    assert_eq!(report["all_passed"], false);
    assert_eq!(report["nonfinite_outputs"], 1);
    assert_eq!(report["repeat_mismatches"], 1);
    assert_eq!(report["selected_oracle_mismatches"], 1);
}

#[test]
fn exact_gate_when_repeat_or_reference_word_differs() {
    let expected = Q8SlicedKReference {
        scheduled_f32: vec![1.0; FC_ROWS],
        output_bf16: vec![0x3f80; FC_ROWS],
        mathematical_f64: vec![1.0; FC_ROWS],
        mathematical_error_bound: vec![0.0; FC_ROWS],
    };
    let mut second = vec![0x3f80; FC_ROWS];
    second[FC_ROWS - 1] = 0x3f81;
    let report =
        compare_sparse(&expected, &[vec![0x3f80; FC_ROWS], second], 1).expect("comparison report");
    assert_eq!(report["all_passed"], json!(false));
    assert_eq!(report["repeat_mismatches"], json!(1));
    assert_eq!(report["exact_bf16_mismatches"], json!(1));
    assert_eq!(report["failures_first_16"][0]["row"], json!(FC_ROWS - 1));
}

#[test]
fn exact_comparator_when_token_four_repeat_two_last_row_is_poisoned_fails() {
    let expected = expected_reference(5 * FC_ROWS);
    let first = vec![0x3f80; 5 * FC_ROWS];
    let mut second = first.clone();
    second[5 * FC_ROWS - 1] = 0x7fc1;
    let report = compare_sparse(&expected, &[first, second], 5).expect("comparison");
    assert_eq!(report["all_passed"], json!(false));
    assert_eq!(report["repeat_mismatches"], json!(1));
    assert_eq!(report["exact_bf16_mismatches"], json!(1));
    assert_eq!(report["failures_first_16"][0]["token"], json!(4));
    assert_eq!(report["failures_first_16"][0]["row"], json!(FC_ROWS - 1));
}

#[test]
fn exact_comparator_when_both_repeats_share_wrong_last_word_still_fails() {
    let expected = expected_reference(5 * FC_ROWS);
    let mut first = vec![0x3f80; 5 * FC_ROWS];
    let mut second = first.clone();
    first[5 * FC_ROWS - 1] = 0x3f81;
    second[5 * FC_ROWS - 1] = 0x3f81;
    let report = compare_sparse(&expected, &[first, second], 5).expect("comparison");
    assert_eq!(report["all_passed"], json!(false));
    assert_eq!(report["repeat_mismatches"], json!(0));
    assert_eq!(report["exact_bf16_mismatches"], json!(2));
}

fn packed_reference(codes: &[(usize, i8)], pattern: DensePattern) -> Q8SlicedKReference {
    let mut object = vec![0_u8; FC_K + GROUPS * 2];
    for &(k, code) in codes {
        object[k] = code.to_ne_bytes()[0];
    }
    for scale in object[FC_K..].as_chunks_mut::<2>().0 {
        scale.copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    let view = Q8MatrixView {
        object_id: "dense-fixture-one-row".into(),
        shape: [1, FC_K],
        padded_k: FC_K,
        group_size: FC_GROUP,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(FC_K).expect("Q8 code byte extent"),
        },
        scale_bits: BytePlane {
            offset: u64::try_from(FC_K).expect("Q8 scale offset"),
            bytes: u64::try_from(GROUPS * 2).expect("Q8 scale byte extent"),
        },
        scale_count: GROUPS,
        source_rows: vec![0],
    };
    let input = input(pattern, 5).expect("dense input");
    native_mtp_q8_sliced_k_fc_reference::run(&object, &view, &input).expect("reference")
}

fn expected_reference(count: usize) -> Q8SlicedKReference {
    Q8SlicedKReference {
        scheduled_f32: vec![1.0; count],
        output_bf16: vec![0x3f80; count],
        mathematical_f64: vec![1.0; count],
        mathematical_error_bound: vec![0.0; count],
    }
}

fn reference(object: &[u8], view: &Q8MatrixView, input: &[u16]) -> Q8SlicedKReference {
    native_mtp_q8_sliced_k_fc_reference::run(object, view, input).expect("reference")
}
