use super::super::state::{aggregate_state, compare_region_bytes};
use super::{
    CursorSnapshot, LayerComparison, StateComparison, WordComparison, compare_cursor_snapshots,
    first_differing_layer, outputs_equal,
};

fn state_from_bytes(left: &[u8], right: &[u8]) -> StateComparison {
    let region = compare_region_bytes("layer.state", left, right).unwrap();
    let hashes = (region.left_sha256.clone(), region.right_sha256.clone());
    aggregate_state(vec![region], true, hashes)
}

fn valid_cursor(past: usize, capacity: usize) -> super::CursorComparison {
    let expected = CursorSnapshot {
        past,
        capacity,
        poisoned: false,
    };
    compare_cursor_snapshots(expected.clone(), expected.clone(), expected)
}

#[test]
fn full_word_comparison_rejects_an_interior_nonwinner_difference() {
    let left = [0_u16, 9, 0, 7];
    let right = [0_u16, 9, 1, 7];
    let mut comparison = WordComparison::new();

    comparison.add(&left, &right, 4).unwrap();

    assert!(!comparison.passed());
    assert_eq!(comparison.differing_words, 1);
    assert_eq!(comparison.first_mismatch_index, Some(2));
}

#[test]
fn state_comparison_rejects_a_region_only_difference() {
    let report = state_from_bytes(&[1, 2, 3], &[1, 2, 4]);
    let comparison = &report.regions[0];
    assert!(!comparison.identical);
    assert_eq!(comparison.differing_bytes, 1);
    assert_eq!(comparison.first_difference, Some(2));
    assert!(!report.equal);
    assert_eq!(
        report.first_differing_region.as_deref(),
        Some("layer.state")
    );
}

#[test]
fn cursor_comparison_rejects_cursor_only_differences() {
    let left = CursorSnapshot {
        past: 8,
        capacity: 32,
        poisoned: false,
    };
    let right = CursorSnapshot {
        past: 9,
        capacity: 32,
        poisoned: false,
    };

    let expected = CursorSnapshot {
        past: 9,
        capacity: 32,
        poisoned: false,
    };
    assert!(!compare_cursor_snapshots(left, right, expected).equal);
}

#[test]
fn continuation_comparison_rejects_hidden_only_differences() {
    let mut hidden = WordComparison::new();
    hidden.add(&[4, 5], &[4, 6], 2).unwrap();
    let mut logits = WordComparison::new();
    logits.add(&[8, 9], &[8, 9], 2).unwrap();
    let mut layer_words = WordComparison::new();
    layer_words.add(&[1, 2], &[1, 2], 2).unwrap();
    let layer = LayerComparison {
        layer: 0,
        words: layer_words,
    };
    let state = state_from_bytes(&[1, 2], &[1, 2]);
    let cursor = valid_cursor(4, 8);

    assert!(!outputs_equal(
        &[layer],
        &hidden,
        &logits,
        Some(&state),
        Some(&cursor)
    ));
}

#[test]
fn nonwinning_logits_difference_fails_output_gate() {
    let mut hidden = WordComparison::new();
    hidden.add(&[1, 2], &[1, 2], 2).unwrap();
    let mut logits = WordComparison::new();
    logits.add(&[9, 1, 0], &[9, 2, 0], 3).unwrap();
    let state = state_from_bytes(&[1, 2], &[1, 2]);
    let cursor = valid_cursor(3, 8);

    assert!(!outputs_equal(
        &[],
        &hidden,
        &logits,
        Some(&state),
        Some(&cursor)
    ));
}

#[test]
fn single_token_expected_output_rejects_extra_words() {
    let mut comparison = WordComparison::new();
    comparison.add(&[1, 2], &[1, 2], 1).unwrap();

    assert!(!comparison.passed());
}

#[test]
fn region_only_difference_fails_the_aggregate_phase_gate() {
    let state = state_from_bytes(&[1, 2], &[1, 3]);
    let mut words = WordComparison::new();
    words.add(&[1], &[1], 1).unwrap();
    let cursor = valid_cursor(1, 8);

    assert!(!outputs_equal(
        &[],
        &words,
        &words,
        Some(&state),
        Some(&cursor)
    ));
}

#[test]
fn equal_region_bytes_pass_the_aggregate_phase_gate() {
    let state = state_from_bytes(&[1, 2], &[1, 2]);
    let mut words = WordComparison::new();
    words.add(&[1], &[1], 1).unwrap();
    let cursor = valid_cursor(1, 8);

    assert!(outputs_equal(
        &[],
        &words,
        &words,
        Some(&state),
        Some(&cursor)
    ));
}

#[test]
fn identical_wrong_or_poisoned_cursors_fail_the_phase_gate() {
    let state = state_from_bytes(&[1], &[1]);
    let mut words = WordComparison::new();
    words.add(&[1], &[1], 1).unwrap();
    for expected_past in [33, 34, 35, 36, 37, 38, 39, 40] {
        let expected = CursorSnapshot {
            past: expected_past,
            capacity: 40,
            poisoned: false,
        };
        for wrong in [
            CursorSnapshot {
                past: expected_past - 1,
                ..expected.clone()
            },
            CursorSnapshot {
                past: expected_past + 1,
                ..expected.clone()
            },
            CursorSnapshot {
                capacity: 41,
                ..expected.clone()
            },
            CursorSnapshot {
                poisoned: true,
                ..expected.clone()
            },
        ] {
            let cursor = compare_cursor_snapshots(wrong.clone(), wrong, expected.clone());

            assert!(!outputs_equal(
                &[],
                &words,
                &words,
                Some(&state),
                Some(&cursor)
            ));
        }
    }
}

#[test]
fn identical_bytes_with_different_layouts_fail_state_aggregation() {
    let region = compare_region_bytes("state", &[1], &[1]).unwrap();
    let hashes = (region.left_sha256.clone(), region.right_sha256.clone());

    let state = aggregate_state(vec![region], false, hashes);

    assert!(!state.equal);
}

#[test]
fn differing_region_bytes_fail_even_when_aggregate_hashes_match() {
    let region = compare_region_bytes("state", &[1, 2], &[1, 3]).unwrap();
    let hash = region.left_sha256.clone();

    let state = aggregate_state(vec![region], true, (hash.clone(), hash));

    assert!(!state.equal);
}

#[test]
fn all_five_rows_are_required_for_a_complete_report() {
    assert!(!super::reports_all_five_cases(&[true; 4]));
    assert!(!super::reports_all_five_cases(&[
        true, true, false, true, true
    ]));
    assert!(super::reports_all_five_cases(&[true; 5]));
}

#[test]
fn reports_the_first_layer_with_a_word_difference() {
    let mut equal = WordComparison::new();
    equal.add(&[1], &[1], 1).unwrap();
    let mut different = WordComparison::new();
    different.add(&[1], &[2], 1).unwrap();
    let layers = [
        LayerComparison {
            layer: 0,
            words: equal.clone(),
        },
        LayerComparison {
            layer: 1,
            words: different,
        },
    ];

    assert_eq!(first_differing_layer(&layers), Some(1));
}
