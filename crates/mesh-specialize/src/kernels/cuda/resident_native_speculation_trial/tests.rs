use super::{fixture, report::{self, Case, CaseReport, CursorReport, Phase}, state};

fn cursor(past: usize) -> CursorReport {
    CursorReport { past, capacity: 20, poisoned: false }
}

fn phase() -> Phase {
    let region = state::compare_region_bytes("cache.k", &[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    let comparison = state::aggregate_state(vec![region], true, ("equal".into(), "equal".into()));
    Phase {
        expected_past: 5, target_cursors: [cursor(5), cursor(5)],
        draft_cursors: [cursor(5), cursor(5)], target_state: Some(comparison.clone()),
        draft_state: Some(comparison), outputs_equal: true, errors: Vec::new(),
    }
}

#[test]
fn forced_rejection_changes_only_requested_position_when_target_prefix_is_valid() {
    let target = [2, 5, 0, 7];

    let proposals = (0..=4).map(|accepted| fixture::forced(&target, accepted, 8).unwrap())
        .collect::<Vec<_>>();

    for (accepted, proposal) in proposals.iter().enumerate() {
        let first_rejection = proposal.iter().zip(target).position(|(actual, expected)| *actual != expected);
        assert_eq!(first_rejection, (accepted < 4).then_some(accepted));
        assert_eq!(proposal.iter().zip(target).filter(|(actual, expected)| **actual != *expected).count(),
            usize::from(accepted < 4));
        assert!(proposal.iter().all(|token| *token < 8));
    }
}

#[test]
fn forced_fixture_rejects_invalid_geometry_when_target_token_is_outside_vocabulary() {
    let target = [1, 2, 3, 8];

    let result = fixture::forced(&target, 2, 8);

    assert!(result.is_err());
}

#[test]
fn complete_region_comparison_fails_when_an_interior_byte_changes() {
    let mut actual = vec![0; 4096];
    actual[2037] = 91;
    let expected = vec![0; 4096];

    let region = state::compare_region_bytes("layer.07.attention.v", &actual, &expected).unwrap();
    let comparison = state::aggregate_state(vec![region], true, ("a".into(), "b".into()));

    assert!(!comparison.equal);
    assert_eq!(comparison.regions[0].length, 4096);
    assert_eq!(comparison.regions[0].first_difference, Some(2037));
    assert_eq!(comparison.regions[0].differing_bytes, 1);
    assert_ne!(comparison.regions[0].left_sha256, comparison.regions[0].right_sha256);
}

#[test]
fn phase_rejects_matching_wrong_cursors_when_both_sides_have_same_wrong_past() {
    let mut report = phase();
    report.target_cursors = [cursor(4), cursor(4)];
    report.draft_cursors = [cursor(4), cursor(4)];

    let passed = report.passed(20);

    assert!(!passed);
}

#[test]
fn phase_rejects_matching_wrong_capacity_when_both_sides_disagree_with_config() {
    let mut report = phase();
    for cursor in report.target_cursors.iter_mut().chain(&mut report.draft_cursors) {
        cursor.capacity = 19;
    }

    let passed = report.passed(20);

    assert!(!passed);
}

#[test]
fn phase_rejects_poison_when_state_bytes_and_past_agree() {
    let mut report = phase();
    report.draft_cursors[1].poisoned = true;

    let passed = report.passed(20);

    assert!(!passed);
}

#[test]
fn case_fails_when_only_continuation_output_differs() {
    let mut report = CaseReport::new(Case { depth: 4, output_tokens: 6, requested_acceptance: Some(4) });
    report.capacity = 20;
    report.acceptance_equal = true;
    report.recovery = Some(phase());
    let mut continuation = phase();
    continuation.outputs_equal = false;
    report.continuation = Some(continuation);
    report.expected_continuation = vec![2, 3];
    report.actual_continuation = vec![2, 4];

    let passed = report.is_correct();

    assert!(!passed);
}

#[test]
fn acceptance_gate_fails_when_run_reports_wrong_first_round_count() {
    let requested = Some(3);
    let actual = Some(4);

    let equal = report::acceptance_equal(requested, actual);

    assert!(!equal);
}

#[test]
fn capacity_probe_preserves_cursor_when_public_begin_rejects_overflow() {
    let capacity = 7;

    let report = fixture::capacity_check(capacity).unwrap();

    assert!(report.passed);
    assert_eq!(report.after.past, capacity);
    assert!(report.rejection.is_some());
}
