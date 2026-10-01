use super::{
    HIDDEN, SHORTLIST,
    compare::{CursorSnapshot, WordComparison, compare_cursor_snapshots},
    report::{CaseReport, PhaseReport, ProposalComparison, Report},
    state::{aggregate_state, compare_region_bytes},
};

fn phase() -> PhaseReport {
    let mut phase = PhaseReport::new("fixture", 5);
    let mut logits = vec![0_u16; SHORTLIST];
    logits[11] = 0x3f80;
    logits[21] = 0x3f80;
    let winner = usize::try_from(crate::engine::sampling::greedy(&logits).unwrap()).unwrap();
    phase.shortlist.add(&logits, &logits, SHORTLIST).unwrap();
    phase
        .hidden
        .add(&vec![0; HIDDEN], &vec![0; HIDDEN], HIDDEN)
        .unwrap();
    let region = compare_region_bytes("mtp.attention.k", &[1, 2, 3], &[1, 2, 3]).unwrap();
    let hashes = (region.left_sha256.clone(), region.right_sha256.clone());
    phase.state = Some(aggregate_state(vec![region], true, hashes));
    let cursor = CursorSnapshot {
        past: 5,
        capacity: 32,
        poisoned: false,
    };
    phase.cursor = Some(compare_cursor_snapshots(
        cursor.clone(),
        cursor.clone(),
        cursor,
    ));
    phase.proposals = Some(ProposalComparison {
        tokens: [200_000; 2],
        rows: [winner; 2],
        selected_rows: [winner; 2],
        mapped_tokens: [200_000; 2],
    });
    phase.output_past = Some([5; 2]);
    phase
}

fn cases() -> Vec<CaseReport> {
    (1..=5)
        .map(|rows| (rows, false))
        .chain([(1, true), (4, true)])
        .map(|(count, recursive)| {
            let mut case = CaseReport::new(count, recursive);
            case.phases = (0..if recursive { count + 1 } else { 4 })
                .map(|_| phase())
                .collect();
            case.passed = case.is_consistent();
            case
        })
        .collect()
}

fn aggregate_passed(cases: Vec<CaseReport>) -> bool {
    serde_json::to_value(Report::new(cases)).unwrap()["all_passed"]
        .as_bool()
        .unwrap()
}

#[test]
fn aggregate_rejects_interior_nonwinning_shortlist_word_when_winner_is_unchanged() {
    let mut given = cases();
    let mut left = vec![0_u16; SHORTLIST];
    left[11] = 0x3f80;
    let mut right = left.clone();
    right[65_537] = 1;
    given[0].phases[0].shortlist = WordComparison::new();
    given[0].phases[0]
        .shortlist
        .add(&left, &right, SHORTLIST)
        .unwrap();

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_rejects_hidden_word_when_only_interior_hidden_changes() {
    let mut given = cases();
    let left = vec![0_u16; HIDDEN];
    let mut right = left.clone();
    right[2_561] = 1;
    given[1].phases[0].hidden = WordComparison::new();
    given[1].phases[0]
        .hidden
        .add(&left, &right, HIDDEN)
        .unwrap();

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_rejects_state_when_only_named_region_changes() {
    let mut given = cases();
    let region = compare_region_bytes("mtp.attention.v", &[1, 2, 3], &[1, 4, 3]).unwrap();
    let hashes = (region.left_sha256.clone(), region.right_sha256.clone());
    given[2].phases[0].state = Some(aggregate_state(vec![region], true, hashes));

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_rejects_cursor_when_identical_cursors_advance_incorrectly() {
    let mut given = cases();
    let wrong = CursorSnapshot {
        past: 4,
        capacity: 32,
        poisoned: false,
    };
    let expected = CursorSnapshot {
        past: 5,
        ..wrong.clone()
    };
    given[3].phases[0].cursor = Some(compare_cursor_snapshots(wrong.clone(), wrong, expected));

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_rejects_continuation_when_prior_steps_match() {
    let mut given = cases();
    let continuation = given[6].phases.last_mut().unwrap();
    let mut left = vec![0_u16; HIDDEN];
    left[2_561] = 1;
    continuation.hidden = WordComparison::new();
    continuation
        .hidden
        .add(&left, &vec![0; HIDDEN], HIDDEN)
        .unwrap();

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_rejects_missing_case_when_remaining_cases_match() {
    let mut given = cases();
    given.pop();

    let when = aggregate_passed(given);

    assert!(!when);
}

#[test]
fn aggregate_accepts_complete_comparisons_when_every_case_matches() {
    let given = cases();

    let when = aggregate_passed(given);

    assert!(when);
}

#[test]
fn aggregate_rejects_mapped_token_when_proposal_row_matches() {
    let mut given = cases();
    given[5].phases[0].proposals.as_mut().unwrap().tokens[1] = 11;

    let when = aggregate_passed(given);

    assert!(!when);
}
