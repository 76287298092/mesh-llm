use super::state::StateComparison;
use crate::engine::session::Cursor;
use serde::Serialize;

#[derive(Clone, Copy, Serialize)]
pub(super) struct Case {
    pub depth: usize,
    pub output_tokens: usize,
    pub requested_acceptance: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub(super) struct CursorReport {
    pub past: usize,
    pub capacity: usize,
    pub poisoned: bool,
}
impl From<&Cursor> for CursorReport {
    fn from(cursor: &Cursor) -> Self {
        Self { past: cursor.past(), capacity: cursor.capacity(), poisoned: cursor.is_poisoned() }
    }
}

pub(super) fn cursors_valid(pair: &[CursorReport; 2], expected: (usize, usize)) -> bool {
    pair.iter().all(|cursor| cursor.past == expected.0
        && cursor.capacity == expected.1 && !cursor.poisoned)
}

#[derive(Serialize)]
pub(super) struct Phase {
    pub expected_past: usize,
    pub target_cursors: [CursorReport; 2],
    pub draft_cursors: [CursorReport; 2],
    pub target_state: Option<StateComparison>,
    pub draft_state: Option<StateComparison>,
    pub outputs_equal: bool,
    pub errors: Vec<String>,
}
impl Phase {
    pub fn passed(&self, capacity: usize) -> bool {
        self.errors.is_empty() && self.outputs_equal
            && cursors_valid(&self.target_cursors, (self.expected_past, capacity))
            && cursors_valid(&self.draft_cursors, (self.expected_past, capacity))
            && self.target_state.as_ref().is_some_and(|state| state.equal)
            && self.draft_state.as_ref().is_some_and(|state| state.equal)
    }
}

#[derive(Serialize)]
pub(super) struct CaseReport {
    pub case: Case,
    pub passed: bool,
    pub stage: &'static str,
    pub capacity: usize,
    pub forced_proposals: Option<Vec<u32>>,
    pub first_round_accepted: Option<usize>,
    pub acceptance_equal: bool,
    pub compact_recovery: Option<bool>,
    pub expected_tokens: Vec<u32>,
    pub actual_tokens: Option<Vec<u32>>,
    pub recovery: Option<Phase>,
    pub continuation: Option<Phase>,
    pub expected_continuation: Vec<u32>,
    pub actual_continuation: Vec<u32>,
    pub errors: Vec<String>,
}
impl CaseReport {
    pub fn new(case: Case) -> Self {
        Self { case, passed: false, stage: "prepare", capacity: 0,
            forced_proposals: None, first_round_accepted: None, acceptance_equal: false,
            compact_recovery: None, expected_tokens: Vec::new(), actual_tokens: None,
            recovery: None, continuation: None, expected_continuation: Vec::new(),
            actual_continuation: Vec::new(), errors: Vec::new() }
    }
    pub fn is_correct(&self) -> bool {
        self.errors.is_empty() && self.acceptance_equal
            && self.recovery.as_ref().is_some_and(|phase| phase.passed(self.capacity))
            && self.continuation.as_ref().is_some_and(|phase| phase.passed(self.capacity))
            && self.expected_continuation.len() == 2 && self.actual_continuation.len() == 2
    }
}

pub(super) fn acceptance_equal(requested: Option<usize>, actual: Option<usize>) -> bool {
    match requested { Some(count) => actual == Some(count), None => actual.is_some() }
}

#[derive(Serialize)]
pub(super) struct CapacityReport {
    pub passed: bool,
    pub exercised: &'static str,
    pub before: CursorReport,
    pub after: CursorReport,
    pub rejection: Option<String>,
}

#[derive(Serialize)]
pub(super) struct Report {
    pub kind: &'static str,
    pub all_passed: bool,
    pub native_mtp_admitted: bool,
    pub source_arithmetic_parity: bool,
    pub performance_claim: bool,
    pub target_batch_gate_replaced: bool,
    pub eos_coverage: &'static str,
    pub oracle: &'static str,
    pub capacity: CapacityReport,
    pub cases: Vec<CaseReport>,
}
