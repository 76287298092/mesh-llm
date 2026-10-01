use super::compare::{CursorComparison, StateComparison, WordComparison, outputs_equal};
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct PhaseReport {
    pub label: String,
    pub errors: Vec<String>,
    pub hidden: WordComparison,
    pub shortlist: WordComparison,
    pub state: Option<StateComparison>,
    pub cursor: Option<CursorComparison>,
    pub proposals: Option<ProposalComparison>,
    pub output_past: Option<[usize; 2]>,
    pub expected_past: usize,
}

#[derive(Serialize)]
pub(super) struct ProposalComparison {
    pub tokens: [u32; 2],
    pub rows: [usize; 2],
    pub selected_rows: [usize; 2],
    pub mapped_tokens: [u32; 2],
}

impl PhaseReport {
    pub fn new(label: &str, expected_past: usize) -> Self {
        Self {
            label: label.to_owned(),
            errors: Vec::new(),
            hidden: WordComparison::new(),
            shortlist: WordComparison::new(),
            state: None,
            cursor: None,
            proposals: None,
            output_past: None,
            expected_past,
        }
    }

    pub fn passed(&self) -> bool {
        self.errors.is_empty()
            && outputs_equal(
                &[],
                &self.hidden,
                &self.shortlist,
                self.state.as_ref(),
                self.cursor.as_ref(),
            )
            && self.output_past == Some([self.expected_past; 2])
            && self.proposals.as_ref().is_some_and(|proposal| {
                proposal.rows == proposal.selected_rows
                    && proposal.tokens == proposal.mapped_tokens
                    && proposal.rows[0] == proposal.rows[1]
                    && proposal.tokens[0] == proposal.tokens[1]
            })
    }
}

#[derive(Serialize)]
pub(super) struct CaseReport {
    pub count: usize,
    pub recursive: bool,
    pub passed: bool,
    pub phases: Vec<PhaseReport>,
    pub errors: Vec<String>,
}

impl CaseReport {
    pub fn new(count: usize, recursive: bool) -> Self {
        Self {
            count,
            recursive,
            passed: false,
            phases: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn is_consistent(&self) -> bool {
        let expected_phases = if self.recursive { self.count + 1 } else { 4 };
        self.errors.is_empty()
            && self.phases.len() == expected_phases
            && self.phases.iter().all(PhaseReport::passed)
    }
}

#[derive(Serialize)]
pub(super) struct Report {
    schema_version: u32,
    kind: &'static str,
    all_passed: bool,
    source_arithmetic_qualified: bool,
    native_mtp_admitted: bool,
    timing_claim: bool,
    model_executable: bool,
    replaces_target_batch_gate: bool,
    comparison_basis: &'static str,
    rows_2_to_4: &'static str,
    proposal_map_contract: &'static str,
    readback_chunk_bytes: usize,
    cases: Vec<CaseReport>,
}

impl Report {
    pub fn new(cases: Vec<CaseReport>) -> Self {
        let complete = cases.len() == 7
            && cases.iter().enumerate().all(|(index, case)| {
                let expected = match index {
                    0..=4 => (index + 1, false),
                    5 => (1, true),
                    6 => (4, true),
                    _ => return false,
                };
                (case.count, case.recursive) == expected && case.is_consistent()
            });
        Self {
            schema_version: 1,
            kind: "resident-native-mtp-loaded-object-consistency",
            all_passed: complete,
            source_arithmetic_qualified: false,
            native_mtp_admitted: false,
            timing_claim: false,
            model_executable: false,
            replaces_target_batch_gate: false,
            comparison_basis: "same implementation repeats and forks; exact BF16 words, not an independent source reference",
            rows_2_to_4: "serial complete T1 steps; correctness only, no speedup claim",
            proposal_map_contract: "caller supplies the ready model's validated signed proposal131072 to target248320 map",
            readback_chunk_bytes: super::READBACK_BYTES,
            cases,
        }
    }
}
