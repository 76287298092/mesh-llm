mod capture;
pub(super) mod compare;
mod execute;
pub(super) mod state;

use super::{
    driver::{Context, Module},
    resident_model::{Model, Session},
};
use crate::kernels::DecoderConfig;
use crate::packages::qwen3_8_27b::target_batch_trial::SelectedRows;
use anyhow::{Context as _, Result, ensure};
use serde::Serialize;

const BATCH_PATH: &str = "forward_recorded_record_true_decode_false";
const DECODE_PATH: &str = "forward_detailed_decode_record_false_decode_true";

pub(super) struct Request<'model, 'weights, 'ctx> {
    pub model: &'model Model<'weights, 'ctx>,
    pub context: &'ctx Context,
    pub module: &'model Module<'ctx>,
    pub config: &'model DecoderConfig,
    pub prefix: &'model [u32],
    pub target_tokens: &'model [u32; 5],
    pub continuation: &'model [u32],
    pub selected_rows: Option<SelectedRows>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    kind: &'static str,
    all_passed: bool,
    selected_rows: Option<SelectedRows>,
    native_mtp_admitted: bool,
    timing_claim: bool,
    batch_path: &'static str,
    ordinary_decode_path: &'static str,
    prefix_tokens: usize,
    target_tokens: [u32; 5],
    continuation_tokens: Vec<u32>,
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
struct CaseReport {
    rows: usize,
    passed: bool,
    verification: execute::PhaseReport,
    continuation: execute::PhaseReport,
}

pub(super) fn run(request: Request<'_, '_, '_>) -> Result<serde_json::Value> {
    validate(&request)?;
    let cases = run_cases(request.selected_rows, |rows| run_case(&request, rows));
    let all_passed = cases_passed(request.selected_rows, &cases);
    let report = serde_json::to_value(Report {
        schema_version: 1,
        kind: "resident-target-batch-decode-correctness",
        all_passed,
        selected_rows: request.selected_rows,
        native_mtp_admitted: false,
        timing_claim: false,
        batch_path: BATCH_PATH,
        ordinary_decode_path: DECODE_PATH,
        prefix_tokens: request.prefix.len(),
        target_tokens: *request.target_tokens,
        continuation_tokens: request.continuation.to_vec(),
        cases,
    })
    .context("serialize target batch trial report")?;
    request
        .context
        .synchronize()
        .context("synchronize target batch trial")?;
    Ok(report)
}

fn run_cases(
    selection: Option<SelectedRows>,
    run: impl FnMut(usize) -> CaseReport,
) -> Vec<CaseReport> {
    SelectedRows::cases(selection).map(run).collect()
}

fn cases_passed(selection: Option<SelectedRows>, cases: &[CaseReport]) -> bool {
    match selection {
        Some(rows) => matches!(cases, [case] if case.rows == rows.get() && case.passed),
        None => compare::reports_all_five_cases(
            &cases.iter().map(|case| case.passed).collect::<Vec<_>>(),
        ),
    }
}

fn validate(request: &Request<'_, '_, '_>) -> Result<()> {
    let config = request.config;
    ensure!(!config.layers.is_empty(), "decoder has no target layers");
    ensure!(
        config.hidden > 0 && config.vocabulary > 0,
        "invalid target dimensions"
    );
    ensure!(
        !request.continuation.is_empty(),
        "continuation fixture is empty"
    );
    ensure!(
        request.model.belongs_to(request.context),
        "model context mismatch"
    );
    ensure!(
        request.module.belongs_to(request.context),
        "module context mismatch"
    );
    let fixture_rows = request
        .prefix
        .len()
        .checked_add(5)
        .and_then(|rows| rows.checked_add(request.continuation.len()))
        .context("target trial token count overflows usize")?;
    ensure!(
        fixture_rows <= config.capacity,
        "target trial fixtures exceed session capacity"
    );
    config
        .hidden
        .checked_mul(fixture_rows)
        .context("target hidden fixture extent overflows usize")?;
    let logits_words = config
        .vocabulary
        .checked_mul(5)
        .context("target logits fixture extent overflows usize")?;
    logits_words
        .checked_mul(2)
        .context("target logits fixture byte extent overflows usize")?;
    usize::try_from(config.state_layout.bytes)
        .context("target resident-state extent does not fit usize")?;
    ensure!(
        request
            .prefix
            .iter()
            .chain(request.target_tokens)
            .chain(request.continuation)
            .all(|&token| usize::try_from(token).is_ok_and(|id| id < config.vocabulary)),
        "target trial token is outside vocabulary"
    );
    for rows in 1..=5 {
        request
            .prefix
            .len()
            .checked_add(rows)
            .and_then(|count| count.checked_add(request.continuation.len()))
            .context("target trial cursor extent overflows usize")?;
    }
    Ok(())
}

fn run_case(request: &Request<'_, '_, '_>, rows: usize) -> CaseReport {
    let (prepared, error) = execute::prepare_prefix(request);
    let base = match (prepared, error) {
        (Some(base), None) => base,
        (Some(session), Some(error)) => return failed_prefix_case(Some(session), rows, error),
        (None, Some(error)) => return failed_prefix_case(None, rows, error),
        (None, None) => {
            return failed_case(rows, "prefix preparation produced no session".to_owned());
        }
    };
    if !request.model.validates_session(&base, request.config) {
        return failed_case(
            rows,
            "prepared base session does not match target config".to_owned(),
        );
    }
    let batch = base.fork(request.context);
    let decode = base.fork(request.context);
    let (mut batch_session, mut decode_session) = match (batch, decode) {
        (Ok(batch_session), Ok(decode_session)) => (batch_session, decode_session),
        (Err(batch_error), Err(decode_error)) => {
            return failed_case(
                rows,
                format!("fork both sessions: {batch_error:#}; {decode_error:#}"),
            );
        }
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => {
            return failed_case(rows, format!("fork target session: {error:#}"));
        }
    };
    let verification = execute::compare_phase(
        request,
        &mut batch_session,
        &mut decode_session,
        &request.target_tokens[..rows],
        request.prefix.len() + rows,
        true,
    );
    let continuation = execute::compare_phase(
        request,
        &mut batch_session,
        &mut decode_session,
        request.continuation,
        request.prefix.len() + rows + request.continuation.len(),
        false,
    );
    let passed = verification.passed && continuation.passed;
    CaseReport {
        rows,
        passed,
        verification,
        continuation,
    }
}

fn failed_prefix_case(session: Option<Session<'_>>, rows: usize, error: String) -> CaseReport {
    let mut verification = execute::failed_phase("verification", error.clone());
    let mut continuation = execute::failed_phase("continuation", error);
    if let Some(session) = session {
        execute::attach_unavailable_sessions(&mut verification, &session);
        execute::attach_unavailable_sessions(&mut continuation, &session);
    }
    CaseReport {
        rows,
        passed: false,
        verification,
        continuation,
    }
}

fn failed_case(rows: usize, error: String) -> CaseReport {
    let verification = execute::failed_phase("verification", error.clone());
    let continuation = execute::failed_phase("continuation", error);
    CaseReport {
        rows,
        passed: false,
        verification,
        continuation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_case_retains_exact_mismatch_and_both_failed_phases() {
        let selection = Some(SelectedRows::try_from(3).unwrap());
        let mut visited = Vec::new();
        let cases = run_cases(selection, |rows| {
            visited.push(rows);
            let mut case = failed_case(rows, "retained failure".to_owned());
            let mut words = compare::WordComparison::new();
            words.add(&[10, 11, 12], &[10, 11, 13], 3).unwrap();
            case.verification.hidden = Some(words);
            case
        });
        assert_eq!(visited, vec![3]);
        assert!(!cases_passed(selection, &cases));
        let saved = serde_json::to_value(&cases).unwrap();
        assert_eq!(saved[0]["verification"]["hidden"]["differing_words"], 1);
        assert_eq!(
            saved[0]["verification"]["hidden"]["first_mismatch_index"],
            2
        );
        assert_eq!(saved[0]["verification"]["hidden"]["passed"], false);
        assert_eq!(saved[0]["verification"]["batch_error"], "retained failure");
        assert_eq!(saved[0]["continuation"]["batch_error"], "retained failure");
    }

    #[test]
    fn passing_selection_does_not_qualify_all_five_cases() {
        let selection = Some(SelectedRows::try_from(2).unwrap());
        let cases = run_cases(selection, |rows| {
            let mut case = failed_case(rows, "fixture".to_owned());
            case.passed = true;
            case
        });
        assert!(cases_passed(selection, &cases));
        assert!(!cases_passed(None, &cases));
        assert!(!cases_passed(
            Some(SelectedRows::try_from(1).unwrap()),
            &cases
        ));
        let default_cases = run_cases(None, |rows| {
            let mut case = failed_case(rows, "fixture".to_owned());
            case.passed = rows != 4;
            case
        });
        assert_eq!(
            default_cases
                .iter()
                .map(|case| case.rows)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert!(!cases_passed(None, &default_cases));
    }
}
