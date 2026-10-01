use super::{
    BATCH_PATH, DECODE_PATH, Request,
    capture::{self, Capture},
    compare::{self, CursorComparison, LayerComparison, StateComparison, WordComparison},
};
use crate::kernels::cuda::resident_model::Session;
use anyhow::{Context as _, Result};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub(super) struct PhaseReport {
    pub phase: &'static str,
    pub batch_path: &'static str,
    pub ordinary_decode_path: &'static str,
    pub passed: bool,
    pub batch_completed: bool,
    pub ordinary_decode_completed: bool,
    pub batch_recovery_records: Option<usize>,
    pub ordinary_decode_recovery_records: Option<usize>,
    pub batch_error: Option<String>,
    pub ordinary_decode_error: Option<String>,
    pub compare_error: Option<String>,
    pub batch_session: Option<compare::CursorSnapshot>,
    pub ordinary_decode_session: Option<compare::CursorSnapshot>,
    pub compared_layers: usize,
    pub first_differing_layer: Option<usize>,
    pub layers: Vec<LayerComparison>,
    pub hidden: Option<WordComparison>,
    pub logits: Option<WordComparison>,
    pub state: Option<StateComparison>,
    pub cursor: Option<CursorComparison>,
}

pub(super) fn compare_phase(
    request: &Request<'_, '_, '_>,
    batch_session: &mut Session<'_>,
    decode_session: &mut Session<'_>,
    tokens: &[u32],
    expected_past: usize,
    recorded: bool,
) -> PhaseReport {
    let phase = if recorded {
        "verification"
    } else {
        "continuation"
    };
    let batch = match recorded {
        true => capture::recorded(request, batch_session, tokens),
        false => capture::decode_sequence(request, batch_session, tokens),
    };
    let decode = capture::decode_sequence(request, decode_session, tokens);
    if batch.is_err() {
        batch_session.mark_poisoned();
    }
    if decode.is_err() {
        decode_session.mark_poisoned();
    }
    let (batch, batch_error) = split_capture(batch);
    let (decode, ordinary_decode_error) = split_capture(decode);
    let mut layers = Vec::new();
    let mut hidden = None;
    let mut logits = None;
    let mut compare_error = None;
    if let (Some(batch), Some(decode)) = (&batch, &decode) {
        match compare_outputs(batch, decode, request, tokens.len()) {
            Ok((compared_layers, compared_hidden, compared_logits)) => {
                layers = compared_layers;
                hidden = Some(compared_hidden);
                logits = Some(compared_logits);
            }
            Err(error) => compare_error = Some(format!("compare output words: {error:#}")),
        }
    }
    let both_completed = batch.is_some() && decode.is_some();
    if compare_error.is_some() {
        batch_session.mark_poisoned();
        decode_session.mark_poisoned();
    }
    let synchronization_error = request
        .context
        .synchronize()
        .err()
        .map(|error| format!("synchronize CUDA target paths: {error:#}"));
    if synchronization_error.is_some() {
        batch_session.mark_poisoned();
        decode_session.mark_poisoned();
    }
    let (state, state_error) = if both_completed && synchronization_error.is_none() {
        match compare::compare_states(&batch_session.state, &decode_session.state) {
            Ok(state) => (Some(state), None),
            Err(error) => {
                batch_session.mark_poisoned();
                decode_session.mark_poisoned();
                (
                    None,
                    Some(format!("compare all resident-state regions: {error:#}")),
                )
            }
        }
    } else {
        (None, None)
    };
    let complete = both_completed && synchronization_error.is_none();
    let cursor = if complete {
        Some(compare::compare_cursors(
            (batch_session, decode_session),
            compare::CursorSnapshot {
                past: expected_past,
                capacity: request.config.capacity,
                poisoned: false,
            },
        ))
    } else {
        None
    };
    let outputs_passed = match (&hidden, &logits) {
        (Some(hidden), Some(logits)) => {
            compare::outputs_equal(&layers, hidden, logits, state.as_ref(), cursor.as_ref())
        }
        _ => false,
    };
    let first_differing_layer = compare::first_differing_layer(&layers);
    PhaseReport {
        phase,
        batch_path: if recorded { BATCH_PATH } else { DECODE_PATH },
        ordinary_decode_path: DECODE_PATH,
        passed: outputs_passed
            && batch_error.is_none()
            && ordinary_decode_error.is_none()
            && compare_error.is_none()
            && synchronization_error.is_none()
            && state_error.is_none()
            && compare::validate_layer_outputs(&layers, request.config.layers.len()),
        batch_completed: batch.is_some(),
        ordinary_decode_completed: decode.is_some(),
        batch_recovery_records: batch.as_ref().map(|capture| capture.recovery_records),
        ordinary_decode_recovery_records: decode.as_ref().map(|capture| capture.recovery_records),
        batch_error,
        ordinary_decode_error,
        compare_error: compare_error.or(synchronization_error).or(state_error),
        batch_session: both_completed.then(|| compare::snapshot(batch_session)),
        ordinary_decode_session: both_completed.then(|| compare::snapshot(decode_session)),
        compared_layers: layers.len(),
        first_differing_layer,
        layers,
        hidden,
        logits,
        state,
        cursor,
    }
}

pub(super) fn failed_phase(phase: &'static str, error: String) -> PhaseReport {
    failed_report(phase, Some(error))
}

pub(super) fn prepare_prefix<'ctx>(
    request: &Request<'_, '_, 'ctx>,
) -> (Option<Session<'ctx>>, Option<String>) {
    let mut session = match Session::new(request.context, request.config) {
        Ok(session) => session,
        Err(error) => return (None, Some(format!("create prefix session: {error:#}"))),
    };
    for (index, &token) in request.prefix.iter().enumerate() {
        if let Err(error) = request.model.forward_detailed_decode(
            request.context,
            request.module,
            token,
            &mut session,
            None,
        ) {
            session.mark_poisoned();
            return (
                Some(session),
                Some(format!("decode prefix token {index}: {error:#}")),
            );
        }
    }
    (Some(session), None)
}

pub(super) fn attach_unavailable_sessions(report: &mut PhaseReport, session: &Session<'_>) {
    let snapshot = compare::snapshot(session);
    report.batch_session = Some(snapshot.clone());
    report.ordinary_decode_session = Some(snapshot);
}

fn failed_report(phase: &'static str, error: Option<String>) -> PhaseReport {
    PhaseReport {
        phase,
        batch_path: if phase == "verification" {
            BATCH_PATH
        } else {
            DECODE_PATH
        },
        ordinary_decode_path: DECODE_PATH,
        passed: false,
        batch_completed: false,
        ordinary_decode_completed: false,
        batch_recovery_records: None,
        ordinary_decode_recovery_records: None,
        batch_error: error.clone(),
        ordinary_decode_error: error,
        compare_error: None,
        batch_session: None,
        ordinary_decode_session: None,
        compared_layers: 0,
        first_differing_layer: None,
        layers: Vec::new(),
        hidden: None,
        logits: None,
        state: None,
        cursor: None,
    }
}

fn split_capture(capture: Result<Capture>) -> (Option<Capture>, Option<String>) {
    match capture {
        Ok(capture) => (Some(capture), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    }
}

fn compare_outputs(
    batch: &Capture,
    decode: &Capture,
    request: &Request<'_, '_, '_>,
    rows: usize,
) -> Result<(Vec<LayerComparison>, WordComparison, WordComparison)> {
    let mut layers = Vec::with_capacity(batch.layers.len().max(decode.layers.len()));
    let count = batch.layers.len().max(decode.layers.len());
    for layer in 0..count {
        let left = batch.layers.get(layer).map_or(&[][..], Vec::as_slice);
        let right = decode.layers.get(layer).map_or(&[][..], Vec::as_slice);
        let mut words = WordComparison::new();
        let expected_words = rows
            .checked_mul(request.config.hidden)
            .context("layer hidden extent overflows usize")?;
        words.add(left, right, expected_words)?;
        layers.push(LayerComparison { layer, words });
    }
    let mut hidden = WordComparison::new();
    let expected_hidden = rows
        .checked_mul(request.config.hidden)
        .context("hidden comparison extent overflows usize")?;
    hidden.add(&batch.hidden, &decode.hidden, expected_hidden)?;
    let mut logits = WordComparison::new();
    let expected_logits = rows
        .checked_mul(request.config.vocabulary)
        .context("logit comparison extent overflows usize")?;
    logits.add(&batch.logits, &decode.logits, expected_logits)?;
    Ok((layers, hidden, logits))
}
