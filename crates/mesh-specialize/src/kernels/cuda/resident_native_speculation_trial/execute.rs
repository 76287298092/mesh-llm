use super::{Request, fixture, oracle::{self, Oracle}, report::{self, CaseReport, Phase}, state};
use crate::kernels::cuda::resident_speculation::{self, Run};
use anyhow::{Context as _, Result, ensure};

pub(super) fn case(request: &Request<'_, '_, '_, '_>, report: &mut CaseReport) -> Result<()> {
    report.capacity = request.config.capacity;
    let mut expected = oracle::prepare(request)?;
    report.stage = "target-derived forced proposals";
    if let Some(accepted) = report.case.requested_acceptance {
        let target = oracle::proposals(request, &expected)?;
        report.forced_proposals = Some(fixture::forced(&target, accepted, request.config.vocabulary)?);
    }
    report.stage = "ordinary target decode and serial native teacher oracle";
    for _ in 1..report.case.output_tokens {
        oracle::advance(request, &mut expected)?;
    }
    report.expected_tokens.clone_from(&expected.tokens);
    report.stage = "run_native (errors discard sessions in runner API)";
    let mut actual = resident_speculation::run_native(request.context, request.module,
        request.target, request.draft, request.config, &resident_speculation::Request {
            tokens: request.prompt, output_tokens: report.case.output_tokens,
            depth: report.case.depth, forced_first_round: report.forced_proposals.as_deref(),
        })?;
    report.actual_tokens = Some(actual.tokens.clone());
    report.first_round_accepted = actual.first_round_accepted;
    report.compact_recovery = Some(actual.compact_recovery);
    report.acceptance_equal = report::acceptance_equal(report.case.requested_acceptance,
        actual.first_round_accepted);
    if !report.acceptance_equal {
        report.errors.push("first_round_accepted mismatch; target-batch arithmetic/recovery gate failure, not a fixture waiver".to_owned());
    }
    report.stage = "recovery comparison";
    let expected_past = request.prompt.len().checked_add(report.case.output_tokens - 1)
        .context("expected recovery cursor overflow")?;
    report.recovery = Some(compare(&actual, &expected, (expected_past, actual.tokens == expected.tokens)));
    report.stage = "ordinary continuation with native teacher forcing";
    let mut pending = *actual.tokens.last().context("native run emitted no pending token")?;
    let continuation = continue_pair(request, (&mut actual, &mut expected),
        (&mut pending, report));
    let end = expected_past.checked_add(2).context("continuation cursor overflow")?;
    let mut phase = compare(&actual, &expected, (end,
        report.actual_continuation == report.expected_continuation));
    if let Err(error) = continuation { phase.errors.push(format!("continuation: {error:#}")); }
    report.continuation = Some(phase);
    report.stage = "complete";
    Ok(())
}

fn continue_pair(request: &Request<'_, '_, '_, '_>, pair: (&mut Run<'_>, &mut Oracle<'_>),
    output: (&mut u32, &mut CaseReport)) -> Result<()> {
    for _ in 0..2 {
        let expected = oracle::advance(request, pair.1).context("expected continuation")?;
        output.1.expected_continuation.push(expected);
        let target = request.target.forward_detailed_decode(request.context, request.module,
            *output.0, &mut pair.0.target_session, None).context("actual continuation target")?;
        ensure!(target.tokens.len() == 1, "actual continuation selection extent differs");
        let next = *target.tokens.first().context("actual continuation selected no token")?;
        oracle::teacher(request, &mut pair.0.draft_session, (next, &target.hidden))
            .context("actual continuation native teacher")?;
        *output.0 = next;
        output.1.actual_continuation.push(next);
    }
    Ok(())
}

fn compare(actual: &Run<'_>, expected: &Oracle<'_>, outcome: (usize, bool)) -> Phase {
    let mut phase = Phase {
        expected_past: outcome.0, outputs_equal: outcome.1,
        target_cursors: [(&actual.target_session.cursor).into(), (&expected.target.cursor).into()],
        draft_cursors: [(&actual.draft_session.cursor).into(), (&expected.draft.cursor).into()],
        target_state: None, draft_state: None, errors: Vec::new(),
    };
    match state::compare_states(&actual.target_session.state, &expected.target.state) {
        Ok(comparison) => phase.target_state = Some(comparison),
        Err(error) => phase.errors.push(format!("all target state readback: {error:#}")),
    }
    match state::compare_states(&actual.draft_session.state, &expected.draft.state) {
        Ok(comparison) => phase.draft_state = Some(comparison),
        Err(error) => phase.errors.push(format!("all draft state readback: {error:#}")),
    }
    phase
}
