use super::{
    HIDDEN, Request, SHORTLIST, compare,
    report::{PhaseReport, ProposalComparison},
    state,
};
use crate::{
    engine::sampling,
    kernels::cuda::{
        driver::Buffer,
        resident_native_mtp_forward::{Output, Session},
    },
};
use anyhow::{Result, ensure};

pub(super) fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    ensure!(
        buffer.len() <= 5 * HIDDEN * 2 && buffer.len().is_multiple_of(2),
        "trial hidden readback exceeds bound or has odd extent"
    );
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

pub(super) struct Pair<'a, 'ctx> {
    pub outputs: (&'a Output<'ctx>, &'a Output<'ctx>),
    pub rows: usize,
    pub sessions: (&'a Session<'ctx>, &'a Session<'ctx>),
}

pub(super) fn compare_outputs(
    request: &Request<'_, '_, '_>,
    pair: Pair<'_, '_>,
    phase: &mut PhaseReport,
) -> Result<()> {
    let (left, right) = pair.outputs;
    phase.hidden.add(
        &words(&left.hidden)?,
        &words(&right.hidden)?,
        pair.rows * HIDDEN,
    )?;
    phase
        .shortlist
        .add(&left.logits, &right.logits, SHORTLIST)?;
    phase.output_past = Some([left.past, right.past]);
    let selected_rows = [
        usize::try_from(sampling::greedy(&left.logits)?)?,
        usize::try_from(sampling::greedy(&right.logits)?)?,
    ];
    let mapped = |row| {
        request
            .proposal_tokens
            .target_id(row)
            .map(|token| token.value())
            .ok_or_else(|| anyhow::anyhow!("trial proposal map row missing"))
    };
    phase.proposals = Some(ProposalComparison {
        tokens: [left.token, right.token],
        rows: [left.proposal_row, right.proposal_row],
        selected_rows,
        mapped_tokens: [mapped(selected_rows[0])?, mapped(selected_rows[1])?],
    });
    compare_sessions(pair.sessions, request, phase)
}

pub(super) fn compare_sessions(
    sessions: (&Session<'_>, &Session<'_>),
    request: &Request<'_, '_, '_>,
    phase: &mut PhaseReport,
) -> Result<()> {
    let snapshot = |session: &Session<'_>| compare::CursorSnapshot {
        past: session.cursor.past(),
        capacity: session.cursor.capacity(),
        poisoned: session.cursor.is_poisoned(),
    };
    phase.cursor = Some(compare::compare_cursor_snapshots(
        snapshot(sessions.0),
        snapshot(sessions.1),
        compare::CursorSnapshot {
            past: phase.expected_past,
            capacity: request.config.capacity,
            poisoned: false,
        },
    ));
    phase.state = Some(state::compare_states(&sessions.0.state, &sessions.1.state)?);
    Ok(())
}
