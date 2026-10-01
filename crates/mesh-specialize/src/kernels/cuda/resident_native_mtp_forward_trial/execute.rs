use super::{HIDDEN, Request, capture, report::PhaseReport};
use crate::kernels::cuda::{
    driver::Buffer,
    resident_native_mtp_forward::{Output, Session},
};
use anyhow::{Context as _, Result};

pub(super) fn case(
    request: &Request<'_, '_, '_>,
    mode: (usize, bool),
    phases: &mut Vec<PhaseReport>,
) -> Result<()> {
    let (count, recursive) = mode;
    let mut left = request.base.fork(request.context)?;
    let mut right = request.base.fork(request.context)?;
    let rows = if recursive { 1 } else { count };
    let input = fixture_rows(request, 0, rows)?;
    let expected = request.base.cursor.past() + rows;
    let mut phase = PhaseReport::new("complete_step_repeat", expected);
    let outputs = run_pair(
        request,
        (&mut left, &mut right),
        (&request.tokens[..rows], &input),
        &mut phase,
    );
    phases.push(phase);
    if recursive {
        let mut previous = outputs;
        for index in 1..count {
            let mut phase = PhaseReport::new("recursive_proposal", expected + index);
            previous = match previous {
                (Some(a), Some(b)) => {
                    run_recursive(request, (&mut left, &mut right), (a, b), &mut phase)
                }
                (None, None) | (Some(_), None) | (None, Some(_)) => {
                    phase
                        .errors
                        .push("recursive predecessor unavailable".to_owned());
                    (None, None)
                }
            };
            phases.push(phase);
        }
    } else {
        let repeat_continuation = match left.fork(request.context) {
            Ok(mut repeat) => continuation(
                request,
                (&mut repeat, &mut right),
                expected + 1,
                "repeat_continuation",
            ),
            Err(error) => {
                let mut phase = PhaseReport::new("repeat_continuation", expected + 1);
                phase.errors.push(format!("fork continuation: {error:#}"));
                phase
            }
        };
        let mut partition = request.base.fork(request.context)?;
        let serial = serial(request, &mut partition, count);
        let mut phase = PhaseReport::new("whole_vs_serial_partition_exact", expected);
        match (outputs.0, serial) {
            (Some(a), Ok(b)) => attach(request, (&left, &partition), (&a, &b), count, &mut phase),
            (None, Ok(_)) => phase.errors.push("whole step unavailable".to_owned()),
            (_, Err(error)) => phase.errors.push(format!("serial partition: {error:#}")),
        }
        phases.push(phase);
        phases.push(continuation(
            request,
            (&mut left, &mut partition),
            expected + 1,
            "partition_continuation",
        ));
        phases.push(repeat_continuation);
    }
    if recursive {
        let end = request.base.cursor.past() + count + 1;
        phases.push(continuation(
            request,
            (&mut left, &mut right),
            end,
            "repeat_continuation",
        ));
    }
    Ok(())
}

fn fixture_rows<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    first: usize,
    rows: usize,
) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(request.context, rows * HIDDEN * 2)?;
    buffer.copy_from_at(
        0,
        request.normalized_target_hidden,
        first * HIDDEN * 2,
        rows * HIDDEN * 2,
    )?;
    Ok(buffer)
}

fn serial<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    session: &mut Session<'ctx>,
    rows: usize,
) -> Result<Output<'ctx>> {
    let hidden = Buffer::new(request.context, rows * HIDDEN * 2)?;
    let mut last = None;
    for row in 0..rows {
        let input = fixture_rows(request, row, 1)?;
        let output = request.model.forward(
            request.context,
            request.module,
            &request.tokens[row..row + 1],
            &input,
            session,
        )?;
        hidden.copy_from_at(row * HIDDEN * 2, &output.hidden, 0, HIDDEN * 2)?;
        last = Some(output);
    }
    let output = last.context("serial partition produced no output")?;
    Ok(Output { hidden, ..output })
}

fn run_pair<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    sessions: (&mut Session<'ctx>, &mut Session<'ctx>),
    input: (&[u32], &Buffer<'_>),
    phase: &mut PhaseReport,
) -> (Option<Output<'ctx>>, Option<Output<'ctx>>) {
    let left = request.model.forward(
        request.context,
        request.module,
        input.0,
        input.1,
        sessions.0,
    );
    let right = request.model.forward(
        request.context,
        request.module,
        input.0,
        input.1,
        sessions.1,
    );
    finish(
        request,
        (&*sessions.0, &*sessions.1),
        (left, right),
        input.0.len(),
        phase,
    )
}

fn run_recursive<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    sessions: (&mut Session<'ctx>, &mut Session<'ctx>),
    previous: (Output<'ctx>, Output<'ctx>),
    phase: &mut PhaseReport,
) -> (Option<Output<'ctx>>, Option<Output<'ctx>>) {
    let left = request.model.forward(
        request.context,
        request.module,
        &[previous.0.token],
        &previous.0.hidden,
        sessions.0,
    );
    let right = request.model.forward(
        request.context,
        request.module,
        &[previous.1.token],
        &previous.1.hidden,
        sessions.1,
    );
    finish(
        request,
        (&*sessions.0, &*sessions.1),
        (left, right),
        1,
        phase,
    )
}

fn finish<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    sessions: (&Session<'ctx>, &Session<'ctx>),
    outputs: (Result<Output<'ctx>>, Result<Output<'ctx>>),
    rows: usize,
    phase: &mut PhaseReport,
) -> (Option<Output<'ctx>>, Option<Output<'ctx>>) {
    let retain = |result: Result<Output<'ctx>>, side: &str, phase: &mut PhaseReport| match result {
        Ok(output) => Some(output),
        Err(error) => {
            phase.errors.push(format!("{side}: {error:#}"));
            None
        }
    };
    let left = retain(outputs.0, "left", phase);
    let right = retain(outputs.1, "right", phase);
    match (&left, &right) {
        (Some(a), Some(b)) => attach(request, sessions, (a, b), rows, phase),
        (None, None) | (Some(_), None) | (None, Some(_)) => {
            if let Err(error) = capture::compare_sessions(sessions, request, phase) {
                phase
                    .errors
                    .push(format!("failed-step state readback: {error:#}"));
            }
        }
    }
    (left, right)
}

fn attach<'ctx>(
    request: &Request<'_, '_, '_>,
    sessions: (&Session<'ctx>, &Session<'ctx>),
    outputs: (&Output<'ctx>, &Output<'ctx>),
    rows: usize,
    phase: &mut PhaseReport,
) {
    if let Err(error) = capture::compare_outputs(
        request,
        capture::Pair {
            outputs,
            rows,
            sessions,
        },
        phase,
    ) {
        phase.errors.push(format!("comparison: {error:#}"));
    }
}

fn continuation<'ctx>(
    request: &Request<'_, '_, 'ctx>,
    sessions: (&mut Session<'ctx>, &mut Session<'ctx>),
    past: usize,
    label: &str,
) -> PhaseReport {
    let mut phase = PhaseReport::new(label, past);
    let _outputs = run_pair(
        request,
        sessions,
        (
            &[request.continuation_token],
            request.normalized_continuation_hidden,
        ),
        &mut phase,
    );
    phase
}
