use super::{LoadedModels, Request, accounting, evidence};
use crate::kernels::cuda::{resident_model::Session, resident_speculation};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Observed<'ctx> {
    pub tokens: Vec<u32>,
    pub target: Session<'ctx>,
    pub draft: Option<crate::kernels::cuda::resident_mtp::Session<'ctx>>,
    pub costs: Value,
    pub decode_seconds: f64,
    pub requested_output_tokens: usize,
}

pub(super) fn baseline<'ctx>(
    models: &LoadedModels<'_, '_, 'ctx>,
    request: &Request<'_>,
) -> Result<Observed<'ctx>> {
    let context = models.context;
    context.synchronize()?;
    let wall_start = Instant::now();
    let mut session = Session::new(context, models.config)?;
    context.synchronize()?;
    let setup_seconds = wall_start.elapsed().as_secs_f64();
    let prefill_start = Instant::now();
    let prefill =
        models
            .target
            .forward(context, models.module, request.prompt, &mut session, None)?;
    context.synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let mut tokens = Vec::with_capacity(request.output_tokens);
    tokens.push(prefill.token);
    let mut pending = prefill.token;
    context.synchronize()?;
    let decode_start = Instant::now();
    for _ in 1..request.output_tokens {
        let output =
            models
                .target
                .forward_decode(context, models.module, &[pending], &mut session)?;
        pending = output.token;
        tokens.push(pending);
    }
    context.synchronize()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let wall_seconds = wall_start.elapsed().as_secs_f64();
    Ok(Observed {
        tokens,
        target: session,
        draft: None,
        decode_seconds,
        requested_output_tokens: request.output_tokens,
        costs: json!({
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "wall_seconds_including_setup": wall_seconds, "session_setup_seconds": setup_seconds,
            "phase_seconds": null, "device_elapsed_seconds": null,
            "decode_method": "one synchronized outer host interval around all ordinary decode forwards and token collection",
            "prefill_scope": "target prompt forward and final logits/first greedy token, excluding session allocation"
        }),
    })
}

pub(super) fn native<'ctx>(
    models: &LoadedModels<'_, '_, 'ctx>,
    request: &Request<'_>,
    depth: usize,
) -> Result<Observed<'ctx>> {
    let context = models.context;
    context.synchronize()?;
    let wall_start = Instant::now();
    let run = resident_speculation::run_native(
        context,
        models.module,
        models.target,
        models.draft,
        models.config,
        &resident_speculation::Request {
            tokens: request.prompt,
            output_tokens: request.output_tokens,
            depth,
            forced_first_round: None,
        },
    )?;
    context.synchronize()?;
    let wall_seconds = wall_start.elapsed().as_secs_f64();
    let phases = &run.phase_seconds;
    let attribution = accounting::attribution(
        run.decode_seconds,
        &[
            phases.draft,
            phases.verification,
            phases.replay,
            phases.teacher,
            phases.fork_copy,
        ],
        true,
    );
    let costs = json!({
        "prefill_seconds": run.prefill_seconds, "decode_seconds": run.decode_seconds,
        "wall_seconds_including_setup": wall_seconds,
        "session_setup_seconds": null, "device_elapsed_seconds": null,
        "phase_seconds": run.phase_seconds, "attribution": attribution,
        "phase_scope": "run_native host attribution: draft excludes measured draft fork; verification/replay/teacher and explicit branch forks are sequential, not added to decode total",
        "fork_copy_scope": "only explicit proposal/verification branch forks; hidden copies and native serial forks inside teacher/prefill remain inside those intervals, not separately exposed",
        "unavailable_costs": ["session allocation alone", "prefill target versus draft split", "hidden-copy alone", "native internal serial fork alone", "device elapsed"],
        "prefill_scope": "target and shifted native draft prefill, final selection and hidden copies; excludes initial session allocations",
        "decode_method": "run_native complete-loop Instant interval; per-forward host logits downloads complete work; outer wall interval additionally CUDA synchronized",
        "rounds": run.rounds, "all_accepted_rounds": run.all_accepted_rounds,
        "first_round_accepted": run.first_round_accepted,
        "drafted": run.drafted, "accepted": run.accepted,
        "accepted_drafted_ratio": accounting::ratio(run.accepted, f64::from(u32::try_from(run.drafted)?)),
        "verify_rows": run.verify_rows, "replay_rows": run.replay_rows,
        "compact_recovery": run.compact_recovery
    });
    Ok(Observed {
        tokens: run.tokens,
        target: run.target_session,
        draft: Some(run.draft_session),
        costs,
        decode_seconds: run.decode_seconds,
        requested_output_tokens: request.output_tokens,
    })
}

pub(super) fn describe(models: &LoadedModels<'_, '_, '_>, run: &Observed<'_>) -> Result<Value> {
    models.context.synchronize()?;
    let draft = match &run.draft {
        Some(session) => Some(evidence::snapshot(&session.state, &session.cursor)?),
        None => None,
    };
    Ok(json!({
        "generated_token_ids": run.tokens,
        "counts": accounting::counts(&run.tokens, run.decode_seconds)?,
        "costs": run.costs,
        "target": evidence::snapshot(&run.target.state, &run.target.cursor)?,
        "draft": draft,
        "complete_fixed_output": run.tokens.len() == run.requested_output_tokens
    }))
}

pub(super) fn continuation<'ctx>(
    models: &LoadedModels<'_, '_, 'ctx>,
    run: &Observed<'_>,
) -> Result<(Session<'ctx>, crate::kernels::cuda::resident_model::Output)> {
    let pending = *run
        .tokens
        .last()
        .ok_or_else(|| anyhow::anyhow!("run omitted pending final output"))?;
    let mut branch = run.target.fork(models.context)?;
    let output =
        models
            .target
            .forward_decode(models.context, models.module, &[pending], &mut branch)?;
    models.context.synchronize()?;
    ensure!(
        output.past == branch.cursor.past(),
        "continuation output/session cursor differs"
    );
    Ok((branch, output))
}
