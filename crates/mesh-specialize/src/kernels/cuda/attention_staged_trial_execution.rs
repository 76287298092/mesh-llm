use super::{
    attention_staged_launch::{Kernels, ScheduleKernels},
    attention_staged_trial_support::{Buffers, GUARD},
    driver::{Context, Event, Module},
};
use crate::kernels::attention_staged_plan::{CoefficientSchedule, KERNELS, Plan};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) struct Functions<'m, 'ctx> {
    pub(super) staged: Kernels<'m, 'ctx>,
    pub(super) candidate: ScheduleKernels<'m, 'ctx>,
}

impl<'m, 'ctx> Functions<'m, 'ctx> {
    pub(super) fn new(module: &'m Module<'ctx>, schedule: CoefficientSchedule) -> Result<Self> {
        Ok(Self {
            staged: Kernels::new(module)?,
            candidate: ScheduleKernels::new(module, schedule)?,
        })
    }

    pub(super) fn report(&self) -> Result<Value> {
        Ok(json!({
            "staged_v1": {
                "kernels": KERNELS,
                "resources": [
                    self.staged.scores.resources()?,
                    self.staged.coefficients.resources()?,
                    self.staged.values.resources()?,
                ],
            },
            "scheduled": {
                "kernels": self.candidate.kernel_names(),
                "resources": self.candidate.resources()?,
            },
        }))
    }
}

pub(super) fn launch(
    functions: &Functions<'_, '_>,
    buffers: &Buffers<'_>,
    plan: Plan,
    candidate: bool,
) -> Result<()> {
    if candidate {
        // SAFETY: The buffers own disjoint guarded payloads in the same CUDA context.
        // Inputs are initialized and finite through plan.length; caller drains the
        // context while every buffer remains live.
        return unsafe {
            functions.candidate.launch(
                [
                    buffers.q.pointer(),
                    buffers.k.pointer(),
                    buffers.v.pointer(),
                    buffers.candidate.bf16.pointer() + GUARD as u64,
                    buffers.candidate.raw.pointer() + GUARD as u64,
                    buffers.workspace.pointer() + GUARD as u64,
                ],
                plan,
            )
        };
    }
    // SAFETY: The buffers own disjoint guarded payloads in the same CUDA context.
    // Inputs are initialized and finite through plan.length; caller drains the
    // context while every buffer remains live.
    unsafe {
        functions.staged.launch(
            [
                buffers.q.pointer(),
                buffers.k.pointer(),
                buffers.v.pointer(),
                buffers.legacy.bf16.pointer() + GUARD as u64,
                buffers.legacy.raw.pointer() + GUARD as u64,
                buffers.legacy_workspace.pointer() + GUARD as u64,
            ],
            Plan::new([1, 24, 4, 256, plan.length - 1, plan.capacity])?,
        )
    }
}

pub(super) fn execute(
    context: &Context,
    functions: &Functions<'_, '_>,
    buffers: &Buffers<'_>,
    plan: Plan,
    candidate: bool,
) -> Result<()> {
    let result = launch(functions, buffers, plan, candidate);
    let synchronize = context.synchronize();
    result?;
    synchronize
}

pub(super) fn timing(
    context: &Context,
    functions: &Functions<'_, '_>,
    buffers: &Buffers<'_>,
    plan: Plan,
    candidate: bool,
) -> Result<Value> {
    for _ in 0..3 {
        execute(context, functions, buffers, plan, candidate)?;
    }
    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let mut samples = Vec::new();
    for _ in 0..5 {
        start.record()?;
        let pending = launch(functions, buffers, plan, candidate)
            .and_then(|()| end.record())
            .and_then(|()| end.synchronize());
        if let Err(error) = pending {
            context.synchronize()?;
            return Err(error);
        }
        samples.push(end.elapsed_since(&start)?);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[2];
    ensure!(
        median.is_finite() && median > 0.0,
        "invalid staged event time"
    );
    Ok(
        json!({"warmups":3,"repetitions":5,"event_ms":samples,"median_event_ms":median,
        "scope":"whole scheduled operator span, not model throughput"}),
    )
}
