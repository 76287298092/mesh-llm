//! One reusable capacity-strided workspace across all M=1 attention layers on this stream.
use super::{layers::Step, ops::Enqueue, program::Shapes};
use crate::kernels::{
    attention_staged_plan::{
        BLOCKS, CoefficientSchedule, KERNELS, PREFIX_BLOCKS, PREFIX_SCHEDULE_KERNELS, Plan, SCALE,
        validate_addresses,
    },
    cuda::driver::{Buffer, Context, Module},
    cuda::attention_staged_launch::ScheduleKernels,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) struct StagedAttention<'m, 'ctx> {
    kernels: ScheduleKernels<'m, 'ctx>,
    workspace: Buffer<'ctx>,
    capacity: usize,
    schedule: CoefficientSchedule,
}
impl<'m, 'ctx> StagedAttention<'m, 'ctx> {
    pub(super) fn new(
        context: &'ctx Context,
        module: &'m Module<'ctx>,
        shapes: &Shapes,
        capacity: usize,
        schedule: CoefficientSchedule,
    ) -> Result<Self> {
        ensure!(
            module.belongs_to(context),
            "stream staged attention module/context mismatch"
        );
        let plan = Plan::new_with_schedule(
            [
                1,
                shapes.query_heads,
                shapes.kv_heads,
                shapes.attention_width,
                0,
                capacity,
            ],
            schedule,
        )?;
        let kernels = ScheduleKernels::new(module, schedule)?;
        let workspace = Buffer::new(context, plan.workspace_bytes)?;
        Ok(Self {
            kernels,
            workspace,
            capacity,
            schedule,
        })
    }
    /// Called before the first forward enqueue so rejected extents cannot mutate KV.
    pub(super) fn plan(&self, step: &Step) -> Result<Plan> {
        ensure!(
            step.capacity == self.capacity,
            "staged session capacity differs from workspace"
        );
        let plan = Plan::new_with_schedule(
            [step.rows, 24, 4, 256, step.past, step.capacity],
            self.schedule,
        )?;
        ensure!(
            plan.workspace_bytes == self.workspace.len(),
            "staged workspace extent mismatch"
        );
        Ok(plan)
    }
    /// Enqueue owns valid same-context arena/state ranges; buffers live until the
    /// enclosing forward's success/error drain. All stages use e's SAME active stream.
    pub(super) fn enqueue(
        &self,
        e: &Enqueue<'_, '_, '_, '_>,
        pointers: [u64; 5],
        step: &Step,
    ) -> Result<()> {
        ensure!(
            e.position.is_none(),
            "staged-fp64 has no qualified graph position variant"
        );
        let plan = self.plan(step)?;
        let [q, k, v, output, raw] = pointers;
        let workspace = self.workspace.pointer();
        validate_addresses(plan, [q, k, v, output, raw, workspace])?;
        let [length, capacity] = plan.dimensions();
        if let Some(kernels) = self.kernels.serial() {
            e.launch(
                &kernels.scores,
                plan.grids[0],
                BLOCKS[0],
                super::ops::Args::new()
                    .ptrs(&[q, k, workspace])
                    .u32(length)
                    .u32(capacity)
                    .f32(SCALE),
            )?;
            e.launch(
                &kernels.coefficients,
                plan.grids[1],
                BLOCKS[1],
                super::ops::Args::new()
                    .ptr(workspace)
                    .u32(length)
                    .u32(capacity),
            )?;
            return e.launch(
                &kernels.values,
                plan.grids[2],
                BLOCKS[2],
                super::ops::Args::new()
                    .ptrs(&[v, workspace, output, raw])
                    .u32(length)
                    .u32(capacity),
            );
        }
        let Some(kernels) = self.kernels.prefix() else {
            return Err(anyhow::anyhow!(
                "prefix-parallel schedule has no launch kernels"
            ));
        };
        let grids = plan.prefix_grids()?;
        e.launch(
            &kernels.scores,
            grids[0],
            BLOCKS[0],
            super::ops::Args::new()
                .ptrs(&[q, k, workspace])
                .u32(length)
                .u32(capacity)
                .f32(SCALE),
        )?;
        for (function, grid, block) in [
            (&kernels.prefix_maxima, grids[1], PREFIX_BLOCKS[0]),
            (&kernels.coefficients, grids[2], PREFIX_BLOCKS[1]),
            (&kernels.normalizer, grids[3], PREFIX_BLOCKS[2]),
        ] {
            e.launch(
                function,
                grid,
                block,
                super::ops::Args::new()
                    .ptr(workspace)
                    .u32(length)
                    .u32(capacity),
            )?;
        }
        e.launch(
            &kernels.values,
            grids[4],
            PREFIX_BLOCKS[3],
            super::ops::Args::new()
                .ptrs(&[v, workspace, output, raw])
                .u32(length)
                .u32(capacity),
        )
    }
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }
    pub(super) fn report(&self) -> Value {
        json!({"workspace_bytes":self.workspace.len(),"capacity":self.capacity,"schedule":self.schedule.name(),"kernels":if self.schedule==CoefficientSchedule::SerialV1 { &KERNELS[..] } else { &PREFIX_SCHEDULE_KERNELS[..] },
            "workspace_layout":if self.schedule==CoefficientSchedule::SerialV1 { "f64 score/alpha/beta planes, each24*capacity, then24 normalizers" } else { "f64 score/alpha/beta planes, each24*capacity, 24 normalizers, then a capacity-strided running-max plane" },
            "stage_order":if self.schedule==CoefficientSchedule::SerialV1 { &["scores","serial coefficients","serial-per-channel values"][..] } else { &["scores","running maxima","parallel exponentials","serial normalizer","serial-per-channel values"][..] },
            "prefix":"length=past+1; tails neither read nor written; entire initialized prefix overwritten every layer",
            "prefill_kernel":"causal_attention_bf16","prefill_min_rows":2,
            "position_abi":"by-value length/capacity; graph rejected","status":"unqualified model opt-in"})
    }
}
