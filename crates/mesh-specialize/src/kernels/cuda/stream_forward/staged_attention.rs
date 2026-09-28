//! One reusable capacity-strided workspace across all M=1 attention layers on this stream.
use super::{
    layers::Step,
    ops::{Args, Enqueue},
    program::Shapes,
};
use crate::kernels::{
    attention_staged_plan::{BLOCKS, KERNELS, Plan, SCALE, validate_addresses},
    cuda::{
        attention_staged_launch::Kernels,
        driver::{Buffer, Context, Module},
    },
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) struct StagedAttention<'m, 'ctx> {
    kernels: Kernels<'m, 'ctx>,
    workspace: Buffer<'ctx>,
    capacity: usize,
}
impl<'m, 'ctx> StagedAttention<'m, 'ctx> {
    pub(super) fn new(
        context: &'ctx Context,
        module: &'m Module<'ctx>,
        shapes: &Shapes,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            module.belongs_to(context),
            "stream staged attention module/context mismatch"
        );
        let plan = Plan::new([
            1,
            shapes.query_heads,
            shapes.kv_heads,
            shapes.attention_width,
            0,
            capacity,
        ])?;
        let kernels = Kernels::new(module)?;
        let workspace = Buffer::new(context, plan.workspace_bytes)?;
        Ok(Self {
            kernels,
            workspace,
            capacity,
        })
    }
    /// Called before the first forward enqueue so rejected extents cannot mutate KV.
    pub(super) fn plan(&self, step: &Step) -> Result<Plan> {
        ensure!(
            step.capacity == self.capacity,
            "staged session capacity differs from workspace"
        );
        let plan = Plan::new([step.rows, 24, 4, 256, step.past, step.capacity])?;
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
        e.launch(
            &self.kernels.scores,
            plan.grids[0],
            BLOCKS[0],
            Args::new()
                .ptrs(&[q, k, workspace])
                .u32(length)
                .u32(capacity)
                .f32(SCALE),
        )?;
        e.launch(
            &self.kernels.coefficients,
            plan.grids[1],
            BLOCKS[1],
            Args::new().ptr(workspace).u32(length).u32(capacity),
        )?;
        e.launch(
            &self.kernels.values,
            plan.grids[2],
            BLOCKS[2],
            Args::new()
                .ptrs(&[v, workspace, output, raw])
                .u32(length)
                .u32(capacity),
        )
    }
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }
    pub(super) fn report(&self) -> Value {
        json!({"workspace_bytes":self.workspace.len(),"capacity":self.capacity,"kernels":KERNELS,
            "workspace_layout":"f64 score/alpha/beta planes, each24*capacity, then24 normalizers",
            "stage_order":["scores","serial coefficients","serial-per-channel values"],
            "prefix":"length=past+1; tails neither read nor written; entire initialized prefix overwritten every layer",
            "prefill_kernel":"causal_attention_bf16","prefill_min_rows":2,
            "position_abi":"by-value length/capacity; graph rejected","status":"unqualified model opt-in"})
    }
}
