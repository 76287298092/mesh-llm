//! Persistent opt-in split attention state. No allocations, lookup or waits in enqueue.
use super::{
    layers::Step,
    ops::{Args, Enqueue, to_u32},
    program::Shapes,
};
use crate::kernels::{
    attention_v2_plan::{Plan, SCALE, persistent_workspace_bytes, split_count, validate_geometry},
    cuda::driver::{Buffer, Context, Function, Module},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

/// Owned by StreamForward alongside its stream and arena. Handles retain the
/// module borrow, and the workspace retains the same CUDA context borrow.
/// StreamForward drains every successful/failed forward before any owner drops.
pub(super) struct SplitAttention<'m, 'ctx> {
    decode: Function<'m, 'ctx>,
    verify: Function<'m, 'ctx>,
    reduce: Function<'m, 'ctx>,
    workspace: Buffer<'ctx>,
    max_rows: usize,
    capacity: usize,
    split_slots: usize,
}

impl<'m, 'ctx> SplitAttention<'m, 'ctx> {
    pub(super) fn new(
        context: &'ctx Context,
        module: &'m Module<'ctx>,
        shapes: &Shapes,
        max_rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            module.belongs_to(context),
            "stream split attention module/context mismatch"
        );
        validate_geometry(shapes.query_heads, shapes.kv_heads, shapes.attention_width)?;
        let bytes = persistent_workspace_bytes(max_rows, capacity)?;
        let split_slots = split_count(capacity)?;
        let decode = module.function("attention_split_decode_bf16")?;
        let verify = module.function("attention_split_bf16")?;
        let reduce = module.function("attention_split_reduce_bf16")?;
        let workspace = Buffer::new(context, bytes)?;
        Ok(Self {
            decode,
            verify,
            reduce,
            workspace,
            max_rows: max_rows.min(8),
            capacity,
            split_slots,
        })
    }

    /// Q/K/V/BF16-out/FP32-out addresses are bounded live arena/state views under
    /// Enqueue's construction contract. KV append must already be enqueued on e.
    /// ActiveStream checks each handle's context against its entered stream; new
    /// checked that the owned workspace belongs to exactly that handle context.
    pub(super) fn enqueue(
        &self,
        e: &Enqueue<'_, '_, '_, '_>,
        pointers: [u64; 5],
        step: &Step,
    ) -> Result<()> {
        let plan = self.plan(step)?;
        let mut arguments = Args::new().ptrs(&[
            pointers[0],
            pointers[1],
            pointers[2],
            self.workspace.pointer(),
        ]);
        for dimension in plan.dimensions() {
            arguments = arguments.u32(dimension);
        }
        let partial = if step.rows == 1 {
            &self.decode
        } else {
            &self.verify
        };
        e.launch(
            partial,
            plan.partial_grid,
            [192, 1, 1],
            arguments.f32(SCALE),
        )?;
        let arguments = Args::new()
            .ptrs(&[self.workspace.pointer(), pointers[3], pointers[4]])
            .u32(to_u32(step.rows)?)
            .u32(to_u32(self.split_slots)?);
        e.launch(&self.reduce, plan.reduce_grid, [256, 1, 1], arguments)
    }

    /// Also called before the first forward enqueue, so admission failures cannot append KV.
    pub(super) fn plan(&self, step: &Step) -> Result<Plan> {
        ensure!(
            step.rows <= self.max_rows,
            "split attention rows exceed persistent workspace"
        );
        ensure!(
            step.capacity == self.capacity,
            "split attention session capacity differs from configured workspace"
        );
        let plan = Plan::new(step.rows, step.past, step.capacity, self.capacity)?;
        ensure!(
            plan.workspace_bytes <= self.workspace.len(),
            "split attention workspace extent mismatch"
        );
        Ok(plan)
    }

    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }

    pub(super) fn report(&self) -> Value {
        json!({"workspace_bytes":self.workspace.len(),"max_rows":self.max_rows,
            "capacity":self.capacity,"split_slots":self.split_slots,
            "kernels":["attention_split_decode_bf16","attention_split_bf16","attention_split_reduce_bf16"],
            "prefill_kernel":"causal_attention_bf16","prefill_min_rows":9,
            "position_abi":"by-value u32 per enqueued step; graph capture not implemented",
            "status":"unqualified model opt-in"})
    }
}
