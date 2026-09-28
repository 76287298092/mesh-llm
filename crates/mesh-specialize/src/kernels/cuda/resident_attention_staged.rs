//! Legacy staged scratch is prepared before KV mutation and retained through final drain.
use super::{
    attention_staged_launch::Kernels,
    driver::{Buffer, Context, Module},
    resident_attention_core::Shape,
};
use crate::kernels::attention_staged_plan::Plan;
use anyhow::{Result, ensure};

pub(super) struct Prepared<'m, 'module_ctx, 'workspace> {
    kernels: Kernels<'m, 'module_ctx>,
    workspace: Buffer<'workspace>,
    plan: Plan,
}
impl<'m, 'module_ctx, 'workspace> Prepared<'m, 'module_ctx, 'workspace> {
    pub(super) fn new(
        context: &'workspace Context,
        module: &'m Module<'module_ctx>,
        shape: &Shape,
    ) -> Result<Self> {
        ensure!(
            module.belongs_to(context),
            "staged attention module/context mismatch"
        );
        let plan = Plan::new([
            shape.rows,
            shape.query_heads,
            shape.kv_heads,
            shape.width,
            shape.past,
            shape.capacity,
        ])?;
        let kernels = Kernels::new(module)?;
        let workspace = Buffer::new(context, plan.workspace_bytes)?;
        Ok(Self {
            kernels,
            workspace,
            plan,
        })
    }
    /// # Safety
    /// Pointers Q/K/V/BF16-out/FP32-out have the validated resident core extents,
    /// initialization and context. All allocations, including self, remain live
    /// through the caller's drain on both success and ANY partial-stage failure.
    pub(super) unsafe fn launch(&self, pointers: [u64; 5]) -> Result<()> {
        let [q, k, v, output, raw] = pointers;
        // SAFETY: The owned scratch is disjoint; full-range checks precede stage1.
        unsafe {
            self.kernels
                .launch([q, k, v, output, raw, self.workspace.pointer()], self.plan)
        }
    }
}
