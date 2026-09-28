//! Opt-in legacy-stream split attention. Prepared before KV append; retained to final drain.
use super::{
    driver::{Buffer, Context, Function, Module},
    resident_attention_core::Shape,
};
use crate::kernels::attention_v2_plan::{Plan, SCALE, validate_geometry};
use anyhow::{Result, ensure};
use std::ffi::c_void;

/// Module and allocation owners can have distinct borrow lifetimes. Their actual
/// CUDA context identity is checked in `new`; the caller retains all source/state
/// and output allocations and drains the stream before dropping this object.
pub(super) struct Prepared<'m, 'module_ctx, 'workspace> {
    partial: Function<'m, 'module_ctx>,
    reduce: Function<'m, 'module_ctx>,
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
            "split attention module/context mismatch"
        );
        validate_geometry(shape.query_heads, shape.kv_heads, shape.width)?;
        let plan = Plan::new(shape.rows, shape.past, shape.capacity, shape.capacity)?;
        let partial = module.function(plan.partial_kernel())?;
        let reduce = module.function("attention_split_reduce_bf16")?;
        let workspace = Buffer::new(context, plan.workspace_bytes)?;
        Ok(Self {
            partial,
            reduce,
            workspace,
            plan,
        })
    }

    /// # Safety
    /// Pointers are Q, initialized K cache, initialized V cache, BF16 output, FP32
    /// output, with exact extents already validated by resident_attention_core.
    /// They belong to the workspace context, are disjoint and remain live until
    /// the caller drains the default stream, including if either launch fails.
    /// KV append is ordered before this call. The caller must retain `self` too.
    pub(super) unsafe fn launch(&self, pointers: [u64; 5]) -> Result<()> {
        let mut partial_pointers = [
            pointers[0],
            pointers[1],
            pointers[2],
            self.workspace.pointer(),
        ];
        let mut dimensions = self.plan.dimensions();
        let mut scale = SCALE;
        let mut arguments: Vec<*mut c_void> = partial_pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        arguments.extend(
            dimensions
                .iter_mut()
                .map(|d| (d as *mut u32).cast::<c_void>()),
        );
        arguments.push((&mut scale as *mut f32).cast());
        // SAFETY: Prepared plan fixes the four-pointer/seven-u32/FP32 ABI. Sources
        // and workspace satisfy the caller's launch/lifetime contract above.
        unsafe {
            self.partial
                .launch(self.plan.partial_grid, [192, 1, 1], 0, &mut arguments)?;
        }
        let mut reduce_pointers = [self.workspace.pointer(), pointers[3], pointers[4]];
        let mut dimensions = [self.plan.rows as u32, self.plan.split_slots as u32];
        let mut arguments: Vec<*mut c_void> = reduce_pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        arguments.extend(
            dimensions
                .iter_mut()
                .map(|d| (d as *mut u32).cast::<c_void>()),
        );
        // SAFETY: All slots are overwritten by the stream-ordered partial launch;
        // caller retains both outputs and this workspace through final drain.
        unsafe {
            self.reduce
                .launch(self.plan.reduce_grid, [256, 1, 1], 0, &mut arguments)
        }
    }
}
