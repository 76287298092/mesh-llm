//! Trial-only launch adapter. Production arena callers use the documented raw ABI.
use super::driver::{Buffer, Function};
use crate::kernels::attention_v2_plan::{Plan, SCALE};
use anyhow::Result;
use std::ffi::c_void;

pub(super) struct Inputs<'a, 'ctx> {
    pub q: &'a Buffer<'ctx>,
    pub k: &'a Buffer<'ctx>,
    pub v: &'a Buffer<'ctx>,
    pub workspace: &'a Buffer<'ctx>,
    pub output: &'a Buffer<'ctx>,
    pub raw: &'a Buffer<'ctx>,
}

pub(super) fn launch(
    partial: &Function<'_, '_>,
    reduce: &Function<'_, '_>,
    buffers: &Inputs<'_, '_>,
    plan: Plan,
    position: Option<&Buffer<'_>>,
) -> Result<()> {
    let mut pointers = vec![
        buffers.q.pointer(),
        buffers.k.pointer(),
        buffers.v.pointer(),
        buffers.workspace.pointer(),
    ];
    let mut dimensions = plan.dimensions().to_vec();
    if let Some(position) = position {
        pointers.push(position.pointer());
        dimensions.remove(4); // Device-position ABI omits by-value past.
    }
    let mut scale = SCALE;
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|d| (d as *mut u32).cast::<c_void>()),
    );
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Trial constructs exact admitted extents, same-context allocations and
    // keeps them live through its synchronization even when either launch fails.
    unsafe {
        partial.launch(plan.partial_grid, [192, 1, 1], 0, &mut args)?;
    }
    let mut pointers = [
        buffers.workspace.pointer(),
        buffers.output.pointer(),
        buffers.raw.pointer(),
    ];
    let mut dimensions = [plan.rows as u32, plan.split_slots as u32];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|d| (d as *mut u32).cast::<c_void>()),
    );
    // SAFETY: The default stream orders the fully initialized workspace before reduce.
    unsafe { reduce.launch(plan.reduce_grid, [256, 1, 1], 0, &mut args) }
}
