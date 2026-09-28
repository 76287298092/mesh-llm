//! Resolved stage handles and allocation-free default-stream launch ABI.
use super::driver::{Function, Module};
use crate::kernels::attention_staged_plan::{BLOCKS, KERNELS, Plan, SCALE, validate_addresses};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

pub(super) struct Kernels<'m, 'ctx> {
    pub(super) scores: Function<'m, 'ctx>,
    pub(super) coefficients: Function<'m, 'ctx>,
    pub(super) values: Function<'m, 'ctx>,
}
impl<'m, 'ctx> Kernels<'m, 'ctx> {
    pub(super) fn new(module: &'m Module<'ctx>) -> Result<Self> {
        Ok(Self {
            scores: module.function(KERNELS[0])?,
            coefficients: module.function(KERNELS[1])?,
            values: module.function(KERNELS[2])?,
        })
    }
    /// # Safety
    /// Q/K/V/BF16-out/FP32-out/workspace pointers name same-context allocations with
    /// the Plan's full extents, finite initialized inputs, and no aliases. All must
    /// remain live through caller synchronization even if an intermediate launch fails.
    /// KV append precedes this call. All three launches use the SAME default stream.
    pub(super) unsafe fn launch(&self, pointers: [u64; 6], plan: Plan) -> Result<()> {
        validate_addresses(plan, pointers)?;
        let [q, k, v, output, raw, workspace] = pointers;
        let length = plan.length as u64;
        let capacity = plan.capacity as u64;
        // SAFETY: Checked ranges and caller's initialization/lifetime contract apply
        // to each stage; default-stream order is the only inter-stage publication.
        unsafe {
            enqueue(
                &self.scores,
                plan.grids[0],
                BLOCKS[0],
                &mut [
                    q,
                    k,
                    workspace,
                    length,
                    capacity,
                    u64::from(SCALE.to_bits()),
                ],
            )?;
            enqueue(
                &self.coefficients,
                plan.grids[1],
                BLOCKS[1],
                &mut [workspace, length, capacity],
            )?;
            enqueue(
                &self.values,
                plan.grids[2],
                BLOCKS[2],
                &mut [v, workspace, output, raw, length, capacity],
            )
        }
    }
}
unsafe fn enqueue(
    function: &Function<'_, '_>,
    grid: [u32; 3],
    block: [u32; 3],
    storage: &mut [u64],
) -> Result<()> {
    ensure!(
        cfg!(target_endian = "little") && storage.len() <= 6,
        "invalid staged launch storage"
    );
    let mut arguments = [ptr::null_mut::<c_void>(); 6];
    for (argument, value) in arguments.iter_mut().zip(storage.iter_mut()) {
        *argument = ptr::from_mut(value).cast();
    }
    // SAFETY: u32/f32 arguments occupy the low4 bytes of8-byte slots; pointer args
    // use all8. The supplied symbol's documented ABI and live ranges are upheld.
    unsafe { function.launch(grid, block, 0, &mut arguments[..storage.len()]) }
}
