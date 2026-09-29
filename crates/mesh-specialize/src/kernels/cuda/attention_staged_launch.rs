//! Resolved stage handles and allocation-free default-stream launch ABI.
use super::driver::{Function, Module};
use crate::kernels::attention_staged_plan::{
    BLOCKS, CoefficientSchedule, KERNELS, PREFIX_BLOCKS, PREFIX_KERNELS, PREFIX_SCHEDULE_KERNELS,
    Plan, SCALE, validate_addresses,
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

pub(super) struct Kernels<'m, 'ctx> {
    pub(super) scores: Function<'m, 'ctx>,
    pub(super) coefficients: Function<'m, 'ctx>,
    pub(super) values: Function<'m, 'ctx>,
}
pub(super) struct PrefixKernels<'m, 'ctx> {
    pub(super) scores: Function<'m, 'ctx>,
    pub(super) coefficients: Function<'m, 'ctx>,
    pub(super) prefix_maxima: Function<'m, 'ctx>,
    pub(super) normalizer: Function<'m, 'ctx>,
    pub(super) values: Function<'m, 'ctx>,
}
pub(super) enum ScheduleKernels<'m, 'ctx> {
    Serial(Kernels<'m, 'ctx>),
    PrefixParallel(PrefixKernels<'m, 'ctx>),
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
impl<'m, 'ctx> ScheduleKernels<'m, 'ctx> {
    pub(super) fn new(module: &'m Module<'ctx>, schedule: CoefficientSchedule) -> Result<Self> {
        match schedule {
            CoefficientSchedule::SerialV1 => Ok(Self::Serial(Kernels::new(module)?)),
            CoefficientSchedule::PrefixParallelV2 => Ok(Self::PrefixParallel(PrefixKernels {
                scores: module.function(KERNELS[0])?,
                prefix_maxima: module.function(PREFIX_KERNELS[0])?,
                coefficients: module.function(PREFIX_KERNELS[1])?,
                normalizer: module.function(PREFIX_KERNELS[2])?,
                values: module.function(KERNELS[2])?,
            })),
        }
    }

    pub(super) unsafe fn launch(&self, pointers: [u64; 6], plan: Plan) -> Result<()> {
        match self {
            Self::Serial(kernels) => unsafe { kernels.launch(pointers, plan) },
            Self::PrefixParallel(kernels) => unsafe { kernels.launch(pointers, plan) },
        }
    }

    pub(super) fn resources(&self) -> Result<Vec<super::driver::FunctionResources>> {
        match self {
            Self::Serial(kernels) => Ok(vec![
                kernels.scores.resources()?,
                kernels.coefficients.resources()?,
                kernels.values.resources()?,
            ]),
            Self::PrefixParallel(kernels) => Ok(vec![
                kernels.scores.resources()?,
                kernels.prefix_maxima.resources()?,
                kernels.coefficients.resources()?,
                kernels.normalizer.resources()?,
                kernels.values.resources()?,
            ]),
        }
    }

    pub(super) fn kernel_names(&self) -> &'static [&'static str] {
        match self {
            Self::Serial(_) => &KERNELS,
            Self::PrefixParallel(_) => &PREFIX_SCHEDULE_KERNELS,
        }
    }

    pub(super) fn serial(&self) -> Option<&Kernels<'m, 'ctx>> {
        match self {
            Self::Serial(kernels) => Some(kernels),
            Self::PrefixParallel(_) => None,
        }
    }

    pub(super) fn prefix(&self) -> Option<&PrefixKernels<'m, 'ctx>> {
        match self {
            Self::Serial(_) => None,
            Self::PrefixParallel(kernels) => Some(kernels),
        }
    }
}

impl PrefixKernels<'_, '_> {
    unsafe fn launch(&self, pointers: [u64; 6], plan: Plan) -> Result<()> {
        validate_addresses(plan, pointers)?;
        let [q, k, v, output, raw, workspace] = pointers;
        let [length, capacity] = plan.dimensions();
        let grids = plan.prefix_grids()?;
        let (maxima, coefficients, normalizer, values) = (
            &self.prefix_maxima,
            &self.coefficients,
            &self.normalizer,
            &self.values,
        );
        unsafe {
            enqueue(
                &self.scores,
                plan.grids[0],
                BLOCKS[0],
                &mut [
                    q,
                    k,
                    workspace,
                    u64::from(length),
                    u64::from(capacity),
                    u64::from(SCALE.to_bits()),
                ],
            )?;
            enqueue(
                maxima,
                grids[1],
                PREFIX_BLOCKS[0],
                &mut [workspace, u64::from(length), u64::from(capacity)],
            )?;
            enqueue(
                coefficients,
                grids[2],
                PREFIX_BLOCKS[1],
                &mut [workspace, u64::from(length), u64::from(capacity)],
            )?;
            enqueue(
                normalizer,
                grids[3],
                PREFIX_BLOCKS[2],
                &mut [workspace, u64::from(length), u64::from(capacity)],
            )?;
            enqueue(
                values,
                grids[4],
                PREFIX_BLOCKS[3],
                &mut [
                    v,
                    workspace,
                    output,
                    raw,
                    u64::from(length),
                    u64::from(capacity),
                ],
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
