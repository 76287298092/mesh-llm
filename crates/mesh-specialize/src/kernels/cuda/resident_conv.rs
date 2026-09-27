//! Causal convolution over resident weights and persistent history.

use super::{
    driver::{Buffer, Context, Module},
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct Convolution<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight: u64,
    channels: usize,
}
impl<'w, 'ctx> Convolution<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        name: &str,
        channels: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=32768).contains(&channels),
            "invalid resident convolution channels"
        );
        let weight = owner.tensor(
            name,
            DType::Bf16,
            &[channels as u64, 1, 4],
            (channels * 8) as u64,
        )?;
        Ok(Self {
            owner,
            weight,
            channels,
        })
    }

    pub(super) fn run<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        name: &str,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        ensure!(
            (1..=2048).contains(&rows),
            "invalid resident convolution rows"
        );
        ensure!(
            self.owner.belongs_to(ctx)
                && module.belongs_to(ctx)
                && input.belongs_to(ctx)
                && state.belongs_to(ctx),
            "convolution CUDA context mismatch"
        );
        let count = rows * self.channels;
        ensure!(
            input.len() == count * 2,
            "resident convolution input extent mismatch"
        );
        let history = state.pointer(name, self.channels * 6)?;
        let next = Buffer::new(ctx, self.channels * 6)?;
        let output = Buffer::new(ctx, count * 2)?;
        let conv = Buffer::new(ctx, count * 4)?;
        let silu = Buffer::new(ctx, count * 4)?;
        let mut pointers = [
            input.pointer(),
            self.weight,
            history,
            next.pointer(),
            output.pointer(),
            conv.pointer(),
            silu.pointer(),
        ];
        let mut dims = [u32::try_from(rows)?, u32::try_from(self.channels)?];
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        args.extend(dims.iter_mut().map(|p| (p as *mut u32).cast()));
        // SAFETY: Exact seven-pointer convolution ABI. History and next-history
        // are disjoint; every buffer extent and owning context was checked.
        unsafe {
            module.function("causal_conv4_bf16")?.launch(
                [u32::try_from(count.div_ceil(256))?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        ctx.synchronize()?;
        state.copy_from(name, &next)?;
        Ok(output)
    }
}
