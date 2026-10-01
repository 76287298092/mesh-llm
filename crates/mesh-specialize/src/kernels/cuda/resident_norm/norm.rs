//! Reuse qualified BF16 normalization kernels on buffers in a resident context.

use super::extents::{row_ids, validate_add_extents, validate_matrix_buffer, validate_matrix_pair};
use crate::artifact::schema::DType;
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    resident_native_mtp::NativeMtpNormBinding,
    resident_weights::ResidentWeights,
};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(in crate::kernels::cuda) struct Norm<'w, 'ctx> {
    owner: NormOwner<'w, 'ctx>,
    weight: u64,
    width: usize,
    epsilon: f32,
}

enum NormOwner<'w, 'ctx> {
    Weights(&'w ResidentWeights<'ctx>),
    NativeMtp(NativeMtpNormBinding<'w, 'ctx>),
}

pub(in crate::kernels::cuda) struct Normalized<'ctx> {
    pub(in crate::kernels::cuda) residual: Buffer<'ctx>,
    pub(in crate::kernels::cuda) normalized: Buffer<'ctx>,
}

impl<'w, 'ctx> Norm<'w, 'ctx> {
    pub(in crate::kernels::cuda) fn new(
        owner: &'w ResidentWeights<'ctx>,
        name: &str,
        width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        validate_parameters(width, epsilon)?;
        let width_u64 = u64::try_from(width).context("resident norm width does not fit u64")?;
        let bytes = width
            .checked_mul(2)
            .context("resident norm byte extent overflows usize")?;
        let bytes_u64 = u64::try_from(bytes).context("resident norm bytes do not fit u64")?;
        let weight = owner.tensor(name, DType::Bf16, &[width_u64], bytes_u64)?;
        Ok(Self {
            owner: NormOwner::Weights(owner),
            weight,
            width,
            epsilon,
        })
    }

    /// Build a norm whose native-MTP arena borrow lasts for the returned norm.
    pub(in crate::kernels::cuda) fn from_native_mtp(
        binding: NativeMtpNormBinding<'w, 'ctx>,
        width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        validate_parameters(width, epsilon)?;
        ensure!(
            binding.elements() == width,
            "native MTP norm width differs from its saved view"
        );
        let weight = binding.pointer()?;
        Ok(Self {
            owner: NormOwner::NativeMtp(binding),
            weight,
            width,
            epsilon,
        })
    }

    pub(in crate::kernels::cuda) fn weight_pointer(&self) -> u64 {
        self.weight
    }

    pub(in crate::kernels::cuda) fn belongs_to(&self, context: &Context) -> bool {
        match &self.owner {
            NormOwner::Weights(owner) => owner.belongs_to(context),
            NormOwner::NativeMtp(owner) => owner.belongs_to(context),
        }
    }

    /// Normalize `[rows, width]` BF16 input using the existing fused entry kernel.
    pub(in crate::kernels::cuda) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        validate_owner_context(self, context, module)?;
        ensure!(
            input.belongs_to(context),
            "resident norm input belongs to another context"
        );
        let (_count, bf16_bytes, fp32_bytes) =
            validate_matrix_buffer(input.len(), rows, self.width)?;
        let row_ids = row_ids(rows)?;
        let ids = Buffer::new(context, row_ids.len())?;
        ids.upload(&row_ids)?;
        let residual = Buffer::new(context, bf16_bytes)?;
        let normalized = Buffer::new(context, bf16_bytes)?;
        let unrounded = Buffer::new(context, fp32_bytes)?;

        let mut pointers = [
            input.pointer(),
            ids.pointer(),
            self.weight,
            residual.pointer(),
            normalized.pointer(),
            unrounded.pointer(),
        ];
        let mut width =
            u32::try_from(self.width).context("resident norm width does not fit u32")?;
        let mut epsilon = self.epsilon;
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        args.push((&mut width as *mut u32).cast());
        args.push((&mut epsilon as *mut f32).cast());
        // SAFETY: The arguments follow embedding_norm_bf16's six-pointer ABI.
        // All rows have exact BF16/FP32 extents, row IDs select each input row,
        // and the owner, module, and buffers were checked against this context.
        unsafe {
            module.function("embedding_norm_bf16")?.launch(
                [u32::try_from(rows)?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        context.synchronize()?;
        drop(ids);
        drop(residual);
        drop(unrounded);
        Ok(normalized)
    }

    /// Add BF16 residual inputs, round, normalize, and return both BF16 outputs.
    pub(in crate::kernels::cuda) fn add<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        residual: &Buffer<'_>,
        branch: &Buffer<'_>,
        rows: usize,
    ) -> Result<Normalized<'a>> {
        validate_owner_context(self, context, module)?;
        ensure!(
            residual.belongs_to(context) && branch.belongs_to(context),
            "resident norm inputs belong to another context"
        );
        let (_count, bf16_bytes, fp32_bytes) =
            validate_matrix_pair(residual.len(), branch.len(), rows, self.width)?;
        let sum = Buffer::new(context, bf16_bytes)?;
        let normalized = Buffer::new(context, bf16_bytes)?;
        let unrounded = Buffer::new(context, fp32_bytes)?;

        let mut pointers = [
            residual.pointer(),
            branch.pointer(),
            self.weight,
            sum.pointer(),
            normalized.pointer(),
            unrounded.pointer(),
        ];
        let mut width =
            u32::try_from(self.width).context("resident norm width does not fit u32")?;
        let mut epsilon = self.epsilon;
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        args.push((&mut width as *mut u32).cast());
        args.push((&mut epsilon as *mut f32).cast());
        // SAFETY: The arguments follow residual_norm_bf16's six-pointer ABI.
        // Exact extents and shared context ownership were checked before launch;
        // fresh output allocations remain live until synchronization completes.
        unsafe {
            module.function("residual_norm_bf16")?.launch(
                [u32::try_from(rows)?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        context.synchronize()?;
        drop(unrounded);
        Ok(Normalized {
            residual: sum,
            normalized,
        })
    }
}

fn validate_owner_context(
    norm: &Norm<'_, '_>,
    context: &Context,
    module: &Module<'_>,
) -> Result<()> {
    ensure!(
        norm.belongs_to(context),
        "resident norm belongs to another context"
    );
    ensure!(
        module.belongs_to(context),
        "resident norm module belongs to another context"
    );
    Ok(())
}

fn validate_parameters(width: usize, epsilon: f32) -> Result<()> {
    ensure!(
        (1..=32768).contains(&width),
        "resident norm width is out of range"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "resident norm epsilon must be positive and finite"
    );
    Ok(())
}

pub(in crate::kernels::cuda) fn residual_add<'a>(
    context: &'a Context,
    module: &Module<'_>,
    left: &Buffer<'_>,
    right: &Buffer<'_>,
) -> Result<Buffer<'a>> {
    ensure!(
        module.belongs_to(context),
        "residual add module belongs to another context"
    );
    ensure!(
        left.belongs_to(context) && right.belongs_to(context),
        "residual add inputs belong to another context"
    );
    let (count, output_bytes) = validate_add_extents(left.len(), right.len())?;
    let output = Buffer::new(context, output_bytes)?;
    let mut pointers = [left.pointer(), right.pointer(), output.pointer()];
    let mut count_arg = u32::try_from(count).context("residual add count does not fit u32")?;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    args.push((&mut count_arg as *mut u32).cast());
    // SAFETY: Three exact-sized BF16 allocations match residual_add_bf16's ABI.
    // Inputs and new output belong to `context`, and padded lanes are guarded.
    unsafe {
        module.function("residual_add_bf16")?.launch(
            [count_arg.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    Ok(output)
}
