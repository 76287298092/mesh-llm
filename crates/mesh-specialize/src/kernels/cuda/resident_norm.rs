//! Reuse qualified BF16 normalization kernels on buffers in a resident context.

use super::{
    driver::{Buffer, Context, Module},
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

const MAX_ELEMENTS: usize = 67_108_864;

pub(super) struct Norm<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight: u64,
    width: usize,
    epsilon: f32,
}

pub(super) struct Normalized<'ctx> {
    pub(super) residual: Buffer<'ctx>,
    pub(super) normalized: Buffer<'ctx>,
}

impl<'w, 'ctx> Norm<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        name: &str,
        width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        ensure!(
            (1..=32768).contains(&width),
            "resident norm width is out of range"
        );
        ensure!(
            epsilon.is_finite() && epsilon > 0.0,
            "resident norm epsilon must be positive and finite"
        );
        let width_u64 = u64::try_from(width).context("resident norm width does not fit u64")?;
        let bytes = width
            .checked_mul(2)
            .context("resident norm byte extent overflows usize")?;
        let bytes_u64 = u64::try_from(bytes).context("resident norm bytes do not fit u64")?;
        let weight = owner.tensor(name, DType::Bf16, &[width_u64], bytes_u64)?;
        Ok(Self {
            owner,
            weight,
            width,
            epsilon,
        })
    }

    /// Normalize `[rows, width]` BF16 input using the existing fused entry kernel.
    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        ensure!(
            self.owner.belongs_to(context),
            "resident norm belongs to another context"
        );
        ensure!(
            module.belongs_to(context),
            "resident norm module belongs to another context"
        );
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
    pub(super) fn add<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        residual: &Buffer<'_>,
        branch: &Buffer<'_>,
        rows: usize,
    ) -> Result<Normalized<'a>> {
        ensure!(
            self.owner.belongs_to(context),
            "resident norm belongs to another context"
        );
        ensure!(
            module.belongs_to(context),
            "resident norm module belongs to another context"
        );
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

pub(super) fn residual_add<'a>(
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

fn matrix_extents(rows: usize, width: usize) -> Result<(usize, usize, usize)> {
    ensure!(
        (1..=2048).contains(&rows),
        "resident norm row count is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "resident norm width is out of range"
    );
    let count = rows
        .checked_mul(width)
        .context("resident norm element count overflows usize")?;
    ensure!(
        count <= MAX_ELEMENTS,
        "resident norm element count is too large"
    );
    let bf16_bytes = count
        .checked_mul(2)
        .context("resident norm BF16 extent overflows usize")?;
    let fp32_bytes = count
        .checked_mul(4)
        .context("resident norm FP32 extent overflows usize")?;
    Ok((count, bf16_bytes, fp32_bytes))
}

fn validate_matrix_buffer(
    input_bytes: usize,
    rows: usize,
    width: usize,
) -> Result<(usize, usize, usize)> {
    let extents = matrix_extents(rows, width)?;
    ensure!(
        input_bytes == extents.1,
        "resident norm input extent mismatch"
    );
    Ok(extents)
}

fn validate_matrix_pair(
    residual_bytes: usize,
    branch_bytes: usize,
    rows: usize,
    width: usize,
) -> Result<(usize, usize, usize)> {
    let extents = matrix_extents(rows, width)?;
    ensure!(
        residual_bytes == extents.1 && branch_bytes == extents.1,
        "resident norm input extent mismatch"
    );
    Ok(extents)
}

fn validate_add_extents(left_bytes: usize, right_bytes: usize) -> Result<(usize, usize)> {
    ensure!(
        left_bytes == right_bytes && left_bytes.is_multiple_of(2),
        "residual add inputs must have the same even byte extent"
    );
    let count = left_bytes / 2;
    ensure!(
        (1..=MAX_ELEMENTS).contains(&count),
        "residual add element count is out of range"
    );
    Ok((count, left_bytes))
}

fn row_ids(rows: usize) -> Result<Vec<u8>> {
    ensure!(
        (1..=2048).contains(&rows),
        "resident norm row count is out of range"
    );
    let byte_count = rows
        .checked_mul(4)
        .context("resident row ID byte extent overflows usize")?;
    let mut bytes = Vec::with_capacity(byte_count);
    for row in 0..rows {
        bytes.extend_from_slice(&u32::try_from(row)?.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_ELEMENTS, matrix_extents, validate_add_extents, validate_matrix_buffer,
        validate_matrix_pair,
    };

    #[test]
    fn validates_norm_input_at_minimum_and_maximum_extents() {
        assert_eq!(matrix_extents(1, 1).unwrap(), (1, 2, 4));
        assert_eq!(
            validate_matrix_buffer(134_217_728, 2048, 32768).unwrap(),
            (MAX_ELEMENTS, 134_217_728, 268_435_456)
        );
    }

    #[test]
    fn rejects_invalid_norm_shapes_and_input_byte_lengths() {
        assert!(matrix_extents(0, 1).is_err());
        assert!(matrix_extents(2049, 1).is_err());
        assert!(matrix_extents(1, 0).is_err());
        assert!(matrix_extents(1, 32769).is_err());
        assert!(validate_matrix_buffer(4, 1, 1).is_err());
        assert!(validate_matrix_pair(2, 4, 1, 1).is_err());
    }

    #[test]
    fn validates_residual_add_even_extents_and_count_bounds() {
        assert_eq!(validate_add_extents(2, 2).unwrap(), (1, 2));
        assert_eq!(
            validate_add_extents(134_217_728, 134_217_728).unwrap(),
            (MAX_ELEMENTS, 134_217_728)
        );
        assert!(validate_add_extents(0, 0).is_err());
        assert!(validate_add_extents(2, 4).is_err());
        assert!(validate_add_extents(3, 3).is_err());
        assert!(validate_add_extents(134_217_730, 134_217_730).is_err());
    }
}
