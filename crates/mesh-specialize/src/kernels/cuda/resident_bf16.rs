//! Device-resident BF16 linear projection over verified model weights.

use super::{
    driver::{Buffer, Context, Module},
    resident_fp8::Output,
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Context as _, Result, anyhow, ensure};
use std::ffi::c_void;

const MAX_ROWS: usize = 2048;
const MAX_WIDTH: usize = 32_768;

pub(super) struct Projection<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight: u64,
    width: usize,
    channels: usize,
}

impl<'w, 'ctx> Projection<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        name: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        validate_dimensions(width, channels)?;
        let elements = checked_product(channels, width, "BF16 projection weight")?;
        let weight_bytes = checked_product(elements, 2, "BF16 projection weight bytes")?;
        let shape = [u64::try_from(channels)?, u64::try_from(width)?];
        let weight = owner.tensor(name, DType::Bf16, &shape, u64::try_from(weight_bytes)?)?;
        Ok(Self {
            owner,
            weight,
            width,
            channels,
        })
    }

    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Output<'a>> {
        ensure!(
            self.owner.belongs_to(context)
                && input.belongs_to(context)
                && module.belongs_to(context),
            "BF16 projection weights, input, and PTX module must belong to the same CUDA context"
        );
        let extents = run_extents(rows, self.width, self.channels)?;
        validate_input_length(input.len(), extents.input_bytes)?;

        let linear = module.function("bf16_linear_decode")?;
        let output = Buffer::new(context, extents.output_bytes)?;
        let unrounded = Buffer::new(context, extents.unrounded_bytes)?;
        let mut pointers = [
            input.pointer(),
            self.weight,
            output.pointer(),
            unrounded.pointer(),
        ];
        let mut dimensions = [
            u32::try_from(rows).context("BF16 projection rows do not fit u32")?,
            u32::try_from(self.channels).context("BF16 projection channels do not fit u32")?,
            u32::try_from(self.width).context("BF16 projection width does not fit u32")?,
        ];
        let mut arguments: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        arguments.extend(
            dimensions
                .iter_mut()
                .map(|dimension| (dimension as *mut u32).cast()),
        );
        let grid = [
            u32::try_from(self.channels.div_ceil(4))?,
            u32::try_from(rows)?,
            1,
        ];
        // SAFETY: Extent checks cover row-major BF16 input/weight and disjoint output buffers;
        // the ordered pointers and dimensions match bf16_linear_decode's four-pointer/seven-argument ABI.
        if let Err(error) = unsafe { linear.launch(grid, [128, 1, 1], 0, &mut arguments) } {
            return Err(synchronize_after_failed_launch(context, error));
        }
        context.synchronize()?;
        Ok(Output {
            values: output,
            unrounded,
        })
    }
}

fn validate_dimensions(width: usize, channels: usize) -> Result<()> {
    ensure!(
        (1..=MAX_WIDTH).contains(&width),
        "BF16 projection width must be in 1..={MAX_WIDTH}"
    );
    ensure!(
        (1..=MAX_WIDTH).contains(&channels),
        "BF16 projection channels must be in 1..={MAX_WIDTH}"
    );
    Ok(())
}

struct RunExtents {
    input_bytes: usize,
    output_bytes: usize,
    unrounded_bytes: usize,
}

fn run_extents(rows: usize, width: usize, channels: usize) -> Result<RunExtents> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "BF16 projection rows must be in 1..={MAX_ROWS}"
    );
    validate_dimensions(width, channels)?;
    let input_elements = checked_product(rows, width, "BF16 projection input")?;
    let output_elements = checked_product(rows, channels, "BF16 projection output")?;
    Ok(RunExtents {
        input_bytes: checked_product(input_elements, 2, "BF16 projection input bytes")?,
        output_bytes: checked_product(output_elements, 2, "BF16 projection output bytes")?,
        unrounded_bytes: checked_product(output_elements, 4, "FP32 projection output bytes")?,
    })
}

fn validate_input_length(actual: usize, expected: usize) -> Result<()> {
    ensure!(
        actual == expected,
        "BF16 projection input has {actual} bytes, expected {expected}"
    );
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
}

fn synchronize_after_failed_launch(context: &Context, error: anyhow::Error) -> anyhow::Error {
    match context.synchronize() {
        Ok(()) => error.context("BF16 projection launch failed"),
        Err(sync_error) => error.context(format!(
            "BF16 projection launch failed; CUDA synchronization also failed: {sync_error:#}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_ROWS, MAX_WIDTH, checked_product, run_extents, validate_dimensions,
        validate_input_length,
    };

    #[test]
    fn validates_dimension_and_exact_run_extents() {
        validate_dimensions(1, 1).unwrap();
        validate_dimensions(MAX_WIDTH, MAX_WIDTH).unwrap();
        assert!(validate_dimensions(0, 1).is_err());
        assert!(validate_dimensions(1, MAX_WIDTH + 1).is_err());

        let small = run_extents(1, 1, 1).unwrap();
        assert_eq!(small.input_bytes, 2);
        assert_eq!(small.output_bytes, 2);
        assert_eq!(small.unrounded_bytes, 4);

        let maximum = run_extents(MAX_ROWS, MAX_WIDTH, MAX_WIDTH).unwrap();
        assert_eq!(maximum.input_bytes, 134_217_728);
        assert_eq!(maximum.output_bytes, 134_217_728);
        assert_eq!(maximum.unrounded_bytes, 268_435_456);
    }

    #[test]
    fn rejects_invalid_rows_lengths_and_overflowing_extents() {
        assert!(run_extents(0, 1, 1).is_err());
        assert!(run_extents(MAX_ROWS + 1, 1, 1).is_err());
        assert!(validate_input_length(4, 2).is_err());
        assert!(validate_input_length(2, 2).is_ok());
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }
}
