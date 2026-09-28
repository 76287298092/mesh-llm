//! Device-resident BF16 linear projection over verified model weights.

use super::{
    driver::{Buffer, Context, Module},
    resident_fp8::Output,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::schema::DType,
    kernels::ab_schedule::{self, Schedule},
};
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

    /// None retains the caller's two original launches and observer ordering.
    pub(super) fn try_run_pair<'a>(
        &self,
        other: &Self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        step: (Schedule, usize),
    ) -> Result<Option<(Output<'a>, Output<'a>)>> {
        let (schedule, rows) = step;
        if schedule.select(rows, self.channels, self.width) != Schedule::PairedFp64 {
            return Ok(None);
        }
        ensure!(
            std::ptr::eq(self.owner, other.owner),
            "paired A/B weights must have the same resident owner"
        );
        ensure!(
            self.width == other.width && self.channels == other.channels,
            "paired A/B N/K mismatch"
        );
        ensure!(
            self.owner.belongs_to(context)
                && input.belongs_to(context)
                && module.belongs_to(context),
            "paired A/B weights, input and module must belong to the same CUDA context"
        );
        let extents = run_extents(1, self.width, self.channels)?;
        validate_input_length(input.len(), extents.input_bytes)?;
        let paired = module.function(ab_schedule::PAIRED_KERNEL)?;
        let a = Output {
            values: Buffer::new(context, extents.output_bytes)?,
            unrounded: Buffer::new(context, extents.unrounded_bytes)?,
        };
        let b = Output {
            values: Buffer::new(context, extents.output_bytes)?,
            unrounded: Buffer::new(context, extents.unrounded_bytes)?,
        };
        let mut pointers = [
            input.pointer(),
            self.weight,
            other.weight,
            a.values.pointer(),
            b.values.pointer(),
            a.unrounded.pointer(),
            b.unrounded.pointer(),
        ];
        validate_launch_access(pointers, self.channels, self.width)?;
        let mut dimensions = [u32::try_from(self.channels)?, u32::try_from(self.width)?];
        let mut arguments: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        arguments.extend(
            dimensions
                .iter_mut()
                .map(|dimension| (dimension as *mut u32).cast()),
        );
        // SAFETY: Exact ABI order and all seven disjoint spans were checked. Owner/input
        // borrows and local outputs stay live through unconditional completion below.
        let launch =
            unsafe { paired.launch([dimensions[0], 2, 1], [128, 1, 1], 0, &mut arguments) };
        if let Err(error) = launch {
            return Err(synchronize_after_failed_launch(context, error));
        }
        context.synchronize()?;
        Ok(Some((a, b)))
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

/// Pure address/extent check for the paired nine-argument ABI. Allocation owners
/// must independently prove that each checked span exists and remains live.
pub(super) fn validate_launch_access(
    pointers: [u64; 7],
    channels: usize,
    width: usize,
) -> Result<()> {
    ensure!(
        Schedule::PairedFp64.select(1, channels, width) == Schedule::PairedFp64,
        "unsupported paired A/B shape"
    );
    let extents = run_extents(1, width, channels)?;
    let weight_bytes = checked_product(channels, extents.input_bytes, "paired A/B weights")?;
    let lengths = [
        extents.input_bytes,
        weight_bytes,
        weight_bytes,
        extents.output_bytes,
        extents.output_bytes,
        extents.unrounded_bytes,
        extents.unrounded_bytes,
    ];
    let alignments = [16, 16, 16, 2, 2, 4, 4];
    let mut ends = [0_u64; 7];
    for i in 0..7 {
        ensure!(
            pointers[i] != 0 && pointers[i].is_multiple_of(alignments[i]),
            "paired A/B argument {i} has invalid pointer alignment"
        );
        ends[i] = pointers[i]
            .checked_add(u64::try_from(lengths[i])?)
            .context("paired A/B address extent overflows u64")?;
        for j in 0..i {
            ensure!(
                ends[i] <= pointers[j] || ends[j] <= pointers[i],
                "paired A/B argument spans {j} and {i} overlap"
            );
        }
    }
    Ok(())
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
        validate_input_length, validate_launch_access,
    };

    #[test]
    fn paired_access_checks_every_span_alignment_and_overflow() {
        let addresses = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000];
        validate_launch_access(addresses, 3, 24).unwrap();
        for i in 0..7 {
            let mut unaligned = addresses;
            unaligned[i] += 1;
            assert!(validate_launch_access(unaligned, 3, 24).is_err());
            let mut null = addresses;
            null[i] = 0;
            assert!(validate_launch_access(null, 3, 24).is_err());
            for j in 0..i {
                let mut alias = addresses;
                alias[i] = alias[j];
                assert!(validate_launch_access(alias, 3, 24).is_err());
            }
        }
        let mut overflow = addresses;
        overflow[0] = u64::MAX - 15;
        assert!(validate_launch_access(overflow, 3, 24).is_err());
        assert!(validate_launch_access(addresses, 257, 24).is_err());
        assert!(validate_launch_access(addresses, 3, 23).is_err());
    }

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
