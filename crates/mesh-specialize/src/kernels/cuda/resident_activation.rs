//! Launch the resident BF16 SiLU-product operation without downloading intermediates.

use super::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

/// Run the resident MLP activation kernel and return its BF16 product buffer.
///
/// The caller must provide buffers allocated in `context` and keep them live
/// through completion. This function validates exact extents and synchronizes
/// before releasing diagnostic temporaries.
pub(super) fn run<'a>(
    context: &'a Context,
    module: &Module<'_>,
    gate: &Buffer<'_>,
    up: &Buffer<'_>,
    count: usize,
) -> Result<Buffer<'a>> {
    ensure!(
        gate.belongs_to(context) && up.belongs_to(context) && module.belongs_to(context),
        "resident activation buffers/module belong to a different CUDA context"
    );
    let (bf16_bytes, fp32_bytes) = validate_lengths(gate.len(), up.len(), count)?;
    let output = Buffer::new(context, bf16_bytes)?;
    let silu = Buffer::new(context, fp32_bytes)?;
    let activated = Buffer::new(context, bf16_bytes)?;
    let unrounded = Buffer::new(context, fp32_bytes)?;

    let mut pointers = [
        gate.pointer(),
        up.pointer(),
        output.pointer(),
        silu.pointer(),
        activated.pointer(),
        unrounded.pointer(),
    ];
    let mut count_arg = u32::try_from(count).context("activation count does not fit u32")?;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    args.push((&mut count_arg as *mut u32).cast());

    // SAFETY: The pointers follow the six-pointer mlp_silu_product ABI. Each
    // allocation has the validated extent, inputs and outputs are disjoint,
    // and all buffers remain live until the exact launch has synchronized.
    unsafe {
        module.function("mlp_silu_product")?.launch(
            [count_arg.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    drop(silu);
    drop(activated);
    drop(unrounded);
    Ok(output)
}

fn validate_lengths(gate_bytes: usize, up_bytes: usize, count: usize) -> Result<(usize, usize)> {
    ensure!(
        (1..=67_108_864).contains(&count),
        "resident activation count is out of range"
    );
    let bf16_bytes = count
        .checked_mul(2)
        .context("resident activation BF16 extent overflows usize")?;
    let fp32_bytes = count
        .checked_mul(4)
        .context("resident activation FP32 extent overflows usize")?;
    ensure!(
        gate_bytes == bf16_bytes && up_bytes == bf16_bytes,
        "resident activation input buffer extent mismatch"
    );
    Ok((bf16_bytes, fp32_bytes))
}

#[cfg(test)]
mod tests {
    use super::validate_lengths;

    #[test]
    fn validates_small_and_maximum_exact_extents() {
        assert_eq!(validate_lengths(2, 2, 1).unwrap(), (2, 4));
        assert_eq!(
            validate_lengths(134_217_728, 134_217_728, 67_108_864).unwrap(),
            (134_217_728, 268_435_456)
        );
    }

    #[test]
    fn rejects_invalid_counts_and_input_extents() {
        assert!(validate_lengths(0, 0, 0).is_err());
        assert!(validate_lengths(134_217_730, 134_217_730, 67_108_865).is_err());
        assert!(validate_lengths(4, 2, 1).is_err());
        assert!(validate_lengths(2, 4, 1).is_err());
    }
}
