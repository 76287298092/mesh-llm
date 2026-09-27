//! Launch the resident BF16 attention sigmoid gate without downloading intermediates.

use super::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, anyhow, ensure};
use std::ffi::c_void;

const MAX_COUNT: usize = 67_108_864;

pub(super) fn run<'a>(
    context: &'a Context,
    module: &Module<'_>,
    attention: &Buffer<'_>,
    gate: &Buffer<'_>,
) -> Result<Buffer<'a>> {
    let extents = validate_extents(attention.len(), gate.len())?;
    ensure!(
        module.belongs_to(context) && attention.belongs_to(context) && gate.belongs_to(context),
        "attention gate inputs and PTX module must belong to the same CUDA context"
    );
    ensure!(
        attention.pointer() != gate.pointer(),
        "attention gate input buffers must be distinct"
    );

    let output = Buffer::new(context, extents.bf16_bytes)?;
    let sigmoid = Buffer::new(context, extents.fp32_bytes)?;
    let activated = Buffer::new(context, extents.bf16_bytes)?;
    let unrounded = Buffer::new(context, extents.fp32_bytes)?;
    launch_and_sync(
        context,
        module,
        [
            attention.pointer(),
            gate.pointer(),
            output.pointer(),
            sigmoid.pointer(),
            activated.pointer(),
            unrounded.pointer(),
        ],
        u32::try_from(extents.count).context("attention gate count does not fit u32")?,
    )?;
    drop((sigmoid, activated, unrounded));
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extents {
    count: usize,
    bf16_bytes: usize,
    fp32_bytes: usize,
}

fn validate_extents(attention_bytes: usize, gate_bytes: usize) -> Result<Extents> {
    ensure!(
        attention_bytes == gate_bytes,
        "attention gate input byte lengths differ"
    );
    ensure!(
        attention_bytes.is_multiple_of(2),
        "attention gate BF16 input extent must be even"
    );
    let count = attention_bytes / 2;
    ensure!(
        (1..=MAX_COUNT).contains(&count),
        "attention gate element count must be in 1..={MAX_COUNT}"
    );
    Ok(Extents {
        count,
        bf16_bytes: checked_product(count, 2, "attention gate BF16 output")?,
        fp32_bytes: checked_product(count, 4, "attention gate FP32 diagnostic")?,
    })
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
}

fn launch_and_sync(
    context: &Context,
    module: &Module<'_>,
    mut pointers: [u64; 6],
    mut count: u32,
) -> Result<()> {
    let function = module.function("attention_gate_bf16")?;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.push((&mut count as *mut u32).cast());
    // SAFETY: The six pointers follow the attention_gate_bf16 ABI and each allocation has the
    // exact validated extent. The 1D launch covers count elements, with its final block guarded.
    if let Err(error) =
        unsafe { function.launch([count.div_ceil(256), 1, 1], [256, 1, 1], 0, &mut arguments) }
    {
        return Err(synchronize_after_failed_launch(context, error));
    }
    context
        .synchronize()
        .context("synchronize attention_gate_bf16")
}

fn synchronize_after_failed_launch(context: &Context, error: anyhow::Error) -> anyhow::Error {
    match context.synchronize() {
        Ok(()) => error.context("attention_gate_bf16 launch failed"),
        Err(sync_error) => error.context(format!(
            "attention_gate_bf16 launch failed; CUDA synchronization also failed: {sync_error:#}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_COUNT, checked_product, validate_extents};

    #[test]
    fn validates_small_and_maximum_padded_element_extents() {
        assert_eq!(
            validate_extents(2, 2).unwrap(),
            super::Extents {
                count: 1,
                bf16_bytes: 2,
                fp32_bytes: 4,
            }
        );
        assert_eq!(
            validate_extents(MAX_COUNT * 2, MAX_COUNT * 2).unwrap(),
            super::Extents {
                count: MAX_COUNT,
                bf16_bytes: MAX_COUNT * 2,
                fp32_bytes: MAX_COUNT * 4,
            }
        );
    }

    #[test]
    fn rejects_mismatched_empty_odd_and_overflowing_extents() {
        assert!(validate_extents(2, 4).is_err());
        assert!(validate_extents(0, 0).is_err());
        assert!(validate_extents(3, 3).is_err());
        assert!(validate_extents(MAX_COUNT * 2 + 2, MAX_COUNT * 2 + 2).is_err());
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }
}
