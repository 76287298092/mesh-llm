use crate::kernels::cuda::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) fn run<'a>(
    context: &'a Context,
    module: &Module<'_>,
    entry: &str,
    left: &Buffer<'_>,
    right: &Buffer<'_>,
) -> Result<Buffer<'a>> {
    ensure!(
        module.belongs_to(context)
            && left.belongs_to(context)
            && right.belongs_to(context)
            && left.len() == right.len()
            && left.len().is_multiple_of(2),
        "native MTP activation inputs have wrong context or extents"
    );
    let count = left.len() / 2;
    ensure!(
        (1..=67_108_864).contains(&count),
        "native MTP activation count is out of range"
    );
    let output = Buffer::new(context, left.len())?;
    let mut pointers = [left.pointer(), right.pointer(), output.pointer()];
    let mut count_arg = u32::try_from(count).context("native MTP activation count exceeds u32")?;
    let mut arguments = pointers
        .iter_mut()
        .map(|pointer| std::ptr::from_mut(pointer).cast::<c_void>())
        .collect::<Vec<_>>();
    arguments.push(std::ptr::from_mut(&mut count_arg).cast::<c_void>());
    // SAFETY: The entry-specific contract takes three exact BF16 arrays and one
    // count; input/output buffers stay live through the synchronized launch.
    let launch = unsafe {
        module.function(entry)?.launch(
            [count_arg.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut arguments,
        )
    };
    let drain = context.synchronize();
    if let Err(error) = launch {
        return Err(error.context(format!(
            "native MTP activation launch failed; synchronized drain: {drain:?}"
        )));
    }
    drain.context("native MTP activation synchronized drain failed")?;
    Ok(output)
}

pub(super) fn attention_gate<'a>(
    context: &'a Context,
    module: &Module<'_>,
    attention: &Buffer<'_>,
    gate: &Buffer<'_>,
) -> Result<Buffer<'a>> {
    run(
        context,
        module,
        "native_mtp_attention_gate",
        attention,
        gate,
    )
}

pub(super) fn silu_mul<'a>(
    context: &'a Context,
    module: &Module<'_>,
    gate: &Buffer<'_>,
    up: &Buffer<'_>,
) -> Result<Buffer<'a>> {
    run(context, module, "native_mtp_silu_mul", gate, up)
}
