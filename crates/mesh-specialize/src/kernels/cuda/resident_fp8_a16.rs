//! Experimental one-row A16 projection over already-validated resident weights.
use super::{
    driver::{Buffer, Context, Module},
    resident_fp8::Output,
};
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) fn run<'a>(
    ctx: &'a Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    weights: [u64; 2],
    shape: [usize; 2],
) -> Result<Output<'a>> {
    let [channels, width] = shape;
    ensure!(
        (1..=262_144).contains(&channels) && (1..=32_768).contains(&width),
        "A16 resident projection dimensions exceed kernel bounds"
    );
    ensure!(
        input.len() == width * 2 && input.belongs_to(ctx) && module.belongs_to(ctx),
        "A16 resident activation extent or context mismatch"
    );
    let values = Buffer::new(ctx, channels * 2)?;
    let unrounded = Buffer::new(ctx, channels * 4)?;
    let mut pointers = [
        input.pointer(),
        weights[0],
        weights[1],
        values.pointer(),
        unrounded.pointer(),
    ];
    let mut dims = [u32::try_from(channels)?, u32::try_from(width)?];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    // SAFETY: The resident projection caller validates weight/scale extents and ownership.
    // One BF16 activation row and both disjoint outputs are checked above; every allocation
    // remains alive through synchronization. Four warps each own one output channel.
    let launch = unsafe {
        module.function("fp8_a16_decode")?.launch(
            [u32::try_from(channels.div_ceil(4))?, 1, 1],
            [128, 1, 1],
            0,
            &mut args,
        )
    };
    if let Err(error) = launch {
        let sync = ctx.synchronize();
        return Err(error.context(format!(
            "A16 projection launch failed; synchronization: {sync:?}"
        )));
    }
    ctx.synchronize()?;
    Ok(Output { values, unrounded })
}
