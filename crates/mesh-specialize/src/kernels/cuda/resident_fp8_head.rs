//! Opt-in A16 head schedules over verified FP8 resident weights.
use super::{
    driver::{Buffer, Context, Module},
    mlp_workspace_projection::{Arithmetic, Binding},
    resident_fp8::Output,
};
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) fn run<'a>(
    ctx: &'a Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    binding: &Binding<'_, '_>,
    rows: usize,
    sliced: bool,
) -> Result<Output<'a>> {
    let (width, channels) = (binding.width, binding.channels);
    validate_shape(rows, channels, width, sliced)?;
    ensure!(
        binding.owner.belongs_to(ctx)
            && input.belongs_to(ctx)
            && module.belongs_to(ctx)
            && input.len() == rows * width * 2
            && matches!(binding.arithmetic, Arithmetic::Fp8),
        "A16 head input, weight binding, or context mismatch"
    );
    if !sliced {
        return super::resident_fp8_a16::run(
            ctx,
            module,
            input,
            binding.weights,
            [channels, width],
        );
    }
    let values = Buffer::new(ctx, rows * channels * 2)?;
    let unrounded = Buffer::new(ctx, rows * channels * 4)?;
    let mut pointers = [
        input.pointer(),
        binding.weights[0],
        binding.weights[1],
        values.pointer(),
        unrounded.pointer(),
    ];
    let mut dims = [
        u32::try_from(rows)?,
        u32::try_from(channels)?,
        u32::try_from(width)?,
    ];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    // SAFETY: The FP8 projection creates this binding only after validating resident
    // weight/scale extents. Context, input and kernel dimensions are checked above.
    // All disjoint allocations and their owners live through completion or error drain.
    let launch = unsafe {
        module.function("fp8_a16_head")?.launch(
            [u32::try_from(channels / 8)?, 1, 1],
            [512, 1, 1],
            0,
            &mut args,
        )
    };
    if let Err(error) = launch {
        let sync = ctx.synchronize();
        return Err(error.context(format!("A16 head launch failed; drain: {sync:?}")));
    }
    ctx.synchronize()?;
    Ok(Output { values, unrounded })
}

fn validate_shape(rows: usize, channels: usize, width: usize, sliced: bool) -> Result<()> {
    ensure!(
        (1..=8).contains(&rows)
            && (8..=262_144).contains(&channels)
            && channels.is_multiple_of(8)
            && (16..=32_768).contains(&width)
            && width.is_multiple_of(16),
        "A16 head requires M1..8, N multiple8, K multiple16 within kernel bounds"
    );
    ensure!(sliced || rows == 1, "A16 head GEMV requires one row");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_shape;

    #[test]
    fn rejects_unimplemented_shapes_before_launch() {
        assert!(validate_shape(1, 248_320, 5120, true).is_ok());
        assert!(validate_shape(8, 248_320, 5120, true).is_ok());
        assert!(validate_shape(1, 248_320, 5120, false).is_ok());
        for (m, n, k) in [
            (0, 8, 16),
            (9, 8, 16),
            (1, 9, 16),
            (1, 8, 17),
            (1, 262_152, 16),
            (1, 8, 32_784),
        ] {
            assert!(validate_shape(m, n, k, true).is_err());
        }
        assert!(validate_shape(2, 8, 16, false).is_err());
    }
}
