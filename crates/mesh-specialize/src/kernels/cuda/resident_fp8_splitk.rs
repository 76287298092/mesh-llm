//! Experimental exact integer split-K scheduling for small FP8 batches.
use super::{
    driver::{Buffer, Context, Module},
    resident_fp8::Output,
};
use anyhow::{Result, ensure};
use std::ffi::c_void;

fn parse(value: Option<&str>) -> Result<Option<usize>> {
    match value {
        None | Some("off") => Ok(None),
        Some("2") => Ok(Some(2)),
        Some("4") => Ok(Some(4)),
        Some("8") => Ok(Some(8)),
        Some("16") => Ok(Some(16)),
        _ => anyhow::bail!("MESH_SPECIALIZE_FP8_SPLIT_K must be off, 2, 4, 8, or 16"),
    }
}
pub(super) fn configured_splits() -> Result<Option<usize>> {
    static VALUE: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    match VALUE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_FP8_SPLIT_K") {
        Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.to_string()),
    }) {
        Ok(value) => Ok(*value),
        Err(error) => Err(anyhow::anyhow!(error.clone())),
    }
}
pub(super) fn selected_splits(rows: usize, channels: usize) -> Result<Option<usize>> {
    Ok(configured_splits()?.filter(|_| (4..16).contains(&rows) && channels < 16_384))
}

/// Caller validates row-major A/W and scales, contexts, and finite numeric inputs.
pub(super) fn run<'a>(
    ctx: &'a Context,
    module: &Module<'_>,
    inputs: [u64; 4],
    shape: [usize; 3],
    splits: usize,
) -> Result<Output<'a>> {
    let [m, n, k] = shape;
    ensure!(
        (1..=2048).contains(&m)
            && (1..=262_144).contains(&n)
            && (1..=32_768).contains(&k)
            && (1..=32).contains(&splits),
        "invalid split-K shape"
    );
    ensure!(module.belongs_to(ctx), "split-K module context mismatch");
    let count = m
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("split-K output extent overflow"))?;
    let partial_bytes = count
        .checked_mul(splits)
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("split-K partial extent overflow"))?;
    let split_kernel = module.function("fp8_verify_splitk")?;
    let reduce_kernel = module.function("fp8_verify_reduce")?;
    let partial = Buffer::new(ctx, partial_bytes)?;
    let values = Buffer::new(ctx, count * 2)?;
    let unrounded = Buffer::new(ctx, count * 4)?;
    let mut pointers = [inputs[0], inputs[1], partial.pointer()];
    let mut dims = [
        u32::try_from(m)?,
        u32::try_from(n)?,
        u32::try_from(k)?,
        u32::try_from(splits)?,
    ];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    // SAFETY: Caller-validated inputs and checked split-major partial extents obey the
    // exact seven-argument ABI. Every warp owns disjoint output slots for one K range.
    let result = unsafe {
        split_kernel.launch(
            [
                u32::try_from(n.div_ceil(16))?,
                u32::try_from(m.div_ceil(8))?,
                dims[3],
            ],
            [32, 1, 1],
            0,
            &mut args,
        )
    };
    if let Err(error) = result {
        return Err(failed_launch(ctx, error));
    }
    let mut pointers = [
        partial.pointer(),
        inputs[2],
        inputs[3],
        values.pointer(),
        unrounded.pointer(),
    ];
    let mut dims = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(splits)?];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    // SAFETY: The same ordered stream completes all partial writes before reduction;
    // checked disjoint output buffers and every input remain live through synchronization.
    let result = unsafe {
        reduce_kernel.launch(
            [u32::try_from(count.div_ceil(256))?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )
    };
    if let Err(error) = result {
        return Err(failed_launch(ctx, error));
    }
    ctx.synchronize()?;
    Ok(Output { values, unrounded })
}
fn failed_launch(ctx: &Context, error: anyhow::Error) -> anyhow::Error {
    let sync = ctx.synchronize();
    error.context(format!("split-K launch failed; synchronization: {sync:?}"))
}

#[cfg(test)]
mod tests {
    use super::parse;
    #[test]
    fn selection_is_bounded_and_explicit() {
        assert_eq!(parse(None).unwrap(), None);
        assert_eq!(parse(Some("off")).unwrap(), None);
        for (text, splits) in [("2", 2), ("4", 4), ("8", 8), ("16", 16)] {
            assert_eq!(parse(Some(text)).unwrap(), Some(splits));
        }
        for text in ["", "0", "1", "32", "native"] {
            assert!(parse(Some(text)).is_err());
        }
    }
}
