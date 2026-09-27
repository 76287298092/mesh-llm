//! Final decoder residual boundary, independently checked in logical element order.
use super::driver::{Buffer, Context, Module};
use crate::{entry_reference::round_bf16, residual_add_reference};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) residual: &'a Buffer<'ctx>,
    pub(super) residual_words: &'a [u16],
    pub(super) branch: &'a Buffer<'ctx>,
    pub(super) branch_words: &'a [u16],
}
pub(super) struct CheckedResidual {
    pub(super) words: Vec<u16>,
    pub(super) report: Value,
}
pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: Input<'_, '_>,
) -> Result<CheckedResidual> {
    let expected = residual_add_reference::run(input.residual_words, input.branch_words)?;
    let output = upload(context, &vec![0xa5; expected.len() * 2])?;
    let mut pointers = [
        input.residual.pointer(),
        input.branch.pointer(),
        output.pointer(),
    ];
    let mut count = u32::try_from(expected.len())?;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.push((&mut count as *mut u32).cast());
    // SAFETY: Three distinct allocations match the validated element count and
    // kernel ABI. Tail threads return without accessing memory; sync precedes read.
    unsafe {
        module.function("residual_add_bf16")?.launch(
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let mut bytes = vec![0; expected.len() * 2];
    output.download(&mut bytes)?;
    let actual: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    ensure!(actual == expected, "final residual BF16 mismatch");
    let report = json!({"all_passed":true,"elements":expected.len(),"bf16_exact":true,"device_inputs_resident":true});
    Ok(CheckedResidual {
        words: actual,
        report,
    })
}
pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for count in [1, 257] {
        let left: Vec<_> = (0..count)
            .map(|i| round_bf16([1.0, -0.0, -2.0, 0.25][i % 4]))
            .collect();
        let right: Vec<_> = (0..count)
            .map(|i| round_bf16([1.0 / 256.0, -0.0, 2.0, 0.5][i % 4]))
            .collect();
        let ld = upload(
            context,
            &left
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let rd = upload(
            context,
            &right
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        reports.push(
            check(
                context,
                module,
                Input {
                    residual: &ld,
                    residual_words: &left,
                    branch: &rd,
                    branch_words: &right,
                },
            )?
            .report,
        );
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let b = Buffer::new(context, bytes.len())?;
    b.upload(bytes)?;
    Ok(b)
}
