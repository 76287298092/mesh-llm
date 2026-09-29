use super::validate::ValidatedView;
use super::super::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) fn run(
    context: &Context,
    module: &Module<'_>,
    object_bytes: &[u8],
    input_bf16: &[u16],
    validated: &ValidatedView,
) -> Result<Vec<u8>> {
    ensure!(module.belongs_to(context), "native Q8 PTX module context mismatch");
    let packed = upload(context, object_bytes)?;
    let input_bytes = input_bf16.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let input = upload(context, &input_bytes)?;
    let row_bytes = validated
        .source_rows
        .iter()
        .flat_map(|row| row.to_le_bytes())
        .collect::<Vec<_>>();
    let rows = upload(context, &row_bytes)?;
    let output_bytes = usize::try_from(validated.selected_rows)?
        .checked_mul(4)
        .context("native Q8 output extent overflows")?;
    let output = Buffer::new(context, output_bytes)?;
    let output_poison = 0x7fc1_2345_u32.to_le_bytes().repeat(output_bytes / 4);
    output.upload(&output_poison)?;
    let mut pointers = [input.pointer(), packed.pointer(), rows.pointer(), output.pointer()];
    let mut dimensions = [
        validated.selected_rows,
        validated.logical_k,
        validated.padded_k,
        validated.scale_offset,
    ];
    let mut arguments = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast::<c_void>()),
    );
    let launch = unsafe {
        // SAFETY: Four disjoint allocations and four scalar arguments match the kernel ABI.
        // Validation bounds all matrix reads; the one-warp launch writes one output per row.
        module.function("native_mtp_q8_gemv")?.launch(
            [validated.selected_rows, 1, 1],
            [32, 1, 1],
            0,
            &mut arguments,
        )
    };
    if let Err(error) = launch {
        let drain = context.synchronize();
        return Err(error.context(format!("native Q8 GEMV launch failed; drain: {drain:?}")));
    }
    context.synchronize()?;
    let mut result = vec![0; output_bytes];
    output.download(&mut result)?;
    Ok(result)
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
