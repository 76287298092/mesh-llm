use super::super::driver::{Buffer, Context, Module};
use super::validate::ValidatedView;
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) struct Output {
    pub(super) raw_f32: Vec<f32>,
    pub(super) logits_bf16: Vec<u16>,
}

pub(super) struct Input<'a> {
    pub(super) object_bytes: &'a [u8],
    pub(super) input_bf16: &'a [u16],
    pub(super) validated: &'a ValidatedView,
}

pub(super) fn run(context: &Context, module: &Module<'_>, input: &Input<'_>) -> Result<Output> {
    ensure!(
        module.belongs_to(context),
        "native Q4 PTX module context mismatch"
    );
    let packed = upload(context, input.object_bytes)?;
    let input_bytes = input
        .input_bf16
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let activations = upload(context, &input_bytes)?;
    let row_bytes = input
        .validated
        .source_rows
        .iter()
        .flat_map(|row| row.to_le_bytes())
        .collect::<Vec<_>>();
    let rows = upload(context, &row_bytes)?;
    let output_count = usize::try_from(input.validated.selected_rows)?;
    let output_f32 = Buffer::new(
        context,
        output_count
            .checked_mul(4)
            .context("native Q4 FP32 output extent overflows")?,
    )?;
    let output_bf16 = Buffer::new(
        context,
        output_count
            .checked_mul(2)
            .context("native Q4 BF16 output extent overflows")?,
    )?;
    output_f32.upload(&0x7fc1_2345_u32.to_le_bytes().repeat(output_count))?;
    output_bf16.upload(&0x7fc1_u16.to_le_bytes().repeat(output_count))?;
    let mut pointers = [
        activations.pointer(),
        packed.pointer(),
        rows.pointer(),
        output_f32.pointer(),
        output_bf16.pointer(),
    ];
    let mut dimensions = [
        input.validated.selected_rows,
        input.validated.logical_k,
        input.validated.padded_k,
        input.validated.scale_offset,
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
    let grid_rows = input.validated.selected_rows.div_ceil(4);
    let launch = unsafe {
        module.function("native_mtp_q4_head_gemv")?.launch(
            [grid_rows, 1, 1],
            [128, 1, 1],
            0,
            &mut arguments,
        )
    };
    if let Err(error) = launch {
        let drain = context.synchronize();
        return Err(error.context(format!(
            "native Q4 head GEMV launch failed; drain: {drain:?}"
        )));
    }
    context.synchronize()?;
    let mut raw_bytes = vec![0; output_count * 4];
    output_f32.download(&mut raw_bytes)?;
    let raw_f32 = raw_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let mut bf16_bytes = vec![0; output_count * 2];
    output_bf16.download(&mut bf16_bytes)?;
    let logits_bf16 = bf16_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect();
    Ok(Output {
        raw_f32,
        logits_bf16,
    })
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
