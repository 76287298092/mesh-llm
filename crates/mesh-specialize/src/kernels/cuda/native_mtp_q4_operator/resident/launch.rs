use super::super::super::{
    driver::{Buffer, Context, Module},
    resident_native_mtp::ResidentNativeMtp,
};
use super::super::validate::ValidatedView;
use crate::packages::qwen3_8_27b::native_mtp_views::Q4MatrixView;
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) struct BoundInput<'run, 'ctx> {
    pub(super) context: &'run Context,
    pub(super) module: &'run Module<'ctx>,
    pub(super) resident: &'run ResidentNativeMtp<'ctx>,
    pub(super) view: &'run Q4MatrixView,
    pub(super) input_bf16: &'run [u16],
    pub(super) validated: &'run ValidatedView,
    pub(super) raw_poison: u32,
    pub(super) bf16_poison: u16,
}

pub(super) struct BoundOutput {
    pub(super) raw_f32_bits: Vec<u32>,
    pub(super) logits_bf16: Vec<u16>,
}

pub(super) fn run_bound(request: BoundInput<'_, '_>) -> Result<BoundOutput> {
    let BoundInput {
        context,
        module,
        resident,
        view,
        input_bf16,
        validated,
        raw_poison,
        bf16_poison,
    } = request;
    ensure!(
        module.belongs_to(context),
        "native Q4 module context mismatch"
    );
    ensure!(
        resident.belongs_to(context),
        "native Q4 resident context mismatch"
    );
    let binding = resident.q4(view)?;
    let view = binding.view();
    ensure!(
        input_bf16.len() == usize::try_from(validated.logical_k)?,
        "native Q4 activation extent mismatch"
    );
    let parent = binding.parent();
    let parent_pointer = parent.pointer()?;
    ensure!(
        binding.codes_pointer()?
            == parent_pointer
                .checked_add(view.codes.offset)
                .context("native Q4 code pointer overflows")?,
        "native Q4 code pointer differs from saved parent plane"
    );
    ensure!(
        binding.scale_bits_pointer()?
            == parent_pointer
                .checked_add(view.scale_bits.offset)
                .context("native Q4 scale pointer overflows")?,
        "native Q4 scale pointer differs from saved parent plane"
    );

    let mut input_bytes = Vec::new();
    input_bytes
        .try_reserve_exact(
            input_bf16
                .len()
                .checked_mul(2)
                .context("Q4 activation extent overflows")?,
        )
        .context("cannot reserve native Q4 activation upload")?;
    for word in input_bf16 {
        input_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let activations = upload(context, &input_bytes)?;

    let mut row_bytes = Vec::new();
    row_bytes
        .try_reserve_exact(
            validated
                .source_rows
                .len()
                .checked_mul(4)
                .context("Q4 row-map extent overflows")?,
        )
        .context("cannot reserve native Q4 row-map upload")?;
    for row in &validated.source_rows {
        row_bytes.extend_from_slice(&row.to_le_bytes());
    }
    let source_rows = upload(context, &row_bytes)?;

    let output_count = usize::try_from(validated.selected_rows)?;
    let raw_bytes = output_count
        .checked_mul(4)
        .context("Q4 raw output extent overflows")?;
    let bf16_bytes = output_count
        .checked_mul(2)
        .context("Q4 BF16 output extent overflows")?;
    let output_f32 = Buffer::new(context, raw_bytes)?;
    let output_bf16 = Buffer::new(context, bf16_bytes)?;
    output_f32.upload(&repeated_bytes(raw_poison.to_le_bytes(), output_count)?)?;
    output_bf16.upload(&repeated_bytes(bf16_poison.to_le_bytes(), output_count)?)?;

    let mut pointers = [
        activations.pointer(),
        parent_pointer,
        source_rows.pointer(),
        output_f32.pointer(),
        output_bf16.pointer(),
    ];
    let mut dimensions = [
        validated.selected_rows,
        validated.logical_k,
        validated.padded_k,
        validated.scale_offset,
    ];
    let mut arguments = [
        std::ptr::from_mut(&mut pointers[0]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[1]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[2]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[3]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[4]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[0]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[1]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[2]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[3]).cast::<c_void>(),
    ];
    let launch = unsafe {
        // SAFETY: Categories #3, #8, #10. The binding borrow retains the parent, local
        // buffers survive the sync, ABI argument types match, and Q4 validation bounds reads.
        module.function("native_mtp_q4_head_gemv")?.launch(
            [validated.selected_rows.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            &mut arguments,
        )
    };
    if let Err(error) = launch {
        let drain = context.synchronize();
        return Err(error.context(format!(
            "native Q4 bound GEMV launch failed; drain: {drain:?}"
        )));
    }
    context.synchronize()?;

    let mut raw_output = vec![0; raw_bytes];
    output_f32.download(&mut raw_output)?;
    let (raw_words, raw_remainder) = raw_output.as_chunks::<4>();
    ensure!(
        raw_remainder.is_empty(),
        "native Q4 raw output has a partial word"
    );
    let raw_f32_bits = raw_words
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let mut bf16_output = vec![0; bf16_bytes];
    output_bf16.download(&mut bf16_output)?;
    let (bf16_words, bf16_remainder) = bf16_output.as_chunks::<2>();
    ensure!(
        bf16_remainder.is_empty(),
        "native Q4 BF16 output has a partial word"
    );
    let logits_bf16 = bf16_words
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect();
    Ok(BoundOutput {
        raw_f32_bits,
        logits_bf16,
    })
}

pub(in crate::kernels::cuda) struct DeviceInput<'run, 'ctx> {
    pub(in crate::kernels::cuda) context: &'run Context,
    pub(in crate::kernels::cuda) module: &'run Module<'ctx>,
    pub(in crate::kernels::cuda) resident: &'run ResidentNativeMtp<'ctx>,
    pub(in crate::kernels::cuda) view: &'run Q4MatrixView,
    pub(in crate::kernels::cuda) input: &'run Buffer<'ctx>,
}

pub(in crate::kernels::cuda) fn run_device_input<'run, 'ctx>(
    request: DeviceInput<'run, 'ctx>,
) -> Result<Buffer<'run>> {
    ensure!(
        request.module.belongs_to(request.context) && request.resident.belongs_to(request.context),
        "native Q4 owners belong to a different CUDA context"
    );
    let binding = request.resident.q4(request.view)?;
    let view = binding.view();
    ensure!(
        view.shape == [131_072, 5_120]
            && view.padded_k == 5_120
            && view.group_size == 64
            && view.source_rows.iter().copied().eq(0..131_072),
        "native Q4 shortlist view differs from the checked full parent"
    );
    ensure!(
        request.input.belongs_to(request.context) && request.input.len() == 5_120 * 2,
        "native Q4 device input context or extent mismatch"
    );
    let code_bytes = 131_072_usize
        .checked_mul(5_120 / 2)
        .context("native Q4 code extent overflow")?;
    let scale_count = 131_072_usize
        .checked_mul(5_120 / 64)
        .context("native Q4 scale count overflow")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("native Q4 scale extent overflow")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("native Q4 scale offset overflow")?;
    ensure!(
        view.codes.offset == 0
            && view.codes.bytes == u64::try_from(code_bytes)?
            && view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?
            && view.scale_count == scale_count
            && binding.parent().bytes()
                == u64::try_from(
                    scale_offset
                        .checked_add(scale_bytes)
                        .context("native Q4 parent extent overflow")?
                )?,
        "native Q4 planes do not cover their verified physical parent"
    );
    let row_map_bytes = view
        .source_rows
        .len()
        .checked_mul(4)
        .context("native Q4 row map extent overflow")?;
    let mut row_map = Vec::new();
    row_map
        .try_reserve_exact(row_map_bytes)
        .context("native Q4 row map allocation failed")?;
    for row in &view.source_rows {
        row_map.extend_from_slice(&u32::try_from(*row)?.to_le_bytes());
    }
    let rows = Buffer::new(request.context, row_map_bytes)?;
    rows.upload(&row_map)?;
    let raw = Buffer::new(request.context, 131_072 * 4)?;
    let logits = Buffer::new(request.context, 131_072 * 2)?;
    let mut pointers = [
        request.input.pointer(),
        binding.parent().pointer()?,
        rows.pointer(),
        raw.pointer(),
        logits.pointer(),
    ];
    let mut dimensions = [131_072_u32, 5_120, 5_120, u32::try_from(scale_offset)?];
    let mut arguments = [
        std::ptr::from_mut(&mut pointers[0]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[1]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[2]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[3]).cast::<c_void>(),
        std::ptr::from_mut(&mut pointers[4]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[0]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[1]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[2]).cast::<c_void>(),
        std::ptr::from_mut(&mut dimensions[3]).cast::<c_void>(),
    ];
    let launch = unsafe {
        // SAFETY: The checked Q4 binding retains the verified full parent, the
        // saved row map and input/output buffers have exact extents, and all live
        // storage matches the nine-argument GEMV ABI through synchronization.
        request.module.function("native_mtp_q4_head_gemv")?.launch(
            [32_768, 1, 1],
            [128, 1, 1],
            0,
            &mut arguments,
        )
    };
    let drain = request.context.synchronize();
    if let Err(error) = launch {
        return Err(error.context(format!(
            "native Q4 device GEMV launch failed; synchronized drain: {drain:?}"
        )));
    }
    drain.context("native Q4 device GEMV synchronized drain failed")?;
    Ok(logits)
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

fn repeated_bytes<const N: usize>(pattern: [u8; N], count: usize) -> Result<Vec<u8>> {
    let capacity = count
        .checked_mul(N)
        .context("native Q4 poison extent overflows")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .context("cannot reserve native Q4 poison buffer")?;
    for _ in 0..count {
        bytes.extend_from_slice(&pattern);
    }
    Ok(bytes)
}
