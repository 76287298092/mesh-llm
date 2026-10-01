use super::super::{
    driver::{Buffer, Context, FunctionResources, Module},
    resident_native_mtp::{NativeMtpQ8Binding, ResidentNativeMtp},
};
use super::fixture::{Candidate, Fixture, K, ROWS, filled};
use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(in crate::kernels::cuda) struct DeviceInputRequest<'request, 'ctx> {
    pub(in crate::kernels::cuda) context: &'request Context,
    pub(in crate::kernels::cuda) module: &'request Module<'ctx>,
    pub(in crate::kernels::cuda) resident: &'request ResidentNativeMtp<'ctx>,
    pub(in crate::kernels::cuda) view: &'request Q8MatrixView,
    pub(in crate::kernels::cuda) input: &'request Buffer<'ctx>,
    pub(in crate::kernels::cuda) tokens: usize,
}

pub(super) struct BoundRequest<'request, 'owner, 'ctx> {
    pub(super) context: &'request Context,
    pub(super) module: &'request Module<'ctx>,
    pub(super) binding: &'request NativeMtpQ8Binding<'owner, 'ctx>,
    pub(super) candidate: Candidate,
    pub(super) input_bf16: &'request [u16],
}

pub(super) struct BoundOutputs {
    pub(super) outputs: [Vec<u16>; 2],
    pub(super) resources: FunctionResources,
}

pub(super) fn run(
    context: &Context,
    module: &Module<'_>,
    fixture: &Fixture,
) -> Result<[Vec<u16>; 2]> {
    ensure!(module.belongs_to(context), "Q8 FC module context mismatch");
    let codes = upload(context, &fixture.object[..fixture.code_bytes])?;
    let scales = upload(context, &fixture.object[fixture.code_bytes..])?;
    let input_extent = fixture
        .input
        .len()
        .checked_mul(2)
        .context("FC input bytes overflow")?;
    let mut input_bytes = filled(input_extent, 0_u8)?;
    for (bytes, word) in input_bytes
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(&fixture.input)
    {
        *bytes = word.to_le_bytes();
    }
    let input = upload(context, &input_bytes)?;
    let count = fixture
        .candidate
        .tokens()
        .checked_mul(ROWS)
        .context("FC output count overflow")?;
    let extent = count.checked_mul(2).context("FC output bytes overflow")?;
    let output = Buffer::new(context, extent)?;
    ensure!(
        [&codes, &scales, &input, &output]
            .iter()
            .all(|buffer| buffer.belongs_to(context) && buffer.pointer() % 16 == 0),
        "FC plane context/alignment mismatch"
    );
    let function = module.function(fixture.candidate.entry())?;
    let mut results = [Vec::new(), Vec::new()];
    for (repeat, result) in results.iter_mut().enumerate() {
        let poison = [0x7fc1_u16, 0xffc2][repeat];
        let mut bytes = filled(extent, 0_u8)?;
        for word in bytes.as_chunks_mut::<2>().0 {
            *word = poison.to_le_bytes();
        }
        output.upload(&bytes)?;
        let mut pointers = [
            codes.pointer(),
            scales.pointer(),
            input.pointer(),
            output.pointer(),
        ];
        let mut tokens = u32::try_from(fixture.candidate.tokens())?;
        let mut arguments = [std::ptr::null_mut(); 5];
        for (argument, pointer) in arguments[..4].iter_mut().zip(&mut pointers) {
            *argument = std::ptr::from_mut(pointer).cast::<c_void>();
        }
        arguments[4] = std::ptr::from_mut(&mut tokens).cast::<c_void>();
        // SAFETY: Four disjoint aligned full-shape planes and live u32 token storage
        // match the C4/T1 or C8/T5 ABI. Buffers remain live through synchronization.
        let launch = unsafe { function.launch([320, 1, 1], [256, 1, 1], 0, &mut arguments) };
        let drain = context.synchronize();
        if let Err(error) = launch {
            return Err(error.context(format!("FC launch failed; synchronized drain: {drain:?}")));
        }
        drain.context("FC synchronized launch drain failed")?;
        output.download(&mut bytes)?;
        result
            .try_reserve_exact(count)
            .context("FC readback allocation failed")?;
        result.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word)),
        );
    }
    Ok(results)
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

pub(super) fn run_bound(request: BoundRequest<'_, '_, '_>) -> Result<BoundOutputs> {
    ensure!(
        request.module.belongs_to(request.context),
        "Q8 FC module context mismatch"
    );
    ensure!(
        request.binding.view().shape == [ROWS, K],
        "resident Q8 FC view has the wrong shape"
    );
    let input_count = request
        .candidate
        .tokens()
        .checked_mul(K)
        .context("resident FC input count overflow")?;
    ensure!(
        request.input_bf16.len() == input_count,
        "resident FC activation extent mismatch"
    );
    let input_extent = input_count
        .checked_mul(2)
        .context("resident FC input bytes overflow")?;
    let mut input_bytes = Vec::new();
    input_bytes
        .try_reserve_exact(input_extent)
        .context("resident FC input allocation failed")?;
    for word in request.input_bf16 {
        input_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let input = upload(request.context, &input_bytes)?;
    let output_count = request
        .candidate
        .tokens()
        .checked_mul(ROWS)
        .context("resident FC output count overflow")?;
    let output_extent = output_count
        .checked_mul(2)
        .context("resident FC output bytes overflow")?;
    let output = Buffer::new(request.context, output_extent)?;
    let codes = request.binding.codes_pointer()?;
    let scales = request.binding.scale_bits_pointer()?;
    ensure!(
        [codes, scales, input.pointer(), output.pointer()]
            .iter()
            .all(|pointer| pointer % 16 == 0),
        "resident FC operand is not 16-byte aligned"
    );
    let function = request.module.function(request.candidate.entry())?;
    let resources = function.resources()?;
    ensure!(
        resources.max_threads_per_block >= 256,
        "Q8 FC block exceeds function limit"
    );
    let mut results = [Vec::new(), Vec::new()];
    for (result, poison) in results.iter_mut().zip([0x7fc1_u16, 0xffc2]) {
        result
            .try_reserve_exact(output_count)
            .context("resident FC readback allocation failed")?;
        let mut output_bytes = filled(output_extent, 0_u8)?;
        for word in output_bytes.as_chunks_mut::<2>().0 {
            *word = poison.to_le_bytes();
        }
        output.upload(&output_bytes)?;
        let mut pointers = [codes, scales, input.pointer(), output.pointer()];
        let mut tokens = u32::try_from(request.candidate.tokens())?;
        let mut arguments = [std::ptr::null_mut(); 5];
        for (argument, pointer) in arguments[..4].iter_mut().zip(&mut pointers) {
            *argument = std::ptr::from_mut(pointer).cast::<c_void>();
        }
        arguments[4] = std::ptr::from_mut(&mut tokens).cast::<c_void>();
        // SAFETY: [Category 8 - FFI boundary UB] The checked full FC view supplies
        // complete packed planes, this module owns the candidate ABI, and all
        // referenced device buffers and host argument storage live through drain.
        let launch = unsafe { function.launch([320, 1, 1], [256, 1, 1], 0, &mut arguments) };
        let drain = request.context.synchronize();
        if let Err(error) = launch {
            return Err(error.context(format!(
                "resident FC launch failed; synchronized drain: {drain:?}"
            )));
        }
        drain.context("resident FC synchronized launch drain failed")?;
        output.download(&mut output_bytes)?;
        result.extend(
            output_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word)),
        );
    }
    Ok(BoundOutputs {
        outputs: results,
        resources,
    })
}

pub(in crate::kernels::cuda) fn run_device_input<'request, 'ctx>(
    request: DeviceInputRequest<'request, 'ctx>,
) -> Result<Buffer<'request>> {
    ensure!(
        matches!(request.tokens, 1 | 5),
        "Q8 FC supports one or five device input rows"
    );
    ensure!(
        request.module.belongs_to(request.context) && request.resident.belongs_to(request.context),
        "resident FC owners belong to a different CUDA context"
    );
    let binding = request.resident.q8(request.view)?;
    validate_device_input(&binding, request.input, request.context, request.tokens)?;
    let output_bytes = request
        .tokens
        .checked_mul(ROWS)
        .and_then(|count| count.checked_mul(2))
        .context("resident FC output extent overflow")?;
    let output = Buffer::new(request.context, output_bytes)?;
    let codes = binding.codes_pointer()?;
    let scales = binding.scale_bits_pointer()?;
    ensure!(
        [codes, scales, request.input.pointer(), output.pointer()]
            .iter()
            .all(|pointer| pointer % 16 == 0),
        "resident FC operand is not 16-byte aligned"
    );
    let entry = match request.tokens {
        1 => "native_mtp_q8_sliced_k_fc_c4",
        5 => "native_mtp_q8_sliced_k_fc_c8",
        _ => unreachable!(),
    };
    let function = request.module.function(entry)?;
    ensure!(
        function.resources()?.max_threads_per_block >= 256,
        "Q8 FC block exceeds function limit"
    );
    let mut pointers = [codes, scales, request.input.pointer(), output.pointer()];
    let mut tokens = u32::try_from(request.tokens)?;
    let mut arguments = [std::ptr::null_mut(); 5];
    for (argument, pointer) in arguments[..4].iter_mut().zip(&mut pointers) {
        *argument = std::ptr::from_mut(pointer).cast::<c_void>();
    }
    arguments[4] = std::ptr::from_mut(&mut tokens).cast::<c_void>();
    // SAFETY: The verified complete FC planes, exact device input/output extents,
    // and scalar token count match the C4/C8 ABI and remain live through drain.
    let launch = unsafe { function.launch([320, 1, 1], [256, 1, 1], 0, &mut arguments) };
    let drain = request.context.synchronize();
    if let Err(error) = launch {
        return Err(error.context(format!(
            "resident FC launch failed; synchronized drain: {drain:?}"
        )));
    }
    drain.context("resident FC synchronized launch drain failed")?;
    Ok(output)
}

fn validate_device_input(
    binding: &NativeMtpQ8Binding<'_, '_>,
    input: &Buffer<'_>,
    context: &Context,
    tokens: usize,
) -> Result<()> {
    let view = binding.view();
    ensure!(
        view.shape == [ROWS, K] && view.padded_k == K && view.group_size == 32,
        "resident FC view geometry mismatch"
    );
    let code_bytes = ROWS
        .checked_mul(K)
        .context("resident FC code extent overflow")?;
    let scale_count = ROWS
        .checked_mul(K / 32)
        .context("resident FC scale count overflow")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("resident FC scale extent overflow")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("resident FC scale offset overflow")?;
    let parent_bytes = scale_offset
        .checked_add(scale_bytes)
        .context("resident FC parent extent overflow")?;
    ensure!(
        view.source_rows.iter().copied().eq(0..ROWS)
            && view.codes.offset == 0
            && view.codes.bytes == u64::try_from(code_bytes)?
            && view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?
            && view.scale_count == scale_count
            && binding.parent().bytes() == u64::try_from(parent_bytes)?,
        "resident FC saved view does not cover the complete physical parent"
    );
    let expected_input_bytes = tokens
        .checked_mul(K)
        .and_then(|count| count.checked_mul(2))
        .context("resident FC input extent overflow")?;
    ensure!(
        input.belongs_to(context) && input.len() == expected_input_bytes,
        "resident FC device input context or extent mismatch"
    );
    Ok(())
}
