// SPDX-License-Identifier: Apache-2.0
// NInfer contributors, pin e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d.
// Modified: parent-bound Rust driver launch for identity-row Q8 MTP projections.
use super::super::{driver::Buffer, resident_native_mtp::NativeMtpQ8Binding};
use super::{
    DeviceProjectionRequest, OutputInitialization, ProjectionKind, ResidentProjectionRequest,
};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) fn project<'a, 'ctx>(
    request: ResidentProjectionRequest<'a, 'ctx>,
) -> Result<Buffer<'a>> {
    let context = request.context;
    let launched = launch(request);
    let drain = context.synchronize();
    let launched = launched.map_err(|error| {
        error.context(format!(
            "Q8 projection launch failed; synchronized drain: {drain:?}"
        ))
    })?;
    drain.context("Q8 projection synchronized drain failed")?;
    Ok(launched)
}

pub(super) fn launch<'a, 'ctx>(request: ResidentProjectionRequest<'a, 'ctx>) -> Result<Buffer<'a>> {
    ensure!(
        request.module.belongs_to(request.context),
        "Q8 projection module context mismatch"
    );
    ensure!(
        request.resident.belongs_to(request.context),
        "Q8 projection parent context mismatch"
    );
    ensure!(
        matches!(request.tokens, 1 | 5),
        "Q8 projection supports T1 or T5"
    );
    request
        .kind
        .validate_carrier(request.resident, request.carrier_view)?;
    let [rows, k] = request.kind.dimensions();
    let parent_rows = request.kind.parent_rows();
    let binding = request.resident.q8(request.carrier_view)?;
    validate_parent(&binding, request.kind)?;
    let input_bytes = request
        .tokens
        .checked_mul(k)
        .and_then(|count| count.checked_mul(2))
        .context("Q8 projection input extent overflow")?;
    ensure!(
        request.input.belongs_to(request.context) && request.input.len() == input_bytes,
        "Q8 projection input device or extent mismatch"
    );
    let output_bytes = request
        .tokens
        .checked_mul(rows)
        .and_then(|count| count.checked_mul(2))
        .context("Q8 projection output extent overflow")?;
    let output = Buffer::new(request.context, output_bytes)?;
    match request.initialization {
        OutputInitialization::Poison(word) => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(output_bytes)
                .context("Q8 poison allocation failed")?;
            for _ in 0..output_bytes / 2 {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            output.upload(&bytes)?;
        }
    }
    let codes = binding.codes_pointer()?;
    let scales = binding.scale_bits_pointer()?;
    let parent_pointer = binding.parent().pointer()?;
    ensure!(
        codes
            == parent_pointer
                .checked_add(binding.view().codes.offset)
                .context("Q8 code parent pointer overflow")?,
        "Q8 codes pointer does not refer to the checked physical parent"
    );
    ensure!(
        scales
            == parent_pointer
                .checked_add(binding.view().scale_bits.offset)
                .context("Q8 scale parent pointer overflow")?,
        "Q8 scales pointer does not refer to the checked physical parent"
    );
    ensure!(
        [codes, scales, request.input.pointer(), output.pointer()]
            .iter()
            .all(|pointer| pointer % 16 == 0),
        "Q8 projection plane is not 16-byte aligned"
    );
    let function = request.module.function(
        request
            .kind
            .entry(request.tokens)
            .context("Q8 projection schedule is unsupported")?,
    )?;
    let threads = request
        .kind
        .split_warps(request.tokens)
        .context("Q8 projection split count is unsupported")?
        .checked_mul(32)
        .context("Q8 projection thread count overflow")?;
    let resources = function.resources()?;
    ensure!(
        resources.max_threads_per_block >= i32::try_from(threads)?,
        "Q8 projection exceeds function block limit"
    );
    let mut pointers = [codes, scales, request.input.pointer(), output.pointer()];
    let mut tokens = u32::try_from(request.tokens)?;
    let mut arguments = [std::ptr::null_mut(); 5];
    for (argument, pointer) in arguments[..4].iter_mut().zip(&mut pointers) {
        *argument = std::ptr::from_mut(pointer).cast::<c_void>();
    }
    arguments[4] = std::ptr::from_mut(&mut tokens).cast::<c_void>();
    // SAFETY: Saved parent planes, full extents, context ownership and argument storage are checked
    // above. The output and source buffers remain live until the caller drains the default stream.
    unsafe {
        function.launch(
            [u32::try_from(parent_rows / 16)?, 1, 1],
            [u32::try_from(threads)?, 1, 1],
            0,
            &mut arguments,
        )
    }?;
    Ok(output)
}

fn validate_parent(binding: &NativeMtpQ8Binding<'_, '_>, kind: ProjectionKind) -> Result<()> {
    let view = binding.view();
    let k = kind.physical_k();
    let parent_rows = kind.parent_rows();
    let parent_code_bytes = parent_rows
        .checked_mul(k)
        .context("Q8 parent code extent overflow")?;
    let parent_scale_count = parent_rows
        .checked_mul(k / 32)
        .context("Q8 parent scale count overflow")?;
    let parent_scale_bytes = parent_scale_count
        .checked_mul(2)
        .context("Q8 parent scale extent overflow")?;
    let padded_code_bytes = parent_code_bytes
        .checked_add((256 - parent_code_bytes % 256) % 256)
        .context("Q8 projection padded code extent overflow")?;
    ensure!(
        view.shape[1] == k && view.padded_k == k && view.group_size == 32,
        "Q8 projection logical view K/group mismatch"
    );
    ensure!(
        view.codes.bytes == u64::try_from(parent_code_bytes)?,
        "Q8 selected binding does not retain the complete packed parent code plane"
    );
    let parent_bytes = padded_code_bytes
        .checked_add(parent_scale_bytes)
        .context("Q8 physical parent extent overflow")?;
    ensure!(
        binding.parent().bytes() == u64::try_from(parent_bytes)?,
        "Q8 selected binding does not retain the complete physical parent"
    );
    ensure!(
        view.codes.offset == 0 && view.codes.bytes == u64::try_from(parent_code_bytes)?,
        "Q8 projection parent codes do not cover the full identity-row matrix"
    );
    ensure!(
        view.scale_bits.offset == u64::try_from(padded_code_bytes)?
            && view.scale_bits.bytes == u64::try_from(parent_scale_bytes)?
            && view.scale_count == parent_scale_count,
        "Q8 projection parent scales do not cover the full identity-row matrix"
    );
    ensure!(
        binding.parent().object_id() == view.object_id,
        "Q8 projection binding changed physical parent"
    );
    Ok(())
}

pub(in crate::kernels::cuda) fn project_device<'a, 'ctx>(
    request: DeviceProjectionRequest<'a, 'ctx>,
) -> Result<Buffer<'a>> {
    ensure!(
        request.module.belongs_to(request.context) && request.resident.belongs_to(request.context),
        "Q8 projection owners belong to a different CUDA context"
    );
    ensure!(
        matches!(request.tokens, 1 | 5),
        "Q8 projection supports T1 or T5"
    );
    request
        .kind
        .validate_carrier(request.resident, request.carrier_view)?;
    let [rows, k] = request.kind.dimensions();
    let binding = request.resident.q8(request.carrier_view)?;
    validate_parent(&binding, request.kind)?;
    let input_bytes = request
        .tokens
        .checked_mul(k)
        .and_then(|count| count.checked_mul(2))
        .context("Q8 projection input extent overflow")?;
    ensure!(
        request.input.belongs_to(request.context) && request.input.len() == input_bytes,
        "Q8 projection device input context or extent mismatch"
    );
    let output_bytes = request
        .tokens
        .checked_mul(rows)
        .and_then(|count| count.checked_mul(2))
        .context("Q8 projection output extent overflow")?;
    let output = Buffer::new(request.context, output_bytes)?;
    let codes = binding.codes_pointer()?;
    let scales = binding.scale_bits_pointer()?;
    let parent_pointer = binding.parent().pointer()?;
    ensure!(
        codes
            == parent_pointer
                .checked_add(binding.view().codes.offset)
                .context("Q8 code parent pointer overflow")?
            && scales
                == parent_pointer
                    .checked_add(binding.view().scale_bits.offset)
                    .context("Q8 scale parent pointer overflow")?,
        "Q8 projection planes do not refer to the checked physical parent"
    );
    ensure!(
        [codes, scales, request.input.pointer(), output.pointer()]
            .iter()
            .all(|pointer| pointer % 16 == 0),
        "Q8 projection operand is not 16-byte aligned"
    );
    let function = request.module.function(
        request
            .kind
            .entry(request.tokens)
            .context("Q8 projection schedule is unsupported")?,
    )?;
    let threads = request
        .kind
        .split_warps(request.tokens)
        .context("Q8 projection split count is unsupported")?
        .checked_mul(32)
        .context("Q8 projection thread count overflow")?;
    ensure!(
        function.resources()?.max_threads_per_block >= i32::try_from(threads)?,
        "Q8 projection exceeds function block limit"
    );
    let mut pointers = [codes, scales, request.input.pointer(), output.pointer()];
    let mut tokens = u32::try_from(request.tokens)?;
    let mut arguments = [std::ptr::null_mut(); 5];
    for (argument, pointer) in arguments[..4].iter_mut().zip(&mut pointers) {
        *argument = std::ptr::from_mut(pointer).cast::<c_void>();
    }
    arguments[4] = std::ptr::from_mut(&mut tokens).cast::<c_void>();
    // SAFETY: Verified parent planes, device input/output extents and scalar rows
    // satisfy the projection ABI; the borrowed binding and buffers live through drain.
    let launch = unsafe {
        function.launch(
            [u32::try_from(request.kind.parent_rows() / 16)?, 1, 1],
            [u32::try_from(threads)?, 1, 1],
            0,
            &mut arguments,
        )
    };
    let drain = request.context.synchronize();
    if let Err(error) = launch {
        return Err(error.context(format!(
            "Q8 projection launch failed; synchronized drain: {drain:?}"
        )));
    }
    drain.context("Q8 projection synchronized drain failed")?;
    Ok(output)
}
