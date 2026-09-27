//! Independent qualification of resident BF16 to logical packed NVFP4 conversion.
use super::driver::{Buffer, Context, Module};
use crate::{entry_reference::round_bf16, nvfp4_quantize_reference as reference};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Quantized<'a> {
    pub(super) packed: Buffer<'a>,
    pub(super) scales: Buffer<'a>,
    pub(super) host: reference::Quantized,
    pub(super) report: Value,
}

pub(super) fn check<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: &Buffer<'_>,
    words: &[u16],
    shape: [usize; 2],
    global: f32,
) -> Result<Quantized<'a>> {
    let [rows, width] = shape;
    let expected = reference::run(words, rows, width, global)?;
    let packed = upload(context, &vec![0xa5; expected.packed.len()])?;
    let scales = upload(context, &vec![0xff; expected.scales.len()])?;
    let effective = upload(context, &vec![0xff; expected.effective.len() * 4])?;
    let mut pointers = [
        input.pointer(),
        packed.pointer(),
        scales.pointer(),
        effective.pointer(),
    ];
    let mut dimensions = [u32::try_from(rows)?, u32::try_from(width)?];
    let mut global = global;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|d| (d as *mut u32).cast()));
    args.push((&mut global as *mut f32).cast());
    // SAFETY: Four distinct live buffers match the ABI and validated exact extents.
    // One complete warp owns each group; every lane participates in shuffles.
    // The reference rejects invalid inputs and nonfinite scale intermediates.
    unsafe {
        module.function("nvfp4_quantize_bf16")?.launch(
            [u32::try_from(expected.scales.len())?, 1, 1],
            [32, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let actual_packed = download(&packed, expected.packed.len())?;
    let actual_scales = download(&scales, expected.scales.len())?;
    let actual_effective = download(&effective, expected.effective.len() * 4)?;
    ensure!(
        actual_packed == expected.packed,
        "NVFP4 activation packed codes differ"
    );
    ensure!(
        actual_scales == expected.scales,
        "NVFP4 activation local scales differ"
    );
    let expected_effective: Vec<_> = expected
        .effective
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    ensure!(
        actual_effective == expected_effective,
        "NVFP4 activation effective scales differ"
    );
    let report = json!({"all_passed":true,"shape":shape,"elements":words.len(),"groups":expected.scales.len(),"input_global_scale":global,
        "packed_codes_exact":true,"local_scales_exact":true,"effective_scales_exact":true,"device_input_resident":true,
        "profile":"FP32 (amax/6)*global -> E4M3FN RNE/saturate; zero scale -> 0.125; effective=local/global; signed E2M1 RNE/saturate; low nibble first"});
    Ok(Quantized {
        packed,
        scales,
        host: expected,
        report,
    })
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let levels = [
        0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0,
        6.0,
    ];
    let mut reports = Vec::new();
    for (rows, width, global) in [
        (1, 16, 1.0),
        (1, 16, 1.0625),
        (1, 16, 1.1875),
        (1, 16, 1000.0),
        (3, 32, 3.5),
        (2, 5120, 127.0),
    ] {
        let words: Vec<_> = (0..rows * width)
            .map(|i| round_bf16(if i / 16 % 3 == 1 { 0.0 } else { levels[i % 16] }))
            .collect();
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let input = upload(context, &bytes)?;
        reports.push(check(context, module, &input, &words, [rows, width], global)?.report);
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn download(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; count];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}
