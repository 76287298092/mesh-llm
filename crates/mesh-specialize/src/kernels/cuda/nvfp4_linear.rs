//! Logical NVFP4 projections with resident activation quantization and f64 oracle.
use super::{
    driver::{Buffer, Context, Module},
    nvfp4_quantize, projections,
};
use crate::{
    entry_reference::round_bf16,
    kernels::Nvfp4Projection,
    nvfp4_linear_reference::{self as reference, Matrix},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct CheckedLinear<'a> {
    pub(super) output: Buffer<'a>,
    pub(super) words: Vec<u16>,
    pub(super) report: Value,
}

pub(super) fn validate(weights: &Nvfp4Projection, width: usize) -> Result<()> {
    ensure!(
        (16..=32768).contains(&width) && width.is_multiple_of(16),
        "NVFP4 invalid width"
    );
    ensure!(
        (1..=32768).contains(&weights.channels),
        "NVFP4 invalid output channels"
    );
    ensure!(
        weights.packed.len() == weights.channels * width / 2
            && weights.scales.len() == weights.channels * width / 16,
        "NVFP4 weight extent mismatch"
    );
    ensure!(
        weights.scales.iter().all(|&v| v <= 126),
        "NVFP4 invalid unsigned scale code"
    );
    let product = weights.input_global * weights.weight_global;
    let factor = 1.0 / product;
    ensure!(
        [weights.input_global, weights.weight_global, product, factor]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0),
        "NVFP4 invalid global scales/factor"
    );
    Ok(())
}

pub(super) fn check<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: &Buffer<'_>,
    words: &[u16],
    weights: &Nvfp4Projection,
    shape: [usize; 2],
) -> Result<CheckedLinear<'a>> {
    let [rows, width] = shape;
    validate(weights, width)?;
    let quantized =
        nvfp4_quantize::check(context, module, input, words, shape, weights.input_global)?;
    let expected = reference::run(
        Matrix {
            packed: &quantized.host.packed,
            scales: &quantized.host.scales,
            rows,
            global: weights.input_global,
        },
        Matrix {
            packed: &weights.packed,
            scales: &weights.scales,
            rows: weights.channels,
            global: weights.weight_global,
        },
        width,
    )?;
    let packed_weight = upload(context, &weights.packed)?;
    let weight_scales = upload(context, &weights.scales)?;
    let count = rows * weights.channels;
    let output = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers = [
        quantized.packed.pointer(),
        packed_weight.pointer(),
        quantized.scales.pointer(),
        weight_scales.pointer(),
        output.pointer(),
        unrounded.pointer(),
    ];
    let mut dimensions = [
        u32::try_from(rows)?,
        u32::try_from(weights.channels)?,
        u32::try_from(width)?,
    ];
    let mut factor = 1.0 / (weights.input_global * weights.weight_global);
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|d| (d as *mut u32).cast()));
    args.push((&mut factor as *mut f32).cast());
    // SAFETY: Six distinct allocations, MNK and global factor match the ABI.
    // Independent logical reference validates shapes/domains and finite outputs.
    // A full warp executes every MMA, including tails; all buffers survive sync.
    unsafe {
        module.function("nvfp4_linear_warp4")?.launch(
            [dimensions[1].div_ceil(32), dimensions[0].div_ceil(16), 1],
            [128, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let mut bytes = vec![0; count * 2];
    output.download(&mut bytes)?;
    let words: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let mut bytes = vec![0; count * 4];
    unrounded.download(&mut bytes)?;
    let floats: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut report = projections::compare(&words, &floats, &expected)?;
    ensure!(
        report["passed"] == true,
        "NVFP4 projection numerical mismatch: {report}"
    );
    report["projection"] = json!(weights.name);
    report["shape_mnk"] = json!([rows, weights.channels, width]);
    report["input_quantization"] = quantized.report;
    report["weight_global_scale"] = json!(weights.weight_global);
    report["global_factor"] = json!(factor);
    report["device_input_resident"] = json!(true);
    Ok(CheckedLinear {
        output,
        words,
        report,
    })
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for [rows, channels, width] in [[1, 1, 16], [17, 13, 80]] {
        let words: Vec<_> = (0..rows * width)
            .map(|i| round_bf16(((i * 7 % 37) as f32 - 18.0) / 8.0))
            .collect();
        let weights = Nvfp4Projection {
            name: "signed-tail-fixture".into(),
            channels,
            packed: (0..channels * width / 2)
                .map(|i| ((i * 5 % 16) | (((i * 11 + 3) % 16) << 4)) as u8)
                .collect(),
            scales: (0..channels * width / 16)
                .map(|i| [0x00, 0x30, 0x38, 0x40, 0x42][(i + 1) % 5])
                .collect(),
            input_global: 3.5,
            weight_global: 2.0,
        };
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let input = upload(context, &bytes)?;
        reports.push(check(context, module, &input, &words, &weights, [rows, width])?.report);
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
