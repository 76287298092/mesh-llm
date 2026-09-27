//! Fused exact FP8 gate/up qualification against independent and separate controls.
use super::{
    driver::{Buffer, Context, Module},
    fp8_exact_trial, resident_activation,
};
use crate::{
    entry_reference::round_bf16, fp8_mlp_reference::Projection, projection_reference::QuantizedRows,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "fusion probe requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for shape in [
        [1, 7, 1],
        [3, 5, 17],
        [2, 9, 129],
        [17, 5, 513],
        [1, 5, 5120],
        [1, 3, 32768],
    ] {
        cases.push(check(&ctx, &module, shape, false)?);
    }
    cases.push(check(&ctx, &module, [3, 5, 5120], true)?);
    Ok(
        json!({"kind":"exact-fp8-fusion-qualification", "device":info,
        "all_passed":cases.iter().all(|v|v["all_passed"]==true), "cases":cases,
        "resources":module.function("fp8_swiglu_exact")?.resources()?,
        "scope":"synthetic independent arithmetic and separate GPU composition; no model performance evidence"}),
    )
}

fn codes(count: usize, multiplier: usize) -> Vec<u8> {
    (0..count)
        .map(|i| {
            let code = ((i * multiplier + i / 7 + 37) % 254) as u8;
            if code >= 127 { code + 1 } else { code }
        })
        .collect()
}
fn fixture(shape: [usize; 3], cancellation: bool) -> (QuantizedRows, Projection, Projection) {
    let [m, n, k] = shape;
    let a = if cancellation {
        (0..m * k)
            .map(|i| if i % 2 == 0 { 0x38 } else { 0xb8 })
            .collect()
    } else {
        codes(m * k, 5)
    };
    let projection = |multiplier| Projection {
        weights: if cancellation {
            vec![0x38; n * k]
        } else {
            codes(n * k, multiplier)
        },
        scales: (0..n)
            .map(|i| round_bf16(((i % 5 + 1) * (multiplier % 3 + 1)) as f32 / 64.0))
            .collect(),
        channels: n,
    };
    (
        QuantizedRows {
            codes: a,
            scales: (0..m).map(|i| (i % 3 + 1) as f32 / 1024.0).collect(),
        },
        projection(7),
        projection(11),
    )
}
fn check(
    ctx: &Context,
    module: &Module<'_>,
    shape: [usize; 3],
    cancellation: bool,
) -> Result<Value> {
    let [m, n, k] = shape;
    let (input, gate, up) = fixture(shape, cancellation);
    let oracle = crate::fp8_swiglu_exact_reference::run(&input, &gate, &up, k)?;
    let (gate_gpu, _) = fp8_exact_trial::execute(
        ctx,
        module,
        [m, n, k, 1],
        &input.codes,
        &gate.weights,
        &input.scales,
        &gate.scales,
    )?;
    let (up_gpu, _) = fp8_exact_trial::execute(
        ctx,
        module,
        [m, n, k, 1],
        &input.codes,
        &up.weights,
        &input.scales,
        &up.scales,
    )?;
    let projection_exact = gate_gpu == oracle.gate && up_gpu == oracle.up;
    let gate_gpu = upload(ctx, &words(&gate_gpu))?;
    let up_gpu = upload(ctx, &words(&up_gpu))?;
    let separate = resident_activation::run(ctx, module, &gate_gpu, &up_gpu, m * n)?;
    let (actual, raw) = fused(ctx, module, shape, &input, &gate, &up)?;
    let mut separate_bytes = vec![0; m * n * 2];
    separate.download(&mut separate_bytes)?;
    let separate = separate_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let differences = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    let oracle_diff = differences(&actual, &oracle.output);
    let separate_diff = differences(&actual, &separate);
    let raw_diff = raw
        .iter()
        .zip(&oracle.unrounded)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let finite = raw.iter().all(|v| v.is_finite());
    Ok(
        json!({"shape":shape,"cancellation":cancellation,"projection_control_exact":projection_exact,
        "oracle_bf16_differences":oracle_diff,"separate_gpu_bf16_differences":separate_diff,
        "oracle_fp32_differences":raw_diff,"finite":finite,
        "all_passed":projection_exact&&oracle_diff==0&&separate_diff==0&&raw_diff==0&&finite}),
    )
}
fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn words(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn fused(
    ctx: &Context,
    module: &Module<'_>,
    shape: [usize; 3],
    input: &QuantizedRows,
    gate: &Projection,
    up: &Projection,
) -> Result<(Vec<u16>, Vec<f32>)> {
    let [m, n, k] = shape;
    let buffers = [
        upload(ctx, &input.codes)?,
        upload(ctx, &gate.weights)?,
        upload(ctx, &up.weights)?,
        upload(
            ctx,
            &input
                .scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        upload(ctx, &words(&gate.scales))?,
        upload(ctx, &words(&up.scales))?,
        upload(ctx, &vec![0xa5; m * n * 2])?,
        upload(ctx, &vec![0xff; m * n * 4])?,
    ];
    let mut pointers = buffers.iter().map(Buffer::pointer).collect::<Vec<_>>();
    let mut dims = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    // SAFETY: Independently validated finite fixtures provide all eight disjoint ABI
    // extents. Tail-safe four-warp launch; buffers remain live until synchronization.
    unsafe {
        module.function("fp8_swiglu_exact")?.launch(
            [u32::try_from(n.div_ceil(4))?, dims[0], 1],
            [128, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    let mut out = vec![0; m * n * 2];
    let mut raw = vec![0; m * n * 4];
    buffers[6].download(&mut out)?;
    buffers[7].download(&mut raw)?;
    Ok((
        out.as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect(),
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
    ))
}
