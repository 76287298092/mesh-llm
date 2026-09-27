//! Independent synthetic qualification for experimental projection profiles.
use super::{
    driver::{Buffer, Context, Module},
    resident_workspace::ResidentWorkspace,
};
use crate::{engine::workspace::WorkspaceLayout, entry_reference::round_bf16};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "feature probes require SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for k in [1, 17, 128, 129, 5120] {
        cases.push(decode(&ctx, &module, 5, k)?);
    }
    for shape in [
        [1, 5, 1],
        [2, 19, 17],
        [17, 65, 64],
        [32, 64, 128],
        [33, 65, 129],
        [2, 5, 5120],
    ] {
        cases.push(prefill(&ctx, &module, shape)?);
    }
    let workspace = workspace(&ctx)?;
    Ok(
        json!({"kind":"experimental-feature-projection-probe","device":info,
        "all_passed":cases.iter().all(|v|v["all_passed"]==true),"cases":cases,"workspace":workspace,
        "decode_resources":module.function("fp8_a16_decode")?.resources()?,
        "prefill_resources":module.function("fp8_prefill_native")?.resources()?,
        "scope":"synthetic independent numerical probes only; no model or Ninfer performance qualification"}),
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
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn codes(count: usize, multiplier: usize) -> Vec<u8> {
    // Exercise every finite E4M3 sign/exponent/mantissa code, not only powers of two.
    (0..count)
        .map(|i| {
            let value = ((i * multiplier + i / 7 + 37) % 254) as u8;
            if value >= 127 { value + 1 } else { value }
        })
        .collect()
}

fn decode(ctx: &Context, module: &Module<'_>, n: usize, k: usize) -> Result<Value> {
    let input = (0..k)
        .map(|i| round_bf16((i as i32 % 19 - 9) as f32 / 8.0))
        .collect::<Vec<_>>();
    let weight = codes(n * k, 7);
    let scales = (0..n)
        .map(|i| round_bf16((i % 3 + 1) as f32 / 4.0))
        .collect::<Vec<_>>();
    let reference = crate::fp8_a16_decode_reference::run(&input, &weight, &scales, n, k)?;
    let input = upload(ctx, &words(&input))?;
    let weight = upload(ctx, &weight)?;
    let scales = upload(ctx, &words(&scales))?;
    let out = upload(ctx, &vec![0xa5; n * 2])?;
    let raw = upload(ctx, &vec![0xa5; n * 4])?;
    let mut pointers = [
        input.pointer(),
        weight.pointer(),
        scales.pointer(),
        out.pointer(),
        raw.pointer(),
    ];
    let mut dimensions = [u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|v| (v as *mut u32).cast::<c_void>()),
    );
    // SAFETY: All five buffers have their independently checked ABI extents and live through sync.
    unsafe {
        module.function("fp8_a16_decode")?.launch(
            [u32::try_from(n.div_ceil(4))?, 1, 1],
            [128, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    compare(
        &out,
        &raw,
        &reference.output_bf16,
        &reference.unrounded_fp32,
        "F01",
        [1, n, k],
    )
}

fn prefill(ctx: &Context, module: &Module<'_>, shape: [usize; 3]) -> Result<Value> {
    let [m, n, k] = shape;
    let a = codes(m * k, 5);
    let w = codes(n * k, 7);
    let sa = (0..m).map(|i| (i % 3 + 1) as f32 / 8.0).collect::<Vec<_>>();
    let sw = (0..n)
        .map(|i| round_bf16((i % 5 + 1) as f32 / 4.0))
        .collect::<Vec<_>>();
    let reference = crate::fp8_native_prefill_reference::fp8_native_prefill_reference(
        &a, &w, &sa, &sw, m, n, k,
    )?;
    let a = upload(ctx, &a)?;
    let w = upload(ctx, &w)?;
    let sa = upload(ctx, &floats(&sa))?;
    let sw = upload(ctx, &words(&sw))?;
    let out = upload(ctx, &vec![0xa5; m * n * 2])?;
    let raw = upload(ctx, &vec![0xa5; m * n * 4])?;
    let mut pointers = [
        a.pointer(),
        w.pointer(),
        sa.pointer(),
        sw.pointer(),
        out.pointer(),
        raw.pointer(),
    ];
    let mut dimensions = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|v| (v as *mut u32).cast::<c_void>()),
    );
    // SAFETY: Buffers implement the six-pointer ABI, dimensions are bounded fixture constants,
    // and all inputs/outputs live through completion. The kernel guards partial tiles.
    unsafe {
        module.function("fp8_prefill_native")?.launch(
            [
                u32::try_from(n.div_ceil(64))?,
                u32::try_from(m.div_ceil(32))?,
                1,
            ],
            [128, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    compare(
        &out,
        &raw,
        &reference.bf16,
        &reference.unrounded,
        "F02",
        shape,
    )
}

fn compare(
    out: &Buffer<'_>,
    raw: &Buffer<'_>,
    expected: &[u16],
    oracle: &[f32],
    feature: &str,
    shape: [usize; 3],
) -> Result<Value> {
    let mut out_bytes = vec![0; out.len()];
    out.download(&mut out_bytes)?;
    let mut raw_bytes = vec![0; raw.len()];
    raw.download(&mut raw_bytes)?;
    let actual = out_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect::<Vec<_>>();
    let unrounded = raw_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect::<Vec<_>>();
    let f = |v: u16| f32::from_bits(u32::from(v) << 16) as f64;
    let differences = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f(a) - f(b)).powi(2))
        .sum::<f64>();
    let norm = expected.iter().map(|&v| f(v).powi(2)).sum::<f64>();
    let l2 = (error / norm.max(1e-30)).sqrt();
    let max_raw = unrounded
        .iter()
        .zip(oracle)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max);
    let max_oracle = oracle
        .iter()
        .map(|v| f64::from(*v).abs())
        .fold(1.0, f64::max);
    let finite =
        actual.iter().all(|&v| f(v).is_finite()) && unrounded.iter().all(|v| v.is_finite());
    // Fixed before the first GPU trial: 1% BF16 normalized L2 and 1e-4 scaled FP32 max error.
    let passed = finite && l2 <= 0.01 && max_raw / max_oracle <= 1e-4;
    Ok(
        json!({"feature":feature,"shape":shape,"all_passed":passed,"finite":finite,
        "bf16_differences":differences,"bf16_normalized_l2":l2,"fp32_max_abs_error":max_raw,
        "fp32_max_scaled_error":max_raw/max_oracle,"bf16_l2_limit":0.01,"fp32_scaled_limit":1e-4}),
    )
}

fn workspace(ctx: &Context) -> Result<Value> {
    let layout = WorkspaceLayout::fp8_projection_chain(2, 17, 5, 4096)?;
    let mut arena = ResidentWorkspace::new(ctx, layout)?;
    let bytes = arena.layout().high_water_bytes();
    let first = {
        let step = arena.begin_step()?;
        let view = step.region(crate::engine::workspace::FP8_CODES_REGION)?;
        ensure!(view.bytes() == 34, "workspace view length differs");
        let pointer = view.pointer();
        step.complete()?;
        pointer
    };
    let second = {
        let step = arena.begin_step()?;
        let pointer = step
            .region(crate::engine::workspace::FP8_CODES_REGION)?
            .pointer();
        step.complete()?;
        pointer
    };
    ensure!(
        first == second && !arena.is_poisoned(),
        "completed workspace reuse failed"
    );
    drop(arena.begin_step()?);
    ensure!(
        arena.is_poisoned() && arena.begin_step().is_err(),
        "aborted workspace allowed reuse"
    );
    Ok(json!({"stable_address":true,"aborted_step_poisoned":true,"allocation_bytes":bytes}))
}
