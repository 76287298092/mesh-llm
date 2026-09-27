//! Synthetic GPU qualification of the independent online-attention candidate.
use super::driver::{Buffer, Context, Module};
use crate::{attention_online_reference as oracle, entry_reference::round_bf16};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "attention probe requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for (rows, past, capacity, magnitude) in [
        (1, 0, 1, 1.0),
        (3, 0, 5, 1.0),
        (3, 9, 19, 1.0),
        (2, 31, 35, 16.0),
    ] {
        cases.push(case(&ctx, &module, rows, past, capacity, magnitude)?);
    }
    Ok(
        json!({"kind":"experimental-online-attention-check","device":info,
        "all_passed":cases.iter().all(|case|case["all_passed"]==true),"cases":cases,
        "resources":module.function("attention_online_bf16")?.resources()?,
        "scope":"synthetic correctness only; no model or performance qualification"}),
    )
}
fn upload<'a>(ctx: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    let bytes = words
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let out = Buffer::new(ctx, bytes.len())?;
    out.upload(&bytes)?;
    Ok(out)
}
fn case(
    ctx: &Context,
    module: &Module<'_>,
    rows: usize,
    past: usize,
    capacity: usize,
    magnitude: f32,
) -> Result<Value> {
    let shape = oracle::Shape {
        rows,
        query_heads: 4,
        kv_heads: 2,
        width: 256,
        past,
        capacity,
        scale: 0.0625,
    };
    let q = (0..rows * 4 * 256)
        .map(|i| round_bf16(((i * 7 % 31) as f32 - 15.0) * magnitude / 16.0))
        .collect::<Vec<_>>();
    let mut k = vec![0x7fc0; capacity * 2 * 256];
    let mut v = k.clone();
    for i in 0..(past + rows) * 2 * 256 {
        k[i] = round_bf16(((i * 11 % 37) as f32 - 18.0) * magnitude / 16.0);
        v[i] = round_bf16(((i * 13 % 41) as f32 - 20.0) / 16.0);
    }
    let expected = oracle::run(&q, &k, &v, &shape)?;
    let q = upload(ctx, &q)?;
    let k = upload(ctx, &k)?;
    let v = upload(ctx, &v)?;
    let output = upload(ctx, &vec![0xa5a5; expected.output.len()])?;
    let raw = upload(ctx, &vec![0xa5a5; expected.output.len() * 2])?;
    let mut pointers = [
        q.pointer(),
        k.pointer(),
        v.pointer(),
        output.pointer(),
        raw.pointer(),
    ];
    let mut dimensions = [
        u32::try_from(rows)?,
        4,
        2,
        256,
        u32::try_from(past)?,
        u32::try_from(capacity)?,
    ];
    let mut scale = shape.scale;
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|p| (p as *mut u32).cast::<c_void>()),
    );
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Fixture buffers have the exact documented BF16/FP32 extents, same context,
    // full initialized visible prefix, and remain live through context synchronization.
    unsafe {
        module.function("attention_online_bf16")?.launch(
            [u32::try_from(rows * 4)?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    let mut bytes = vec![0; raw.len()];
    raw.download(&mut bytes)?;
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let mut bytes = vec![0; output.len()];
    output.download(&mut bytes)?;
    let bf16 = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let mut failures = 0;
    let mut max_error = 0.0_f32;
    for (i, &actual) in values.iter().enumerate() {
        let error = (actual - expected.unrounded[i]).abs();
        let budget =
            oracle::COMPONENT_ABS_BUDGET + oracle::COMPONENT_REL_BUDGET * expected.value_bounds[i];
        if !actual.is_finite() || error > budget {
            failures += 1;
        }
        max_error = max_error.max(error);
    }
    let finite = bf16
        .iter()
        .all(|&v| f32::from_bits(u32::from(v) << 16).is_finite());
    Ok(
        json!({"rows":rows,"past":past,"capacity":capacity,"magnitude":magnitude,
        "all_passed":failures==0&&finite,"raw_budget_failures":failures,"max_abs_error":max_error,
        "bf16_differences":bf16.iter().zip(&expected.output).filter(|(a,b)|a!=b).count(),
        "abs_budget":oracle::COMPONENT_ABS_BUDGET,"value_relative_budget":oracle::COMPONENT_REL_BUDGET}),
    )
}
