//! Bounded synthetic qualification and CUDA-event timing; never model throughput.
use super::{
    attention_v2_launch::{self, Inputs},
    driver::{Buffer, Context, Event, Function, Module},
};
use crate::{
    attention_v2_reference as oracle, entry_reference::round_bf16, kernels::attention_v2_plan::Plan,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

const GUARD: usize = 64;
const WARMUP: usize = 3;
const REPETITIONS: usize = 5;

struct Fixture {
    rows: usize,
    past: usize,
    capacity: usize,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// Index-hashed inputs avoid repeated row patterns; every head/channel is distinct.
fn value(index: usize, seed: u32, magnitude: f32) -> u16 {
    let mut x = (index as u32).wrapping_add(seed);
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    round_bf16(((x % 1025) as f32 - 512.0) * magnitude / 512.0)
}

fn fixture(rows: usize, past: usize, magnitude: f32, uniform: bool) -> Fixture {
    let capacity = past + rows + 19;
    let q = (0..rows * 24 * 256)
        .map(|i| if uniform { 0 } else { value(i, 17, magnitude) })
        .collect();
    let mut k = vec![0x7fc0; capacity * 4 * 256];
    let mut v = k.clone();
    for i in 0..(past + rows) * 4 * 256 {
        k[i] = value(i, 311, magnitude);
        v[i] = value(i, 971, 1.0);
    }
    Fixture {
        rows,
        past,
        capacity,
        q,
        k,
        v,
    }
}

fn words_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload<'a>(ctx: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    let bytes = words_bytes(words);
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(&bytes)?;
    Ok(buffer)
}
fn scratch(ctx: &Context, bytes: usize) -> Result<Buffer<'_>> {
    let buffer = Buffer::new(ctx, bytes + GUARD)?;
    buffer.upload(&vec![0xa5; bytes + GUARD])?;
    Ok(buffer)
}
fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}
fn guarded_read(buffer: &Buffer<'_>) -> Result<(Vec<u8>, bool)> {
    let mut bytes = download(buffer)?;
    let guard = bytes.split_off(bytes.len() - GUARD);
    Ok((bytes, guard.iter().all(|&x| x == 0xa5)))
}

fn check(
    buffers: &Inputs<'_, '_>,
    expected: &crate::attention_online_reference::Attention,
) -> Result<Value> {
    let (raw, raw_guard) = guarded_read(buffers.raw)?;
    let (bits, output_guard) = guarded_read(buffers.output)?;
    let (_, workspace_guard) = guarded_read(buffers.workspace)?;
    let raw = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let bits = bits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let mut result = oracle::compare(&raw, &bits, expected)?;
    let guards = raw_guard && output_guard && workspace_guard;
    result["guards_intact"] = json!(guards);
    result["all_passed"] = json!(guards && result["all_passed"] == true);
    Ok(result)
}

fn execute(
    partial: &Function<'_, '_>,
    reduce: &Function<'_, '_>,
    buffers: &Inputs<'_, '_>,
    plan: Plan,
    position: Option<&Buffer<'_>>,
) -> Result<()> {
    let result = attention_v2_launch::launch(partial, reduce, buffers, plan, position);
    // Complete or surface queued work while all allocations are still live, including on error.
    let sync = buffers.q.context().synchronize();
    result?;
    sync
}

fn timing(
    ctx: &Context,
    partial: &Function<'_, '_>,
    reduce: &Function<'_, '_>,
    buffers: &Inputs<'_, '_>,
    plan: Plan,
) -> Result<Value> {
    for _ in 0..WARMUP {
        execute(partial, reduce, buffers, plan, None)?;
    }
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut samples = Vec::new();
    for _ in 0..REPETITIONS {
        start.record()?;
        let result = attention_v2_launch::launch(partial, reduce, buffers, plan, None);
        if let Err(error) = result {
            let _ = ctx.synchronize();
            return Err(error);
        }
        end.record()?;
        end.synchronize()?;
        samples.push(end.elapsed_since(&start)?);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    ensure!(
        median.is_finite() && median > 0.0,
        "invalid CUDA event duration"
    );
    // Each CTA stages each initialized K/V row once across six heads and all M.
    // Bytes exclude Q, workspace, output, and cache effects; pair time includes reduction.
    let kv_bytes = (plan.past + plan.rows) * 4 * 256 * 2 * 2;
    let gbps = kv_bytes as f64 / (f64::from(median) * 1.0e6);
    Ok(
        json!({"warmups":WARMUP,"repetitions":REPETITIONS,"pair_event_ms":samples,
        "median_pair_event_ms":median,"logical_kv_bytes":kv_bytes,"logical_kv_gb_s":gbps,
        "bandwidth_scope":"logical K+V bytes / partial-plus-reduce event time; not measured DRAM traffic, not model throughput"}),
    )
}

fn case(ctx: &Context, module: &Module<'_>, f: Fixture) -> Result<Value> {
    let plan = Plan::new(f.rows, f.past, f.capacity, f.capacity)?;
    let expected = oracle::run(&f.q, &f.k, &f.v, f.rows, f.past, f.capacity)?;
    let q = upload(ctx, &f.q)?;
    let k = upload(ctx, &f.k)?;
    let v = upload(ctx, &f.v)?;
    let workspace = scratch(ctx, plan.workspace_bytes)?;
    let output = scratch(ctx, f.q.len() * 2)?;
    let raw = scratch(ctx, f.q.len() * 4)?;
    let buffers = Inputs {
        q: &q,
        k: &k,
        v: &v,
        workspace: &workspace,
        output: &output,
        raw: &raw,
    };
    let partial = module.function(if f.rows == 1 {
        "attention_split_decode_bf16"
    } else {
        "attention_split_bf16"
    })?;
    let device_partial = module.function(if f.rows == 1 {
        "attention_split_decode_bf16_position"
    } else {
        "attention_split_bf16_position"
    })?;
    let reduce = module.function("attention_split_reduce_bf16")?;
    execute(&partial, &reduce, &buffers, plan, None)?;
    let fixed = check(&buffers, &expected)?;
    let position = Buffer::new(ctx, 4)?;
    position.upload(&(f.past as u32).to_le_bytes())?;
    execute(&device_partial, &reduce, &buffers, plan, Some(&position))?;
    let device_position = check(&buffers, &expected)?;
    let times = timing(ctx, &partial, &reduce, &buffers, plan)?;
    // Reuse exactly the same pointers and grids after a large-to-small rewind.
    // Future cache tokens remain initialized, so causal exclusion is tested too.
    position.upload(&0_u32.to_le_bytes())?;
    execute(&device_partial, &reduce, &buffers, plan, Some(&position))?;
    let rewind_expected = oracle::run(&f.q, &f.k, &f.v, f.rows, 0, f.capacity)?;
    let rewind = check(&buffers, &rewind_expected)?;
    let unchanged = download(&q)? == words_bytes(&f.q)
        && download(&k)? == words_bytes(&f.k)
        && download(&v)? == words_bytes(&f.v);
    Ok(
        json!({"rows":f.rows,"past":f.past,"capacity":f.capacity,"split_slots":plan.split_slots,
        "workspace_bytes":plan.workspace_bytes,"fixed":fixed,"device_position":device_position,
        "rewind_same_addresses":rewind,"inputs_unchanged":unchanged,"timing":times,
        "all_passed":fixed["all_passed"]==true && device_position["all_passed"]==true && rewind["all_passed"]==true && unchanged}),
    )
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "split attention trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for (rows, past, magnitude, uniform) in [
        (1, 0, 1.0, false),
        (1, 1, 1.0, false),
        (1, 513, 1.0, false),
        (1, 8191, 1.0, false),
        (1, 32767, 1.0, false),
        (5, 8190, 1.0, false),
        (2, 14, 1.0, false),
        (3, 15, 1.0, false),
        (4, 16, 1.0, false),
        (6, 62, 1.0, false),
        (7, 63, 1.0, false),
        (8, 61, 1.0, true),
        (5, 67, 16.0, false),
    ] {
        cases.push(case(
            &ctx,
            &module,
            fixture(rows, past, magnitude, uniform),
        )?);
    }
    Ok(
        json!({"kind":"experimental-bf16-split-attention-v2","device":info,
        "all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,
        "resources":{"decode":module.function("attention_split_decode_bf16")?.resources()?,
            "decode_position":module.function("attention_split_decode_bf16_position")?.resources()?,"partial":module.function("attention_split_bf16")?.resources()?,
            "position":module.function("attention_split_bf16_position")?.resources()?,
            "reduce":module.function("attention_split_reduce_bf16")?.resources()?},
        "scope":"synthetic operator correctness and timing only; graph replay, model quality and production integration unqualified"}),
    )
}
