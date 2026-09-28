//! Synthetic operator qualification only. Strict control bits AND independent oracle budgets.
use super::driver::{Buffer, Context, Event, Function, Module};
use crate::{
    attention_v2_reference as oracle,
    attention_warp_reference::{Fixture, Pattern, QUERY_ELEMENTS, fixture},
    kernels::attention_warp_plan as plan,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD: usize = 64;
const WARMUP: usize = 3;
const REPETITIONS: usize = 5;

struct Inputs<'ctx> {
    q: Buffer<'ctx>,
    k: Buffer<'ctx>,
    v: Buffer<'ctx>,
}
struct Outputs<'ctx> {
    bf16: Buffer<'ctx>,
    raw: Buffer<'ctx>,
}
struct Snapshot {
    bf16: Vec<u16>,
    raw: Vec<f32>,
    guards: bool,
}

fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload<'ctx>(ctx: &'ctx Context, words: &[u16]) -> Result<Buffer<'ctx>> {
    let data = bytes(words);
    let buffer = Buffer::new(ctx, data.len())?;
    buffer.upload(&data)?;
    Ok(buffer)
}
fn guarded(ctx: &Context, count: usize) -> Result<Buffer<'_>> {
    let b = Buffer::new(ctx, count + 2 * GUARD)?;
    b.upload(&vec![0xa5; count + 2 * GUARD])?;
    Ok(b)
}
fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut data = vec![0; buffer.len()];
    buffer.download(&mut data)?;
    Ok(data)
}
fn outputs(ctx: &Context) -> Result<Outputs<'_>> {
    Ok(Outputs {
        bf16: guarded(ctx, QUERY_ELEMENTS * 2)?,
        raw: guarded(ctx, QUERY_ELEMENTS * 4)?,
    })
}
fn snapshot(output: &Outputs<'_>) -> Result<Snapshot> {
    let bf16 = download(&output.bf16)?;
    let raw = download(&output.raw)?;
    let guards = [&bf16, &raw].iter().all(|b| {
        b[..GUARD]
            .iter()
            .chain(&b[b.len() - GUARD..])
            .all(|&x| x == 0xa5)
    });
    Ok(Snapshot {
        bf16: bf16[GUARD..bf16.len() - GUARD]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| u16::from_le_bytes(*x))
            .collect(),
        raw: raw[GUARD..raw.len() - GUARD]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect(),
        guards,
    })
}

fn launch(
    function: &Function<'_, '_>,
    input: &Inputs<'_>,
    output: &Outputs<'_>,
    dimensions: [u32; 6],
    warp: bool,
) -> Result<()> {
    // Both paths in this trial intentionally use only M=1 fixed model geometry.
    plan::validate(dimensions)?;
    let mut pointers = [
        input.q.pointer(),
        input.k.pointer(),
        input.v.pointer(),
        output.bf16.pointer() + GUARD as u64,
        output.raw.pointer() + GUARD as u64,
    ];
    let mut dimensions = dimensions;
    let mut scale = 0.0625_f32;
    let mut arguments = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|d| (d as *mut u32).cast::<c_void>()),
    );
    arguments.push((&mut scale as *mut f32).cast());
    let (grid, block) = if warp {
        (plan::GRID, plan::BLOCK)
    } else {
        ([24, 1, 1], [256, 1, 1])
    };
    // SAFETY: Admitted dimensions and guarded allocations match the unchanged ABI.
    // The trial drains work before any buffer drops, including launch failure paths.
    unsafe { function.launch(grid, block, 0, &mut arguments) }
}

fn execute(
    ctx: &Context,
    function: &Function<'_, '_>,
    input: &Inputs<'_>,
    output: &Outputs<'_>,
    dimensions: [u32; 6],
    warp: bool,
) -> Result<()> {
    let result = launch(function, input, output, dimensions, warp);
    let sync = ctx.synchronize();
    result?;
    sync
}

fn timing(
    ctx: &Context,
    function: &Function<'_, '_>,
    input: &Inputs<'_>,
    output: &Outputs<'_>,
    dimensions: [u32; 6],
    warp: bool,
) -> Result<Value> {
    for _ in 0..WARMUP {
        execute(ctx, function, input, output, dimensions, warp)?;
    }
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut samples = Vec::new();
    for _ in 0..REPETITIONS {
        start.record()?;
        let pending = launch(function, input, output, dimensions, warp)
            .and_then(|()| end.record())
            .and_then(|()| end.synchronize());
        if let Err(error) = pending {
            let _ = ctx.synchronize();
            return Err(error);
        }
        samples.push(end.elapsed_since(&start)?);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    ensure!(
        median.is_finite() && median > 0.0,
        "invalid attention event duration"
    );
    Ok(
        json!({"warmups":WARMUP,"repetitions":REPETITIONS,"event_ms":samples,
        "median_event_ms":median,"scope":"synthetic kernel only, not model throughput"}),
    )
}

fn compare(actual: &Snapshot, control: &Snapshot) -> Value {
    let raw_differences = actual
        .raw
        .iter()
        .zip(&control.raw)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let bf16_differences = actual
        .bf16
        .iter()
        .zip(&control.bf16)
        .filter(|(a, b)| a != b)
        .count();
    let extent_match = actual.raw.len() == QUERY_ELEMENTS
        && control.raw.len() == QUERY_ELEMENTS
        && actual.bf16.len() == QUERY_ELEMENTS
        && control.bf16.len() == QUERY_ELEMENTS;
    json!({"all_passed":extent_match && raw_differences == 0 && bf16_differences == 0
        && actual.guards && control.guards,
        "raw_fp32_bit_differences":raw_differences,"bf16_bit_differences":bf16_differences,
        "extent_match":extent_match,"candidate_guards_intact":actual.guards,
        "control_guards_intact":control.guards})
}

fn case(ctx: &Context, module: &Module<'_>, f: Fixture, pattern: Pattern) -> Result<Value> {
    let expected = oracle::run(&f.q, &f.k, &f.v, 1, f.past, f.capacity)?;
    let dimensions = [
        1,
        24,
        4,
        256,
        u32::try_from(f.past)?,
        u32::try_from(f.capacity)?,
    ];
    let input = Inputs {
        q: upload(ctx, &f.q)?,
        k: upload(ctx, &f.k)?,
        v: upload(ctx, &f.v)?,
    };
    let control = outputs(ctx)?;
    let candidate = outputs(ctx)?;
    let baseline = module.function("causal_attention_bf16")?;
    let warp = module.function(plan::KERNEL)?;
    execute(ctx, &baseline, &input, &control, dimensions, false)?;
    execute(ctx, &warp, &input, &candidate, dimensions, true)?;
    let control_snapshot = snapshot(&control)?;
    let first = snapshot(&candidate)?;
    let strict = compare(&first, &control_snapshot);
    let independent = oracle::compare(&first.raw, &first.bf16, &expected)?;
    let control_oracle = oracle::compare(&control_snapshot.raw, &control_snapshot.bf16, &expected)?;
    // No performance evidence is collected for a candidate that fails either gate.
    let passed = strict["all_passed"] == true
        && independent["all_passed"] == true
        && control_oracle["all_passed"] == true;
    let times = if passed {
        json!({"control":timing(ctx,&baseline,&input,&control,dimensions,false)?,
            "candidate":timing(ctx,&warp,&input,&candidate,dimensions,true)?})
    } else {
        Value::Null
    };
    execute(ctx, &warp, &input, &candidate, dimensions, true)?;
    let repeated = snapshot(&candidate)?;
    let repeat = compare(&repeated, &first);
    let final_control = compare(&snapshot(&control)?, &control_snapshot);
    let unchanged = download(&input.q)? == bytes(&f.q)
        && download(&input.k)? == bytes(&f.k)
        && download(&input.v)? == bytes(&f.v);
    Ok(
        json!({"rows":1,"past":f.past,"capacity":f.capacity,"pattern":format!("{pattern:?}"),
        "strict_control":strict,"independent_oracle":independent,"control_oracle":control_oracle,
        "repeat":repeat,"control_repeat":final_control,"inputs_and_poison_tail_unchanged":unchanged,
        "timing":times,"all_passed":passed && repeat["all_passed"] == true
            && final_control["all_passed"] == true && unchanged}),
    )
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "warp-fp64 attention trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for past in [0, 1, 32, 127, 512, 8191, 32767] {
        cases.push(case(
            &ctx,
            &module,
            fixture(past, Pattern::Hashed),
            Pattern::Hashed,
        )?);
    }
    for pattern in [
        Pattern::Uniform,
        Pattern::Cancellation,
        Pattern::SignedZero,
        Pattern::WideExponent,
    ] {
        cases.push(case(&ctx, &module, fixture(127, pattern), pattern)?);
    }
    Ok(json!({"kind":"bf16-warp-fp64-exact-order-v1","device":info,
        "all_passed":cases.iter().all(|c| c["all_passed"] == true),"cases":cases,
        "resources":{"control":module.function("causal_attention_bf16")?.resources()?,
            "candidate":module.function(plan::KERNEL)?.resources()?},
        "scope":"M=1 synthetic kernel qualification only; graph, whole-model same-input/state identity and model timing unqualified"}))
}
