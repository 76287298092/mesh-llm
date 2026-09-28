//! Isolated exponential-delivery attention experiment; control CTA geometry is unchanged.
use super::driver::{Buffer, Context, Event, Function, Module};
use crate::{
    attention_online_reference as oracle,
    attention_v2_reference::compare as compare_oracle,
    attention_warp_reference::{Fixture, Pattern, QUERY_ELEMENTS, fixture},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const KERNEL: &str = "causal_attention_unrolled_fp64";
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
    let buffer = Buffer::new(ctx, count + 2 * GUARD)?;
    buffer.upload(&vec![0xa5; buffer.len()])?;
    Ok(buffer)
}
fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut data = vec![0; buffer.len()];
    buffer.download(&mut data)?;
    Ok(data)
}
fn outputs(ctx: &Context, count: usize) -> Result<Outputs<'_>> {
    Ok(Outputs {
        bf16: guarded(ctx, count * 2)?,
        raw: guarded(ctx, count * 4)?,
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
    out: &Outputs<'_>,
    dims: [u32; 6],
) -> Result<()> {
    let [rows, qh, kh, width, past, capacity] = dims;
    ensure!(
        (1..=128).contains(&rows)
            && [qh, kh, width] == [24, 4, 256]
            && past < capacity
            && rows <= capacity - past
            && capacity <= 262_144,
        "invalid bounded unrolled attention trial geometry"
    );
    let count = rows as usize * QUERY_ELEMENTS;
    ensure!(
        input.q.len() == count * 2
            && input.k.len() == capacity as usize * 1024 * 2
            && input.v.len() == input.k.len()
            && out.bf16.len() == count * 2 + 2 * GUARD
            && out.raw.len() == count * 4 + 2 * GUARD,
        "unrolled trial buffer extent mismatch"
    );
    let mut pointers = [
        input.q.pointer(),
        input.k.pointer(),
        input.v.pointer(),
        out.bf16.pointer() + GUARD as u64,
        out.raw.pointer() + GUARD as u64,
    ];
    let mut dims = dims;
    let mut scale = 0.0625_f32;
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast::<c_void>()));
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Validated independent buffers and unchanged control ABI/geometry.
    // Caller drains the stream before dropping any buffer, including on failure.
    unsafe { function.launch([rows * qh, 1, 1], [256, 1, 1], 0, &mut args) }
}
fn execute(
    ctx: &Context,
    function: &Function<'_, '_>,
    input: &Inputs<'_>,
    out: &Outputs<'_>,
    dims: [u32; 6],
) -> Result<()> {
    let result = launch(function, input, out, dims);
    let sync = ctx.synchronize();
    result?;
    sync
}
fn timing(
    ctx: &Context,
    function: &Function<'_, '_>,
    input: &Inputs<'_>,
    out: &Outputs<'_>,
    dims: [u32; 6],
) -> Result<Value> {
    for _ in 0..WARMUP {
        execute(ctx, function, input, out, dims)?;
    }
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut samples = Vec::new();
    for _ in 0..REPETITIONS {
        start.record()?;
        let pending = launch(function, input, out, dims)
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
        json!({"warmups":WARMUP,"repetitions":REPETITIONS,"event_ms":samples,"median_event_ms":median}),
    )
}

fn compare(actual: &Snapshot, control: &Snapshot, count: usize) -> Value {
    let raw = actual
        .raw
        .iter()
        .zip(&control.raw)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let bf16 = actual
        .bf16
        .iter()
        .zip(&control.bf16)
        .filter(|(a, b)| a != b)
        .count();
    let extents = actual.raw.len() == count
        && control.raw.len() == count
        && actual.bf16.len() == count
        && control.bf16.len() == count;
    json!({"all_passed":raw==0 && bf16==0 && extents && actual.guards && control.guards,
        "raw_fp32_bit_differences":raw,"bf16_bit_differences":bf16,"extent_match":extents,
        "candidate_guards_intact":actual.guards,"control_guards_intact":control.guards})
}

fn row_fixture(rows: usize, past: usize, pattern: Pattern) -> Fixture {
    let mut f = fixture(past + rows - 1, pattern);
    f.past = past;
    let query = f.q;
    f.q = (0..rows * QUERY_ELEMENTS)
        .map(|i| {
            let row = i / QUERY_ELEMENTS;
            query[(i % QUERY_ELEMENTS + row * 97) % QUERY_ELEMENTS]
        })
        .collect();
    f
}

fn case(
    ctx: &Context,
    module: &Module<'_>,
    rows: usize,
    past: usize,
    pattern: Pattern,
) -> Result<Value> {
    let f = row_fixture(rows, past, pattern);
    let shape = oracle::Shape {
        rows,
        past,
        capacity: f.capacity,
        query_heads: 24,
        kv_heads: 4,
        width: 256,
        scale: 0.0625,
    };
    let expected = oracle::run(&f.q, &f.k, &f.v, &shape)?;
    let dims = [
        u32::try_from(rows)?,
        24,
        4,
        256,
        u32::try_from(past)?,
        u32::try_from(f.capacity)?,
    ];
    let input = Inputs {
        q: upload(ctx, &f.q)?,
        k: upload(ctx, &f.k)?,
        v: upload(ctx, &f.v)?,
    };
    let control = outputs(ctx, f.q.len())?;
    let candidate = outputs(ctx, f.q.len())?;
    let baseline = module.function("causal_attention_bf16")?;
    let unrolled = module.function(KERNEL)?;
    execute(ctx, &baseline, &input, &control, dims)?;
    execute(ctx, &unrolled, &input, &candidate, dims)?;
    let old = snapshot(&control)?;
    let new = snapshot(&candidate)?;
    let strict = compare(&new, &old, f.q.len());
    let independent = compare_oracle(&new.raw, &new.bf16, &expected)?;
    let control_oracle = compare_oracle(&old.raw, &old.bf16, &expected)?;
    let mut repeat_passed = true;
    for _ in 0..3 {
        execute(ctx, &baseline, &input, &control, dims)?;
        execute(ctx, &unrolled, &input, &candidate, dims)?;
        repeat_passed &= compare(&snapshot(&control)?, &old, f.q.len())["all_passed"] == true
            && compare(&snapshot(&candidate)?, &new, f.q.len())["all_passed"] == true;
    }
    let passed = strict["all_passed"] == true
        && independent["all_passed"] == true
        && control_oracle["all_passed"] == true
        && repeat_passed;
    let times = if passed {
        json!({"control":timing(ctx,&baseline,&input,&control,dims)?,
        "candidate":timing(ctx,&unrolled,&input,&candidate,dims)?})
    } else {
        Value::Null
    };
    let final_repeat = compare(&snapshot(&control)?, &old, f.q.len())["all_passed"] == true
        && compare(&snapshot(&candidate)?, &new, f.q.len())["all_passed"] == true;
    let unchanged = download(&input.q)? == bytes(&f.q)
        && download(&input.k)? == bytes(&f.k)
        && download(&input.v)? == bytes(&f.v);
    Ok(
        json!({"rows":rows,"past":past,"capacity":f.capacity,"pattern":format!("{pattern:?}"),
        "strict_control":strict,"independent_oracle":independent,"control_oracle":control_oracle,
        "repeat_bits_and_guards_passed":repeat_passed && final_repeat,
        "inputs_and_poison_tail_unchanged":unchanged,"timing":times,
        "all_passed":passed && final_repeat && unchanged}),
    )
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "unrolled FP64 trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let exponential = super::exponential_trial::run(&ctx, &module)?;
    let mut cases = Vec::new();
    // Do not benchmark attention after the prerequisite helper bit gate fails.
    if exponential["all_passed"] == true {
        for past in [0, 1, 32, 127, 512, 8191, 32767] {
            cases.push(case(&ctx, &module, 1, past, Pattern::Hashed)?);
        }
        for pattern in [
            Pattern::Uniform,
            Pattern::Cancellation,
            Pattern::SignedZero,
            Pattern::WideExponent,
        ] {
            cases.push(case(&ctx, &module, 1, 127, pattern)?);
        }
        cases.push(case(&ctx, &module, 17, 32, Pattern::Hashed)?);
        cases.push(case(&ctx, &module, 128, 0, Pattern::Hashed)?);
    }
    Ok(
        json!({"kind":"bf16-unrolled-fp64-exact-order-v1","device":info,"exponential":exponential,
        "all_passed":exponential["all_passed"]==true && !cases.is_empty() && cases.iter().all(|c| c["all_passed"]==true),
        "cases":cases,"resources":{"control":module.function("causal_attention_bf16")?.resources()?,
            "candidate":module.function(KERNEL)?.resources()?},
        "scope":"isolated exponential delivery, synthetic M=1/17/128 operator qualification only; graph and model performance unqualified"}),
    )
}
