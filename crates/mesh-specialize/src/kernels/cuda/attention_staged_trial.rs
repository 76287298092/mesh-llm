//! Staged FP64 operator qualification. Sanitizer-safe default: short scope, timing off.
use super::{
    attention_staged_launch::Kernels,
    driver::{Buffer, Context, Event, Function, Module},
};
use crate::{
    attention_v2_reference as oracle,
    attention_warp_reference::{Fixture, Pattern, QUERY_ELEMENTS, fixture},
    kernels::attention_staged_plan::{FULL_TRIAL_PASTS, Plan, SHORT_TRIAL_PASTS, TrialOptions},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD: usize = 64;
struct Outputs<'ctx> {
    bf16: Buffer<'ctx>,
    raw: Buffer<'ctx>,
}
struct Buffers<'ctx> {
    q: Buffer<'ctx>,
    k: Buffer<'ctx>,
    v: Buffer<'ctx>,
    workspace: Buffer<'ctx>,
    control: Outputs<'ctx>,
    candidate: Outputs<'ctx>,
}
struct Functions<'m, 'ctx> {
    control: Function<'m, 'ctx>,
    staged: Kernels<'m, 'ctx>,
}
struct Snapshot {
    bf16: Vec<u16>,
    raw: Vec<f32>,
    guards: bool,
}
fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn download(b: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut v = vec![0; b.len()];
    b.download(&mut v)?;
    Ok(v)
}
fn upload<'ctx>(ctx: &'ctx Context, words: &[u16]) -> Result<Buffer<'ctx>> {
    let v = bytes(words);
    let b = Buffer::new(ctx, v.len())?;
    b.upload(&v)?;
    Ok(b)
}
fn guarded(ctx: &Context, size: usize) -> Result<Buffer<'_>> {
    let b = Buffer::new(ctx, size + 2 * GUARD)?;
    b.upload(&vec![0xa5; b.len()])?;
    Ok(b)
}
fn outputs(ctx: &Context) -> Result<Outputs<'_>> {
    Ok(Outputs {
        bf16: guarded(ctx, QUERY_ELEMENTS * 2)?,
        raw: guarded(ctx, QUERY_ELEMENTS * 4)?,
    })
}
fn workspace_poison(allocation_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0xa5; allocation_bytes];
    for word in bytes[GUARD..allocation_bytes - GUARD]
        .as_chunks_mut::<8>()
        .0
    {
        word.copy_from_slice(&0x7ff8_a5a5_a5a5_a5a5_u64.to_le_bytes());
    }
    bytes
}
fn buffers<'ctx>(ctx: &'ctx Context, f: &Fixture, p: Plan) -> Result<Buffers<'ctx>> {
    let workspace = guarded(ctx, p.workspace_bytes)?;
    workspace.upload(&workspace_poison(workspace.len()))?;
    Ok(Buffers {
        q: upload(ctx, &f.q)?,
        k: upload(ctx, &f.k)?,
        v: upload(ctx, &f.v)?,
        workspace,
        control: outputs(ctx)?,
        candidate: outputs(ctx)?,
    })
}
fn guards(bytes: &[u8]) -> bool {
    bytes[..GUARD]
        .iter()
        .chain(&bytes[bytes.len() - GUARD..])
        .all(|&b| b == 0xa5)
}
fn snapshot(out: &Outputs<'_>) -> Result<Snapshot> {
    let bf16 = download(&out.bf16)?;
    let raw = download(&out.raw)?;
    Ok(Snapshot {
        guards: guards(&bf16) && guards(&raw),
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
    })
}
fn compare(actual: &Snapshot, control: &Snapshot) -> Value {
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
    let extent = actual.raw.len() == QUERY_ELEMENTS
        && control.raw.len() == QUERY_ELEMENTS
        && actual.bf16.len() == QUERY_ELEMENTS
        && control.bf16.len() == QUERY_ELEMENTS;
    json!({"all_passed":extent&&raw==0&&bf16==0&&actual.guards&&control.guards,
        "raw_fp32_bit_differences":raw,"bf16_bit_differences":bf16,"extent_match":extent,
        "candidate_guards_intact":actual.guards,"control_guards_intact":control.guards})
}

fn launch(functions: &Functions<'_, '_>, b: &Buffers<'_>, p: Plan, candidate: bool) -> Result<()> {
    if candidate {
        // SAFETY: Complete independently allocated guarded payloads, same context,
        // initialized finite Q/K/V prefix, and caller drains while owners remain live.
        return unsafe {
            functions.staged.launch(
                [
                    b.q.pointer(),
                    b.k.pointer(),
                    b.v.pointer(),
                    b.candidate.bf16.pointer() + GUARD as u64,
                    b.candidate.raw.pointer() + GUARD as u64,
                    b.workspace.pointer() + GUARD as u64,
                ],
                p,
            )
        };
    }
    let mut pointers = [
        b.q.pointer(),
        b.k.pointer(),
        b.v.pointer(),
        b.control.bf16.pointer() + GUARD as u64,
        b.control.raw.pointer() + GUARD as u64,
    ];
    let mut dims = [1_u32, 24, 4, 256, (p.length - 1) as u32, p.capacity as u32];
    let mut scale = 0.0625_f32;
    let mut args = pointers
        .iter_mut()
        .map(|x| (x as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|x| (x as *mut u32).cast::<c_void>()));
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Identical initialized inputs and guarded full outputs satisfy the
    // unchanged control ABI; lifetime extends through execute/timing's failure drain.
    unsafe {
        functions
            .control
            .launch([24, 1, 1], [256, 1, 1], 0, &mut args)
    }
}
fn execute(
    ctx: &Context,
    functions: &Functions<'_, '_>,
    b: &Buffers<'_>,
    p: Plan,
    candidate: bool,
) -> Result<()> {
    let result = launch(functions, b, p, candidate);
    let sync = ctx.synchronize();
    result?;
    sync
}
fn timing(
    ctx: &Context,
    functions: &Functions<'_, '_>,
    b: &Buffers<'_>,
    p: Plan,
    candidate: bool,
) -> Result<Value> {
    for _ in 0..3 {
        execute(ctx, functions, b, p, candidate)?;
    }
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut samples = Vec::new();
    for _ in 0..5 {
        start.record()?;
        let pending = launch(functions, b, p, candidate)
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
    let median = sorted[2];
    ensure!(
        median.is_finite() && median > 0.0,
        "invalid staged event time"
    );
    Ok(
        json!({"warmups":3,"repetitions":5,"event_ms":samples,"median_event_ms":median,
        "scope":"whole three-stage operator span or original control; never model throughput"}),
    )
}

fn workspace_check(p: Plan, current: &[u8], before: &[u8]) -> Value {
    let mut stale = 0;
    let mut invalid = 0;
    for (i, (actual, old)) in current[GUARD..current.len() - GUARD]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(before[GUARD..before.len() - GUARD].as_chunks::<8>().0)
        .enumerate()
    {
        if p.initialized(i) {
            let value = f64::from_le_bytes(*actual);
            if !value.is_finite()
                || (i >= p.head_elements
                    && i < 3 * p.head_elements
                    && !(0.0..=1.0).contains(&value))
                || (i >= 3 * p.head_elements && !(0.0..=p.length as f64).contains(&value))
            {
                invalid += 1;
            }
            if i >= 3 * p.head_elements && value == 0.0 {
                invalid += 1;
            }
        } else if actual != old {
            stale += 1;
        }
    }
    json!({"all_passed":guards(current)&&stale==0&&invalid==0,
        "guards_intact":guards(current),"uninitialized_suffix_writes":stale,"invalid_initialized_values":invalid})
}

fn rewind(
    ctx: &Context,
    functions: &Functions<'_, '_>,
    b: &Buffers<'_>,
    f: &Fixture,
    p: Plan,
) -> Result<Value> {
    let short = Plan::new([1, 24, 4, 256, 0, p.capacity])?;
    let before = download(&b.workspace)?;
    execute(ctx, functions, b, short, false)?;
    execute(ctx, functions, b, short, true)?;
    let control = snapshot(&b.control)?;
    let first = snapshot(&b.candidate)?;
    let strict = compare(&first, &control);
    let expected = oracle::run(&f.q, &f.k, &f.v, 1, 0, p.capacity)?;
    let independent = oracle::compare(&first.raw, &first.bf16, &expected)?;
    let suffix = workspace_check(short, &download(&b.workspace)?, &before);
    // Same addresses, now poison ALL old intermediates. This exposes a hidden
    // dependency on stale longer prefixes or previously initialized normalizers.
    let poison = workspace_poison(b.workspace.len());
    b.workspace.upload(&poison)?;
    execute(ctx, functions, b, short, true)?;
    let poisoned = compare(&snapshot(&b.candidate)?, &first);
    let poison_suffix = workspace_check(short, &download(&b.workspace)?, &poison);
    Ok(
        json!({"all_passed":strict["all_passed"]==true&&independent["all_passed"]==true
        &&suffix["all_passed"]==true&&poisoned["all_passed"]==true&&poison_suffix["all_passed"]==true,
        "past":0,"same_addresses":true,"future_kv_still_initialized":true,"strict_control":strict,
        "independent_oracle":independent,"long_to_short_suffix":suffix,
        "poisoned_workspace_repeat":poisoned,"poisoned_suffix":poison_suffix}),
    )
}

fn case(
    ctx: &Context,
    module: &Module<'_>,
    f: Fixture,
    pattern: Pattern,
    options: TrialOptions,
) -> Result<Value> {
    let p = Plan::new([1, 24, 4, 256, f.past, f.capacity])?;
    let expected = oracle::run(&f.q, &f.k, &f.v, 1, f.past, f.capacity)?;
    let b = buffers(ctx, &f, p)?;
    let functions = Functions {
        control: module.function("causal_attention_bf16")?,
        staged: Kernels::new(module)?,
    };
    let before = download(&b.workspace)?;
    execute(ctx, &functions, &b, p, false)?;
    execute(ctx, &functions, &b, p, true)?;
    let old = snapshot(&b.control)?;
    let first = snapshot(&b.candidate)?;
    let strict = compare(&first, &old);
    let independent = oracle::compare(&first.raw, &first.bf16, &expected)?;
    let control_oracle = oracle::compare(&old.raw, &old.bf16, &expected)?;
    let workspace = workspace_check(p, &download(&b.workspace)?, &before);
    execute(ctx, &functions, &b, p, true)?;
    let repeated = compare(&snapshot(&b.candidate)?, &first);
    let rewound = rewind(ctx, &functions, &b, &f, p)?;
    let before_restore = download(&b.workspace)?;
    execute(ctx, &functions, &b, p, true)?;
    let restored = compare(&snapshot(&b.candidate)?, &first);
    let restored_workspace = workspace_check(p, &download(&b.workspace)?, &before_restore);
    let passed = [
        &strict,
        &independent,
        &control_oracle,
        &workspace,
        &repeated,
        &rewound,
        &restored,
        &restored_workspace,
    ]
    .iter()
    .all(|v| v["all_passed"] == true);
    let times = if passed && options.timing {
        json!({"control":timing(ctx,&functions,&b,p,false)?,
        "candidate":timing(ctx,&functions,&b,p,true)?})
    } else {
        Value::Null
    };
    // Timing can overwrite outputs; recheck full output bits if it ran.
    let timing_bits = !options.timing
        || !passed
        || (compare(&snapshot(&b.control)?, &old)["all_passed"] == true
            && compare(&snapshot(&b.candidate)?, &first)["all_passed"] == true);
    let unchanged = download(&b.q)? == bytes(&f.q)
        && download(&b.k)? == bytes(&f.k)
        && download(&b.v)? == bytes(&f.v);
    Ok(
        json!({"rows":1,"past":f.past,"capacity":f.capacity,"pattern":format!("{pattern:?}"),
        "workspace_bytes":p.workspace_bytes,"strict_control":strict,"independent_oracle":independent,
        "control_oracle":control_oracle,"workspace":workspace,"repeat":repeated,"rewind":rewound,
        "restore_long_prefix":restored,"restored_workspace":restored_workspace,"inputs_and_kv_tail_unchanged":unchanged,
        "timing":times,"post_timing_bits_passed":timing_bits,"all_passed":passed&&unchanged&&timing_bits}),
    )
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow::anyhow!("invalid {name}: {error}")),
    }
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let scope = optional_env("MESH_SPECIALIZE_STAGED_TRIAL_SCOPE")?;
    let timing = optional_env("MESH_SPECIALIZE_STAGED_TRIAL_TIMING")?;
    let options = TrialOptions::parse(scope.as_deref(), timing.as_deref())?;
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "staged attention trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    let pasts = if options.long_cases {
        &FULL_TRIAL_PASTS[..]
    } else {
        &SHORT_TRIAL_PASTS[..]
    };
    for &past in pasts {
        cases.push(case(
            &ctx,
            &module,
            fixture(past, Pattern::Hashed),
            Pattern::Hashed,
            options,
        )?);
    }
    for pattern in [
        Pattern::Uniform,
        Pattern::Cancellation,
        Pattern::SignedZero,
        Pattern::WideExponent,
    ] {
        cases.push(case(&ctx, &module, fixture(32, pattern), pattern, options)?);
    }
    let functions = Kernels::new(&module)?;
    Ok(
        json!({"kind":"bf16-staged-fp64-exact-order-v1","device":info,
        "scope":if options.long_cases { "full" } else { "short" },"operator_timing_enabled":options.timing,
        "all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,
        "resources":{"control":module.function("causal_attention_bf16")?.resources()?,
            "scores":functions.scores.resources()?,"coefficients":functions.coefficients.resources()?,"values":functions.values.resources()?},
        "qualification":"synthetic operator only; whole-model same-input/logit/state identity and model timing remain parent gates"}),
    )
}
