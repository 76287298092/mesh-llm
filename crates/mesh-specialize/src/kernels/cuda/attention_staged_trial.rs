//! Staged FP64 operator qualification. Sanitizer-safe default: short scope, timing off.
use super::{
    attention_staged_trial_execution::{Functions, execute, timing},
    attention_staged_trial_support::{
        Buffers, buffers, bytes, compare, download, snapshot, workspace_check, workspace_poison,
    },
    driver::{Context, Module},
};
use crate::{
    attention_v2_reference as oracle,
    attention_warp_reference::{Fixture, Pattern, fixture},
    kernels::attention_staged_plan::{
        CoefficientSchedule, FULL_TRIAL_PASTS, Plan, SHORT_TRIAL_PASTS, TrialOptions,
    },
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

fn rewind(
    ctx: &Context,
    functions: &Functions<'_, '_>,
    b: &Buffers<'_>,
    f: &Fixture,
    p: Plan,
) -> Result<Value> {
    let short = Plan::new_with_schedule([1, 24, 4, 256, 0, p.capacity], p.coefficient_schedule)?;
    let before = download(&b.workspace)?;
    execute(ctx, functions, b, short, false)?;
    execute(ctx, functions, b, short, true)?;
    let control = snapshot(&b.legacy)?;
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
    schedule: CoefficientSchedule,
) -> Result<Value> {
    let p = Plan::new_with_schedule([1, 24, 4, 256, f.past, f.capacity], schedule)?;
    let expected = oracle::run(&f.q, &f.k, &f.v, 1, f.past, f.capacity)?;
    let b = buffers(ctx, &f, p)?;
    let functions = Functions::new(module, schedule)?;
    let before = download(&b.workspace)?;
    execute(ctx, &functions, &b, p, false)?;
    execute(ctx, &functions, &b, p, true)?;
    let old = snapshot(&b.legacy)?;
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
        json!({"staged_v1":timing(ctx,&functions,&b,p,false)?,
        "candidate":timing(ctx,&functions,&b,p,true)?})
    } else {
        Value::Null
    };
    // Timing can overwrite outputs; recheck full output bits if it ran.
    let timing_bits = !options.timing
        || !passed
        || (compare(&snapshot(&b.legacy)?, &old)["all_passed"] == true
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
    let schedule = CoefficientSchedule::current()?;
    let mut cases = Vec::new();
    let prefix_parallel_short_pasts = [0, 1, 32, 105, 127, 128];
    let prefix_parallel_full_pasts = [0, 1, 32, 105, 127, 128, 511, 8190];
    let pasts: &[usize] = match (schedule, options.long_cases) {
        (CoefficientSchedule::PrefixParallelV2, false) => &prefix_parallel_short_pasts,
        (CoefficientSchedule::PrefixParallelV2, true) => &prefix_parallel_full_pasts,
        (CoefficientSchedule::SerialV1, false) => &SHORT_TRIAL_PASTS,
        (CoefficientSchedule::SerialV1, true) => &FULL_TRIAL_PASTS,
    };
    for &past in pasts {
        cases.push(case(
            &ctx,
            &module,
            fixture(past, Pattern::Hashed),
            Pattern::Hashed,
            options,
            schedule,
        )?);
    }
    for pattern in [
        Pattern::Uniform,
        Pattern::Cancellation,
        Pattern::SignedZero,
        Pattern::WideExponent,
    ] {
        cases.push(case(
            &ctx,
            &module,
            fixture(32, pattern),
            pattern,
            options,
            schedule,
        )?);
    }
    let functions = Functions::new(&module, schedule)?;
    let scheduled_resources = functions.report()?;
    Ok(
        json!({"kind":if schedule==CoefficientSchedule::PrefixParallelV2 { "bf16-staged-fp64-prefix-parallel-v2" } else { "bf16-staged-fp64-exact-order-v1" },"coefficient_schedule":schedule.name(),"device":info,
        "scope":if options.long_cases { "full" } else { "short" },"operator_timing_enabled":options.timing,
        "all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,
        "resources":{"control":module.function("causal_attention_bf16")?.resources()?,"scheduled":scheduled_resources},
        "qualification":"synthetic operator only; whole-model same-input/logit/state identity and model timing remain parent gates"}),
    )
}
