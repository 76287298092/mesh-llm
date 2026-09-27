use super::{
    driver::{Buffer, Context, Function, Module},
    upload_words,
};
use crate::kernels::{memory_fixtures, nvfp4_layout, ordinary_fixtures};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn run(ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "instruction probes must target sm_120a"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "instruction probes require SM120"
    );
    let module = Module::load_with_register_limit(&context, ptx, Some(64))?;
    let mut cases = shared_cases(&context, &module)?;
    cases.extend(mma_cases(&context, &module)?);
    cases.push(register_budget_case(&context, &module)?);
    Ok(
        json!({"schema_version":1, "kind":"rust-instruction-qualification",
        "device": info, "jit_register_limit":64, "jit_log":module.jit_log(),
        "all_passed": cases.iter().all(|case| case["passed"] == true), "cases":cases}),
    )
}

fn shared_cases(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let function = module.function("probe_shared_load")?;
    let resources = function.resources()?;
    let mut cases = Vec::new();
    for seed in [1393, 5090] {
        let input = memory_fixtures::input(seed);
        let device_input = upload_words(context, &input)?;
        let output = upload_words(context, &[0xffff_ffff; 128])?;
        for mut mode in 0..8_u32 {
            output.upload(&[0xff; 512])?;
            let expected = memory_fixtures::expected(&input, mode).map_err(anyhow::Error::msg)?;
            let mut a = device_input.pointer();
            let mut d = output.pointer();
            let mut args = [
                (&mut a as *mut u64).cast(),
                (&mut d as *mut u64).cast(),
                (&mut mode as *mut u32).cast(),
            ];
            launch(context, &function, 32, &mut args)?;
            let actual = download_words(&output)?;
            cases.push(json!({"kernel":"probe_shared_load", "seed":seed,"mode":mode,
                "passed":actual==expected,"actual":actual,"expected":expected,"resources":resources}));
        }
    }
    Ok(cases)
}

fn mma_cases(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut cases = Vec::new();
    for case in ordinary_fixtures::fixtures().map_err(anyhow::Error::msg)? {
        let function = module.function(case.kernel)?;
        let a = upload_words(context, &case.a)?;
        let b = upload_words(context, &case.b)?;
        let output = upload_words(context, &[0xffff_ffff; 128])?;
        let mut pointers = [a.pointer(), b.pointer(), output.pointer()];
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|x| (x as *mut u64).cast())
            .collect();
        launch(context, &function, 32, &mut args)?;
        let raw = download_words(&output)?;
        let packed: Vec<_> = raw
            .into_iter()
            .map(|word| {
                if case.integer_output {
                    (word as i32) as f32
                } else {
                    f32::from_bits(word)
                }
            })
            .collect();
        let actual = nvfp4_layout::unpack_output(&packed).map_err(anyhow::Error::msg)?;
        cases.push(
            json!({"kernel":case.kernel,"fixture":case.name,"passed":actual==case.expected,
            "actual":actual,"expected":case.expected,"resources":function.resources()?}),
        );
    }
    Ok(cases)
}

fn register_budget_case(context: &Context, module: &Module<'_>) -> Result<Value> {
    let function = module.function("probe_register_budget")?;
    let resources = function.resources()?;
    // The release/inc pair is unsafe unless the initial allocation can fund the
    // reacquisition. Refuse launch if a compiler treats the JIT limit only as a cap.
    if resources.registers < 64 {
        return Ok(
            json!({"kernel":"probe_register_budget","passed":false,"launched":false,
            "resources":resources,"reason":"JIT did not reserve 64 initial registers per thread"}),
        );
    }
    let input: Vec<_> = (0..128_u32).map(|i| i.wrapping_mul(0x10203)).collect();
    let expected: Vec<_> = input.iter().map(|i| i ^ 0x1393_5090).collect();
    let a = upload_words(context, &input)?;
    let output = upload_words(context, &[0; 128])?;
    let mut ap = a.pointer();
    let mut dp = output.pointer();
    let mut args = [(&mut ap as *mut u64).cast(), (&mut dp as *mut u64).cast()];
    launch(context, &function, 128, &mut args)?;
    let actual = download_words(&output)?;
    Ok(
        json!({"kernel":"probe_register_budget","passed":actual==expected,"launched":true,
        "resources":resources,"actual":actual,"expected":expected}),
    )
}

fn launch(
    context: &Context,
    function: &Function<'_, '_>,
    threads: u32,
    args: &mut [*mut c_void],
) -> Result<()> {
    // SAFETY: private call sites supply exact signatures, sized live allocations,
    // and one full warp or warpgroup. Synchronize before those allocations drop.
    unsafe {
        function.launch([1, 1, 1], [threads, 1, 1], 0, args)?;
    }
    context.synchronize()
}

fn download_words(buffer: &Buffer<'_>) -> Result<Vec<u32>> {
    let mut bytes = vec![0; 512];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_ne_bytes(*b))
        .collect())
}
