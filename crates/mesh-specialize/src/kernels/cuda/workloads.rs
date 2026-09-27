use super::{
    driver::{Context, Module},
    gemm, rms_norm,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "workloads must target sm_120a"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "workloads require SM120"
    );
    let before = context.memory()?;
    let module = Module::load(&context, ptx)?;
    let mut cases = rms_norm::cases(&context, &module)?;
    cases.extend(gemm::cases(&context, &module)?);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"rust-representative-kernel-trial",
        "device":info,"jit_log":module.jit_log(),
        "memory_before_module":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_after_cases":{"free_bytes":after.0,"total_bytes":after.1},
        "all_passed":cases.iter().all(|c|c["passed"]==true),"cases":cases,
        "model_prefill_tokens_per_second":null,"model_decode_tokens_per_second":null,
        "model_context_tokens":null,"library_reference":"pending"}),
    )
}
