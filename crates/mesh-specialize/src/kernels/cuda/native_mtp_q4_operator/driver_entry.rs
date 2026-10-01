use super::super::driver::{Context, Module};
use super::fixture;
use anyhow::{Result, ensure};
use serde_json::Value;

pub(in crate::kernels) fn synthetic(ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "native MTP Q4 head GEMV requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "native MTP Q4 head GEMV requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    module.function("native_mtp_q4_head_gemv")?;
    let mut report = fixture::run_synthetic(&context, &module)?;
    report["device"] = serde_json::to_value(info)?;
    report["jit_log"] = serde_json::json!(module.jit_log());
    Ok(report)
}
