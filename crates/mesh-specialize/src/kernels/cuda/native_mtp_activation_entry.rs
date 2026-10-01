use super::{
    driver::{Context, Module},
    native_mtp_activation_trial,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(ptx: &str, device: i32) -> Result<Value> {
    ensure!(device >= 0, "device ordinal must be nonnegative");
    ensure!(
        ptx.contains(".target sm_120a") && !ptx.contains('\0'),
        "activation trial requires SM120a PTX without NUL bytes"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "activation trial requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let mut report = native_mtp_activation_trial::run(&context, &module)?;
    report["device"] = json!(info);
    Ok(report)
}
