use crate::command::{DynResult, print_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    run_trial(args, mesh_specialize::packages::qwen3_8_27b::trial)
}

pub(super) fn resident_gdn(args: &[String]) -> DynResult<()> {
    run_trial(
        args,
        mesh_specialize::packages::qwen3_8_27b::resident_gdn::trial,
    )
}

pub(super) fn residency(args: &[String]) -> DynResult<()> {
    run_trial(
        args,
        mesh_specialize::packages::qwen3_8_27b::residency::trial,
    )
}

pub(super) fn fp8_mlp(args: &[String]) -> DynResult<()> {
    run_trial(args, mesh_specialize::packages::qwen3_8_27b::fp8_mlp::trial)
}

pub(super) fn projections(args: &[String]) -> DynResult<()> {
    run_trial(
        args,
        mesh_specialize::packages::qwen3_8_27b::projections::trial,
    )
}

pub(super) fn attention(args: &[String]) -> DynResult<()> {
    run_trial(
        args,
        mesh_specialize::packages::qwen3_8_27b::attention::trial,
    )
}

fn run_trial<E: std::fmt::Display>(
    args: &[String],
    trial: fn(&Path, &str, i32) -> Result<serde_json::Value, E>,
) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        ptx_flag,
        ptx,
        device_flag,
        device,
        output_flag,
        output,
    ] = args
    else {
        return Err("usage: xtask specialize <qwen-entry-check|qwen-projection-check|qwen-attention-check|qwen-residency-check|qwen-fp8-mlp-check|qwen-resident-gdn-check> --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if artifact_flag != "--artifact"
        || ptx_flag != "--ptx"
        || device_flag != "--device"
        || output_flag != "--output"
    {
        return Err("expected --artifact, --ptx, --device, --output in order".into());
    }
    let device = device.parse()?;
    let ptx_bytes = fs::read_to_string(ptx)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let started = Instant::now();
    let result = trial(Path::new(artifact), &ptx_bytes, device);
    let mut report = match result {
        Ok(report) => report,
        Err(error) => {
            json!({"all_passed":false,"error":format!("{error:#}"),"model_executable":false})
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx_bytes.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"all_passed":report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err("Qwen GPU check failed; inspect saved report".into());
    }
    Ok(())
}
