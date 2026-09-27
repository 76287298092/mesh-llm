use crate::command::{DynResult, print_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::nvfp4_probe)
}

pub(super) fn instructions(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::instruction_probe)
}

pub(super) fn workloads(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::workload_probe)
}

pub(super) fn workload_check(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::workload_check)
}

fn run_probe<E: std::fmt::Display>(
    args: &[String],
    probe: fn(&str, i32) -> Result<serde_json::Value, E>,
) -> DynResult<()> {
    let [ptx_flag, ptx_path, device_flag, device, output_flag, output] = args else {
        return Err(
            "usage: xtask specialize <nvfp4-probe|instruction-probe|workload-probe|workload-check> --ptx PATH --device ORDINAL --output NEW_FILE"
                .into(),
        );
    };
    if ptx_flag != "--ptx" || device_flag != "--device" || output_flag != "--output" {
        return Err("expected --ptx, --device, and --output in that order".into());
    }
    let ptx = fs::read_to_string(ptx_path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let result = probe(&ptx, device.parse()?);
    let (mut report, error) = match result {
        Ok(report) => (report, None),
        Err(error) => (
            json!({"all_passed": false, "error": error.to_string()}),
            Some(error.to_string()),
        ),
    };
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    report["ptx_path"] = json!(ptx_path);
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output": output, "all_passed": report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err(error
            .unwrap_or_else(|| "instruction qualification failed; inspect the saved report".into())
            .into());
    }
    Ok(())
}

pub(super) fn feature_projection(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::feature_projection_trial)
}

pub(super) fn feature_attention(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::feature_attention_trial)
}

pub(super) fn feature_graph(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::feature_graph_trial)
}

pub(super) fn feature_gdn_replay(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::feature_gdn_replay_trial)
}
