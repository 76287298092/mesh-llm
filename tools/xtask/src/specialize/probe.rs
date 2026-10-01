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

pub(super) fn run_probe<E: std::fmt::Display>(
    args: &[String],
    probe: impl FnOnce(&str, i32) -> Result<serde_json::Value, E>,
) -> DynResult<()> {
    let [ptx_flag, ptx_path, device_flag, device, output_flag, output] = args else {
        return Err(
            "usage: xtask specialize <nvfp4-probe|instruction-probe|workload-probe|workload-check|native-parameter-check|attention-warp-check|attention-staged-check|fp8-quantize-check|nvfp4-prmt-check|nvfp4-a16-swiglu-check|nvfp4-a16-swiglu-real-check> --ptx PATH --device ORDINAL --output NEW_FILE"
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

pub(super) fn fp8_exact(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::fp8_exact_trial)
}

pub(super) fn fp8_quantize(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::fp8_quantize_trial)
}

pub(super) fn feature_fusion(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::feature_fusion_trial)
}

pub(super) fn greedy(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::greedy_trial)
}

pub(super) fn a16_head(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::a16_head_trial)
}

pub(super) fn nvfp4_pipeline(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::nvfp4_pipeline_trial)
}

pub(super) fn nvfp4_a16_swiglu(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::nvfp4_swiglu_a16_trial)
}

pub(super) fn native_mtp_q8_gemv(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::native_mtp_q8_gemv_trial)
}

pub(super) fn native_mtp_q8_fc(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::native_mtp_q8_fc_trial)
}

pub(super) fn native_mtp_q4_head(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::native_mtp_q4_head_trial)
}

pub(super) fn bf16_ab_decode(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::bf16_ab_decode_trial)
}

pub(super) fn attention_staged(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::attention_staged_trial)
}

pub(super) fn attention_unrolled(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::attention_unrolled_trial)
}

pub(super) fn exponential_unrolled(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::exponential_unrolled_trial)
}

pub(super) fn attention_warp(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::attention_warp_trial)
}

pub(super) fn nvfp4_a16_swiglu_real(args: &[String]) -> DynResult<()> {
    let [artifact_flag, artifact, rest @ ..] = args else {
        return Err("usage: xtask specialize nvfp4-a16-swiglu-real-check --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if artifact_flag != "--artifact" {
        return Err(
            "nvfp4-a16-swiglu-real-check requires --artifact first, then --ptx, --device, --output"
                .into(),
        );
    }
    run_probe(rest, |ptx, device| {
        mesh_specialize::kernels::nvfp4_swiglu_a16_real_trial(
            std::path::Path::new(artifact),
            ptx,
            device,
        )
    })
}

pub(super) fn attention_v2(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::attention_v2_trial)
}

pub(super) fn native_parameter(args: &[String]) -> DynResult<()> {
    run_probe(args, mesh_specialize::kernels::native_parameter_trial)
}
