//! Explicit bounded PRMT checks, independent of the legacy whole-model CPU reference.
use super::{
    driver::{Context, Module},
    nvfp4_decode_prmt_trial,
    resident_weights::ResidentWeights,
};
use crate::{artifact::model_source::ModelArtifact, packages::qwen3_8_27b::schedule};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

const MLP_PREFIX: &str = "tensors/model.language_model.layers.0.mlp.";
const MAX_RESIDENT_BYTES: u64 = 160 * 1024 * 1024;
const TRIAL_HEADROOM_BYTES: u64 = 128 * 1024 * 1024;

fn context(ptx: &str, device: i32) -> Result<Context> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "PRMT check requires SM120a PTX"
    );
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "PRMT check requires SM120"
    );
    Ok(ctx)
}

pub(in crate::kernels) fn synthetic(ptx: &str, device: i32) -> Result<Value> {
    let ctx = context(ptx, device)?;
    let module = Module::load(&ctx, ptx)?;
    let mut report = nvfp4_decode_prmt_trial::run(&ctx, &module)?;
    report["schema_version"] = json!(1);
    report["kind"] = json!("nvfp4-prmt-synthetic-check");
    report["device"] = json!(ctx.info());
    report["scope"] =
        json!("synthetic operator correctness; no artifact or whole-model CPU reference used");
    Ok(report)
}

/// Open either supported source through the existing verified reader. Only the
/// layer-zero text MLP tensor subset is resident. No conversion or full-model
/// CPU reference is constructed; native NInfer sources are supported directly.
pub(in crate::kernels) fn real(artifact_path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = ModelArtifact::open(artifact_path)?;
    let objects = schedule::text_objects(artifact.directory())?
        .into_iter()
        .filter(|object| object.name.starts_with(MLP_PREFIX))
        .collect::<Vec<_>>();
    let resident_bytes = objects
        .iter()
        .try_fold(0_u64, |sum, object| sum.checked_add(object.length))
        .context("layer-zero MLP byte count overflow")?;
    ensure!(
        !objects.is_empty() && objects.len() <= 32 && resident_bytes <= MAX_RESIDENT_BYTES,
        "layer-zero MLP subset exceeds bounded PRMT admission"
    );
    let ctx = context(ptx, device)?;
    let module = Module::load(&ctx, ptx)?;
    let resources = json!({
        "control":module.function("nvfp4_decode_exact")?.resources()?,
        "candidate":module.function("nvfp4_decode_exact_prmt")?.resources()?,
    });
    let before = ctx.memory()?;
    ensure!(
        u64::try_from(before.0)? >= resident_bytes + TRIAL_HEADROOM_BYTES,
        "insufficient free memory for bounded PRMT real-weight check"
    );
    let weights = ResidentWeights::load(&ctx, &mut artifact, &objects)?;
    let verification = weights.verify()?;
    let verified = verification.iter().all(|row| row["matches"] == true);
    let mut report = if verified {
        nvfp4_decode_prmt_trial::run_real(&ctx, &module, &weights)?
    } else {
        json!({"all_passed":false,"error":"resident tensor readback digest mismatch"})
    };
    drop(weights);
    ctx.synchronize()?;
    let after = ctx.memory()?;
    report["schema_version"] = json!(1);
    report["kind"] = json!("nvfp4-prmt-real-weight-check");
    report["device"] = json!(ctx.info());
    report["model_identity"] = json!(artifact.identity());
    report["artifact_path"] = json!(artifact_path);
    report["source_verification"] = artifact.verification_report();
    report["weight_verification"] = json!(verification);
    report["resources"] = resources;
    report["resident_subset_bytes"] = json!(resident_bytes);
    report["resident_subset_prefix"] = json!(MLP_PREFIX);
    report["memory_before"] = json!({"free_bytes":before.0,"total_bytes":before.1});
    report["memory_after_free"] = json!({"free_bytes":after.0,"total_bytes":after.1});
    report["scope"] = json!(
        "verified current artifact layer-zero MLP weights and synthetic activations; no whole-model reference, execution, or throughput claim"
    );
    Ok(report)
}
