use super::super::driver::{Context, Module};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

const MAX_OPERATOR_BYTES: u64 = 160 * 1024 * 1024;
const PREFIX: &str = "tensors/model.language_model.layers.0.mlp";

pub(in crate::kernels) fn synthetic(ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "NVFP4 A16 SwiGLU requires SM120a PTX"
    );
    let context = Context::new(device)?;
    ensure!(
        (context.info().major, context.info().minor) == (12, 0),
        "NVFP4 A16 SwiGLU requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    super::fixture::run_synthetic(&context, &module)
}

pub(in crate::kernels) fn real_from_artifact(
    artifact_path: &Path,
    ptx: &str,
    device: i32,
) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "NVFP4 A16 SwiGLU requires SM120a PTX"
    );
    let mut artifact = crate::artifact::model_source::ModelArtifact::open(artifact_path)?;
    let objects = select_gate_up(artifact.directory())?;
    let resident_bytes = objects.iter().try_fold(0_u64, |sum, object| {
        sum.checked_add(object.length)
            .context("real gate/up object byte count overflows u64")
    })?;
    ensure!(
        resident_bytes <= MAX_OPERATOR_BYTES,
        "real operator weights exceed the memory bound"
    );
    let context = Context::new(device)?;
    ensure!(
        (context.info().major, context.info().minor) == (12, 0),
        "NVFP4 A16 SwiGLU requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    module.function("nvfp4_swiglu_a16")?;
    let weights =
        super::super::resident_weights::ResidentWeights::load(&context, &mut artifact, &objects)?;
    let verification = weights.verify()?;
    ensure!(
        verification.iter().all(|entry| entry["matches"] == true),
        "resident gate/up digest mismatch"
    );
    let operator = super::real::run(&context, &module, &weights)?;
    Ok(json!({
        "kind": "nvfp4-a16-swiglu-real-weight-operator-check-v1",
        "device": context.info(),
        "artifact_path": artifact_path,
        "model_identity": artifact.identity(),
        "source_verification": artifact.verification_report(),
        "resident_weight_verification": verification,
        "resident_weight_bytes": resident_bytes,
        "operator": operator,
        "all_passed": operator["all_passed"] == true,
        "scope": "verified layer-zero gate/up tensors and deterministic BF16 input; not model quality or timing",
    }))
}

fn select_gate_up(
    directory: &crate::artifact::schema::Directory,
) -> Result<Vec<crate::artifact::schema::Object>> {
    let objects = directory
        .objects
        .iter()
        .filter(|object| {
            ["gate_proj", "up_proj"].into_iter().any(|projection| {
                [
                    "weight_packed",
                    "weight_scale",
                    "input_global_scale",
                    "weight_global_scale",
                ]
                .into_iter()
                .any(|suffix| object.name == format!("{PREFIX}.{projection}.{suffix}"))
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        objects.len() == 8,
        "expected eight layer-zero gate/up tensors"
    );
    Ok(objects)
}
