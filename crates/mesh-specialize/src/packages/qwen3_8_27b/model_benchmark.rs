//! Bounded fixed-length raw-token performance experiment for the resident decoder.
use crate::{artifact::model_source::ModelArtifact, kernels::ModelBenchRequest};
use anyhow::{Context, Result, ensure};
use std::path::Path;

pub fn run(
    path: &Path,
    ptx: &str,
    device: i32,
    request: &ModelBenchRequest<'_>,
) -> Result<serde_json::Value> {
    ensure!(
        (1..=512).contains(&request.tokens.len())
            && (2..=512).contains(&request.output_tokens)
            && (1..=3).contains(&request.repetitions),
        "benchmark request exceeds bounds"
    );
    let capacity = request
        .tokens
        .len()
        .checked_add(request.output_tokens - 1)
        .context("benchmark capacity overflow")?;
    let mut artifact = ModelArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let config = super::decoder::config(capacity)?;
    let mut report =
        crate::kernels::model_benchmark(ptx, device, &mut artifact, &objects, &config, request)?;
    report["identity"] = serde_json::json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    Ok(report)
}

/// Real-weight isolated workspace experiment, separate from model throughput.
pub fn mlp_workspace(path: &Path, ptx: &str, device: i32) -> Result<serde_json::Value> {
    let mut artifact = ModelArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let cases = [(0, false), (56, true)].map(|(layer, fp8)| crate::kernels::MlpWorkspaceCase {
        prefix: format!("tensors/model.language_model.layers.{layer}.mlp"),
        width: 5120,
        channels: 17408,
        fp8,
    });
    let objects = super::schedule::text_objects(artifact.directory())?
        .into_iter()
        .filter(|o| {
            cases
                .iter()
                .any(|c| o.name.starts_with(&format!("{}.", c.prefix)))
        })
        .collect::<Vec<_>>();
    let mut report =
        crate::kernels::mlp_workspace_trial(ptx, device, &mut artifact, &objects, &cases)?;
    report["identity"] = serde_json::json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    Ok(report)
}
