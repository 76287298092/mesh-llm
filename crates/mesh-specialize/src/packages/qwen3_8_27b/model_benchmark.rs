//! Bounded fixed-length raw-token performance experiment for the resident decoder.
use crate::{artifact::reader::VerifiedArtifact, kernels::ModelBenchRequest};
use anyhow::{Context, Result, ensure};
use std::path::Path;

pub fn run(
    path: &Path,
    ptx: &str,
    device: i32,
    request: &ModelBenchRequest<'_>,
) -> Result<serde_json::Value> {
    ensure!(
        (1..=128).contains(&request.tokens.len())
            && (2..=16).contains(&request.output_tokens)
            && (1..=3).contains(&request.repetitions),
        "benchmark request exceeds bounds"
    );
    let capacity = request
        .tokens
        .len()
        .checked_add(request.output_tokens - 1)
        .context("benchmark capacity overflow")?;
    let mut artifact = VerifiedArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let config = super::decoder::config(capacity)?;
    let mut report =
        crate::kernels::model_benchmark(ptx, device, &mut artifact, &objects, &config, request)?;
    report["identity"] = serde_json::json!(artifact.identity());
    Ok(report)
}
