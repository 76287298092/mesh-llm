//! Windowed teacher-forced scoring of fixed token streams over the resident decoder.
use crate::{
    artifact::model_source::ModelArtifact,
    kernels::{ModelScoreRequest, ScoreSink},
};
use anyhow::{Result, ensure};
use std::path::Path;

/// Largest context one forward accepts until chunked prefill exists.
pub const MAX_CONTEXT: usize = 512;

pub fn run(
    path: &Path,
    ptx: &str,
    device: i32,
    request: &ModelScoreRequest<'_>,
    sink: &mut ScoreSink<'_>,
) -> Result<serde_json::Value> {
    ensure!(
        (2..=MAX_CONTEXT).contains(&request.context),
        "context {} is unsupported: teacher-forced scoring runs one forward per window and is limited to 2..={MAX_CONTEXT} until chunked prefill exists",
        request.context
    );
    ensure!(
        (1..request.context).contains(&request.stride),
        "stride must be in 1..context"
    );
    let mut artifact = ModelArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let config = super::decoder::config(request.context)?;
    let mut report =
        crate::kernels::model_score(ptx, device, &mut artifact, &objects, &config, request, sink)?;
    report["identity"] = serde_json::json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    Ok(report)
}
