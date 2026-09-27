//! One resident decode profiled against an identical unprofiled session.
use crate::artifact::reader::VerifiedArtifact;
use anyhow::{Result, ensure};
use std::path::Path;

pub fn run(path: &Path, ptx: &str, device: i32, tokens: &[u32]) -> Result<serde_json::Value> {
    ensure!(
        (1..=512).contains(&tokens.len()),
        "model profile requires 1..512 prefix tokens"
    );
    let mut artifact = VerifiedArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let config = super::decoder::config(tokens.len() + 1)?;
    let mut report =
        crate::kernels::model_profile(ptx, device, &mut artifact, &objects, &config, tokens)?;
    report["identity"] = serde_json::json!(artifact.identity());
    Ok(report)
}
