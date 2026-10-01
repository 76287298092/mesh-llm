pub use super::target_batch_trial::Fixture;
use crate::{artifact::model_source::ModelArtifact, kernels::NativeMtpForwardLoadRequest};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

pub struct Request<'a> {
    pub artifact: &'a Path,
    pub ptx: &'a str,
    pub device: i32,
    pub fixture: &'a Fixture,
}

pub fn run(request: Request<'_>) -> Result<Value> {
    request.fixture.validate()?;
    ensure!(request.device >= 0, "device ordinal must be nonnegative");
    let mut artifact = ModelArtifact::open(request.artifact)?;
    let result = (|| {
        super::inventory::validate(artifact.directory())?;
        let objects = super::schedule::text_objects(artifact.directory())?;
        let config = super::decoder::config(request.fixture.capacity)?;
        crate::kernels::native_mtp_forward_check(NativeMtpForwardLoadRequest {
            artifact: &mut artifact,
            objects: &objects,
            config: &config,
            ptx: request.ptx,
            device: request.device,
            fixture: request.fixture,
        })
    })();
    let mut report = result.unwrap_or_else(|error| {
        json!({
            "all_passed": false, "error": format!("{error:#}")
        })
    });
    report["identity"] = json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    for claim in [
        "source_arithmetic_qualified",
        "native_mtp_admitted",
        "model_executable",
        "timing_claim",
    ] {
        report[claim] = json!(false);
    }
    Ok(report)
}
