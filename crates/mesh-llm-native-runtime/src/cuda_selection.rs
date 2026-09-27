//! Driver-only manifest policy. Selected-device evidence comes from the caller's
//! CUDA context, not from a union of architectures across host GPUs.

#[cfg(test)]
#[path = "cuda_selection_tests.rs"]
mod tests;

use crate::cuda_admission::{self, CudaAdmissionRejection, CudaSelectedDevice};
use crate::{CandidateRejection, NativeRuntimeArtifact, NativeRuntimeBackendKind};
use anyhow::{Result, ensure};

pub(crate) fn is_driver_only(artifact: &NativeRuntimeArtifact) -> bool {
    artifact
        .backend
        .cuda
        .as_ref()
        .is_some_and(|cuda| cuda.driver_only.is_some())
}

pub(crate) fn validate_driver_only_artifact(artifact: &NativeRuntimeArtifact) -> Result<()> {
    let Some(cuda) = &artifact.backend.cuda else {
        return Ok(());
    };
    let Some(driver_only) = &cuda.driver_only else {
        return Ok(());
    };
    ensure!(
        artifact.backend.kind == NativeRuntimeBackendKind::Cuda,
        "driver-only requirements need a CUDA backend"
    );
    ensure!(
        !artifact.serves.is_empty(),
        "driver-only prototype requires exact served identities"
    );
    ensure!(
        cuda.toolkit_major == 0,
        "driver-only CUDA must set toolkit_major=0, not declare a toolkit dependency"
    );
    ensure!(
        cuda.min_driver.is_none(),
        "driver-only CUDA uses min_driver_api_version, not min_driver"
    );
    driver_only.validate()?;
    cuda_admission::validate_supported_arches(&cuda.gpu_arches)
}

pub(crate) fn evaluate(
    artifact: &NativeRuntimeArtifact,
    selected: Option<&CudaSelectedDevice>,
    reasons: &mut Vec<CandidateRejection>,
) {
    if validate_driver_only_artifact(artifact).is_err() {
        reasons.push(CandidateRejection::CudaAdmission(
            CudaAdmissionRejection::InvalidRequirements,
        ));
        return;
    }
    let Some(cuda) = &artifact.backend.cuda else {
        return;
    };
    let Some(requirements) = &cuda.driver_only else {
        return;
    };
    reasons.extend(
        cuda_admission::evaluate(requirements, &cuda.gpu_arches, selected)
            .into_iter()
            .map(CandidateRejection::CudaAdmission),
    );
}
