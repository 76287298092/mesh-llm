//! Exact identity policy. Artifact residency and verified identity extraction
//! belong to the host's artifact reader; this module performs no I/O.

use crate::cuda_admission::CudaSelectedDevice;
use crate::{CandidateEvaluation, HostRuntimeProfile, NativeRuntimeArtifact, RuntimeSelection};
use crate::{
    CandidateRejection,
    model_identity::{ModelIdentity, serves_model},
};

/// Verified model identity and an optional snapshot from the CUDA context that
/// will execute it. Memory admission is a snapshot, not an allocation reservation.
#[derive(Clone, Copy, Debug)]
pub struct ModelRuntimeRequest<'a> {
    pub identity: &'a ModelIdentity,
    pub cuda_device: Option<&'a CudaSelectedDevice>,
}

/// Pure eligibility check for an identity already verified from a resident
/// artifact by the caller. This function neither downloads nor verifies files.
pub fn evaluate_native_runtime_artifact_for_model(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
    requested: &ModelIdentity,
) -> CandidateEvaluation {
    evaluate_native_runtime_artifact_for_request(
        artifact,
        profile,
        mesh_version,
        skippy_abi,
        selection,
        &ModelRuntimeRequest {
            identity: requested,
            cuda_device: None,
        },
    )
}

/// Evaluate an exact model and selected device without probing, loading, or
/// downloading. Driver-only candidates require the selected device evidence.
pub fn evaluate_native_runtime_artifact_for_request(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
    request: &ModelRuntimeRequest<'_>,
) -> CandidateEvaluation {
    crate::resolver::evaluate_artifact_for_request(
        artifact,
        profile,
        mesh_version,
        skippy_abi,
        selection,
        Some(request),
    )
}

/// Pure ranking policy over caller-supplied artifacts for a verified model.
/// Automatic startup remains on the identity-absent path until resident-artifact
/// discovery is integrated; this helper performs no installation or download.
pub fn select_native_runtime_from_artifacts_for_model(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
    requested: &ModelIdentity,
) -> Option<CandidateEvaluation> {
    select_native_runtime_from_artifacts_for_request(
        artifacts,
        profile,
        mesh_version,
        skippy_abi,
        selection,
        &ModelRuntimeRequest {
            identity: requested,
            cuda_device: None,
        },
    )
}

/// Rank supplied runtime artifacts for a verified model and selected CUDA device.
/// This performs no I/O and never reserves the observed free device memory.
pub fn select_native_runtime_from_artifacts_for_request(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
    request: &ModelRuntimeRequest<'_>,
) -> Option<CandidateEvaluation> {
    let evaluated: Vec<_> = artifacts
        .iter()
        .map(|artifact| {
            evaluate_native_runtime_artifact_for_request(
                artifact,
                profile,
                mesh_version,
                skippy_abi,
                selection,
                request,
            )
        })
        .collect();
    crate::resolver::best_candidate(&evaluated).cloned()
}

pub(crate) fn rejection(
    serves: &[ModelIdentity],
    requested: Option<&ModelIdentity>,
) -> Option<CandidateRejection> {
    if serves_model(serves, requested) {
        return None;
    }
    Some(match requested {
        Some(requested) => CandidateRejection::ModelNotServed {
            requested: requested.clone(),
            serves: serves.to_vec(),
        },
        None => CandidateRejection::ModelIdentityMissing,
    })
}

#[cfg(test)]
mod tests;
