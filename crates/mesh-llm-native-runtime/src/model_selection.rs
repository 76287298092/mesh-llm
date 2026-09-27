//! Exact identity policy. Artifact residency and verified identity extraction
//! belong to the host's artifact reader; this module performs no I/O.

use crate::{CandidateEvaluation, HostRuntimeProfile, NativeRuntimeArtifact, RuntimeSelection};
use crate::{
    CandidateRejection,
    model_identity::{ModelIdentity, serves_model},
};

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
    crate::resolver::evaluate_artifact_for_model(
        artifact,
        profile,
        mesh_version,
        skippy_abi,
        selection,
        Some(requested),
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
    let evaluated: Vec<_> = artifacts
        .iter()
        .map(|artifact| {
            evaluate_native_runtime_artifact_for_model(
                artifact,
                profile,
                mesh_version,
                skippy_abi,
                selection,
                requested,
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
