use super::{
    evaluate_native_runtime_artifact_for_model, select_native_runtime_from_artifacts_for_model,
};
use crate::model_identity::ModelIdentity;
use crate::{
    CandidateRejection, HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeBackend,
    NativeRuntimeBackendKind, NativeRuntimeCache, NativeRuntimeManifest, NativeRuntimePlatform,
    NativeRuntimeReleaseManifest, NativeRuntimeResolver, NativeRuntimeSource, RuntimeSelection,
    evaluate_native_runtime_artifact,
};
use std::collections::BTreeSet;
use std::path::Path;

const MESH_VERSION: &str = "0.76.1";
const SKIPPY_ABI: &str = "0.1.64";

fn identity(model_id: &str, weights_id: &str) -> ModelIdentity {
    ModelIdentity {
        model_id: model_id.to_string(),
        weights_id: weights_id.to_string(),
    }
}

fn artifact(id: &str, rank: i64, serves: Vec<ModelIdentity>) -> NativeRuntimeArtifact {
    NativeRuntimeArtifact {
        id: id.to_string(),
        serves,
        mesh_version: Some(MESH_VERSION.to_string()),
        skippy_abi: SKIPPY_ABI.to_string(),
        platform: NativeRuntimePlatform {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            target: None,
            min_glibc: None,
        },
        backend: NativeRuntimeBackend::cpu(),
        rank,
        libraries: vec!["lib/libllama.so".to_string()],
        files: Default::default(),
        tools: Default::default(),
        url: None,
        sha256: None,
        signature: None,
    }
}

fn profile() -> HostRuntimeProfile {
    HostRuntimeProfile {
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
        target_triple: None,
        glibc_version: None,
        available_flavors: BTreeSet::from([NativeRuntimeBackendKind::Cpu]),
        gpus: Vec::new(),
        cuda: None,
        rocm: None,
        vulkan: None,
    }
}

fn candidates() -> Vec<NativeRuntimeArtifact> {
    vec![
        artifact("general", 0, Vec::new()),
        artifact(
            "specialized",
            10_000,
            vec![identity("model-a", "weights-a")],
        ),
    ]
}

fn same_id_artifacts() -> (NativeRuntimeArtifact, NativeRuntimeArtifact) {
    let runtime_id = "meshllm-runtime-linux-x86_64-cpu";
    (
        artifact(runtime_id, 0, Vec::new()),
        artifact(runtime_id, 0, vec![identity("model-a", "weights-a")]),
    )
}

fn write_runtime(path: &Path, runtime: NativeRuntimeArtifact) {
    std::fs::create_dir_all(path.join("lib")).unwrap();
    std::fs::write(path.join("lib/libllama.so"), b"resident runtime fixture").unwrap();
    NativeRuntimeManifest { runtime }
        .write_to_dir(path)
        .unwrap();
}

fn resolver(
    release_runtime: NativeRuntimeArtifact,
    cache: NativeRuntimeCache,
) -> NativeRuntimeResolver {
    NativeRuntimeResolver::new(
        MESH_VERSION,
        profile(),
        NativeRuntimeReleaseManifest {
            mesh_version: MESH_VERSION.to_string(),
            skippy_abi: SKIPPY_ABI.to_string(),
            artifacts: vec![release_runtime],
        },
        cache,
    )
    .with_skippy_abi_version(SKIPPY_ABI)
}

fn selected_id(requested: &ModelIdentity) -> String {
    select_native_runtime_from_artifacts_for_model(
        &candidates(),
        &profile(),
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Recommended,
        requested,
    )
    .expect("general CPU runtime should remain eligible")
    .artifact
    .id
}

#[test]
fn legacy_artifact_defaults_serves_and_omits_it_when_empty() {
    let mut value = serde_json::to_value(artifact("legacy", 0, Vec::new())).unwrap();
    value.as_object_mut().unwrap().remove("serves");

    let decoded: NativeRuntimeArtifact = serde_json::from_value(value).unwrap();
    assert!(decoded.serves.is_empty());
    let encoded = serde_json::to_value(decoded).unwrap();
    assert!(!encoded.as_object().unwrap().contains_key("serves"));
}

#[test]
fn manifest_validation_rejects_invalid_served_identity() {
    let runtime = artifact(
        "invalid-identity",
        0,
        vec![identity("model-a ", "weights-a")],
    );
    assert!(NativeRuntimeManifest { runtime }.validate().is_err());
}

#[test]
fn legacy_evaluation_rejects_specialization_without_identity_even_for_exact_id() {
    let specialization = candidates().pop().unwrap();
    let evaluation = evaluate_native_runtime_artifact(
        &specialization,
        &profile(),
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Id("specialized".to_string()),
    );
    assert!(!evaluation.compatible);
    assert!(
        evaluation
            .rejection_reasons
            .contains(&CandidateRejection::ModelIdentityMissing)
    );
}

#[test]
fn exact_model_and_weights_select_the_high_rank_specialization() {
    let requested = identity("model-a", "weights-a");
    let selected = select_native_runtime_from_artifacts_for_model(
        &candidates(),
        &profile(),
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Recommended,
        &requested,
    )
    .unwrap();
    assert_eq!(selected.artifact.id, "specialized");
    assert!(selected.rank > 100);
}

#[test]
fn different_model_or_weights_falls_back_to_general_runtime() {
    assert_eq!(selected_id(&identity("model-a", "weights-b")), "general");
    assert_eq!(selected_id(&identity("model-b", "weights-a")), "general");
}

#[test]
fn exact_identity_does_not_bypass_abi_or_platform_compatibility() {
    let specialization = candidates().pop().unwrap();
    let requested = identity("model-a", "weights-a");

    let wrong_abi = evaluate_native_runtime_artifact_for_model(
        &specialization,
        &profile(),
        MESH_VERSION,
        Some("different-abi"),
        &RuntimeSelection::Id("specialized".to_string()),
        &requested,
    );
    assert!(!wrong_abi.compatible);
    assert!(
        wrong_abi
            .rejection_reasons
            .iter()
            .any(|reason| matches!(reason, CandidateRejection::SkippyAbiMismatch { .. }))
    );

    let mut wrong_platform = profile();
    wrong_platform.os = "macos".to_string();
    let wrong_platform = evaluate_native_runtime_artifact_for_model(
        &specialization,
        &wrong_platform,
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Id("specialized".to_string()),
        &requested,
    );
    assert!(!wrong_platform.compatible);
    assert!(
        wrong_platform
            .rejection_reasons
            .iter()
            .any(|reason| matches!(reason, CandidateRejection::OsMismatch { .. }))
    );
}

#[test]
fn same_id_bundle_with_different_serves_metadata_is_not_selected_as_resident() {
    let bundle = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let (general, specialized) = same_id_artifacts();
    write_runtime(bundle.path(), specialized);

    let resolution = resolver(general, NativeRuntimeCache::new(cache_root.path()))
        .with_bundle_dirs(vec![bundle.path().to_path_buf()])
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

    assert!(resolution.selected.serves.is_empty());
    assert!(matches!(resolution.source, NativeRuntimeSource::Missing));
}

#[test]
fn same_id_cached_runtime_with_different_serves_metadata_is_not_selected_as_resident() {
    let resident_dir = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let (general, specialized) = same_id_artifacts();
    write_runtime(resident_dir.path(), specialized);
    let cache = NativeRuntimeCache::new(cache_root.path());
    cache.install_from_dir(resident_dir.path()).unwrap();

    let resolution = resolver(general, cache)
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

    assert!(resolution.selected.serves.is_empty());
    assert!(matches!(resolution.source, NativeRuntimeSource::Missing));
}

#[test]
fn resident_backend_requirements_must_match_the_evaluated_artifact() {
    for cached in [false, true] {
        let resident = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let general = artifact("same-runtime", 0, Vec::new());
        let mut different_backend = general.clone();
        different_backend.backend = NativeRuntimeBackend::cuda(12, vec!["sm_86".into()]);
        write_runtime(resident.path(), different_backend);
        let cache = NativeRuntimeCache::new(cache_root.path());
        if cached {
            cache.install_from_dir(resident.path()).unwrap();
        }
        let resolver = resolver(general, cache).with_bundle_dirs(if cached {
            Vec::new()
        } else {
            vec![resident.path().to_path_buf()]
        });
        let resolution = resolver.resolve(&RuntimeSelection::Recommended).unwrap();
        assert_eq!(
            resolution.selected.backend.kind,
            NativeRuntimeBackendKind::Cpu
        );
        assert!(matches!(resolution.source, NativeRuntimeSource::Missing));
    }
}
