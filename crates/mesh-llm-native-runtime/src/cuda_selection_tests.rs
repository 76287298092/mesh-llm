use crate::cuda_admission::{
    CudaAdmissionRejection, CudaDriverOnlyRequirements, CudaSelectedDevice,
};
use crate::model_identity::ModelIdentity;
use crate::model_selection::{
    ModelRuntimeRequest, evaluate_native_runtime_artifact_for_model,
    evaluate_native_runtime_artifact_for_request, select_native_runtime_from_artifacts_for_request,
};
use crate::{
    CandidateRejection, HostCudaProfile, HostRuntimeProfile, NativeRuntimeArtifact,
    NativeRuntimeBackend, NativeRuntimeBackendKind, NativeRuntimeManifest, NativeRuntimePlatform,
    RuntimeSelection,
};
use std::collections::BTreeSet;

const MESH_VERSION: &str = "0.76.1";
const SKIPPY_ABI: &str = "0.1.64";
const DRIVER_API: u32 = 13_040;
const MIN_MEMORY: u64 = 1_000;

fn identity() -> ModelIdentity {
    ModelIdentity {
        model_id: "model-5090".to_string(),
        weights_id: "weights-5090".to_string(),
    }
}

fn artifact(id: &str, rank: i64, backend: NativeRuntimeBackend) -> NativeRuntimeArtifact {
    NativeRuntimeArtifact {
        id: id.to_string(),
        serves: vec![identity()],
        mesh_version: Some(MESH_VERSION.to_string()),
        skippy_abi: SKIPPY_ABI.to_string(),
        platform: NativeRuntimePlatform {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            target: None,
            min_glibc: None,
        },
        backend,
        rank,
        libraries: vec!["lib/libllama.so".to_string()],
        files: Default::default(),
        tools: Default::default(),
        url: None,
        sha256: None,
        signature: None,
    }
}

fn general_runtime() -> NativeRuntimeArtifact {
    let mut general = artifact("general-cpu", 0, NativeRuntimeBackend::cpu());
    general.serves.clear();
    general
}

fn driver_only_runtime() -> NativeRuntimeArtifact {
    let mut backend = NativeRuntimeBackend::cuda(0, vec!["sm_120".to_string()]);
    backend.cuda.as_mut().unwrap().driver_only = Some(CudaDriverOnlyRequirements {
        min_driver_api_version: DRIVER_API,
        min_device_memory_bytes: MIN_MEMORY,
    });
    artifact("cuda-5090-specialized", 20_000, backend)
}

fn selected_device(
    arch: &str,
    driver_api_version: u32,
    total: u64,
    free: u64,
) -> CudaSelectedDevice {
    CudaSelectedDevice {
        ordinal: 1,
        uuid: "GPU-5090-test-uuid".to_string(),
        compute_arch: arch.to_string(),
        driver_api_version,
        total_memory_bytes: total,
        free_memory_bytes: free,
    }
}

fn profile() -> HostRuntimeProfile {
    HostRuntimeProfile {
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
        target_triple: None,
        glibc_version: None,
        available_flavors: BTreeSet::from([
            NativeRuntimeBackendKind::Cpu,
            NativeRuntimeBackendKind::Cuda,
        ]),
        gpus: Vec::new(),
        cuda: Some(HostCudaProfile {
            toolkit_majors: BTreeSet::new(),
            driver_max_major: Some(13),
            driver_version: Some("575.57.08".to_string()),
            gpu_arches: BTreeSet::from(["sm_86".to_string(), "sm_120".to_string()]),
        }),
        rocm: None,
        vulkan: None,
    }
}

fn request<'a>(
    identity: &'a ModelIdentity,
    device: Option<&'a CudaSelectedDevice>,
) -> ModelRuntimeRequest<'a> {
    ModelRuntimeRequest {
        identity,
        cuda_device: device,
    }
}

fn select(request: &ModelRuntimeRequest<'_>) -> Option<crate::CandidateEvaluation> {
    let artifacts = [general_runtime(), driver_only_runtime()];
    select_native_runtime_from_artifacts_for_request(
        &artifacts,
        &profile(),
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Recommended,
        request,
    )
}

fn evaluate(
    artifact: &NativeRuntimeArtifact,
    request: &ModelRuntimeRequest<'_>,
    skippy_abi: Option<&str>,
) -> crate::CandidateEvaluation {
    evaluate_native_runtime_artifact_for_request(
        artifact,
        &profile(),
        MESH_VERSION,
        skippy_abi,
        &RuntimeSelection::Recommended,
        request,
    )
}

#[test]
fn old_cuda_json_defaults_driver_only_to_none_and_omits_it() {
    let old_json = serde_json::json!({
        "kind": "cuda",
        "cuda": {"toolkit_major": 12, "gpu_arches": ["sm_120"]},
        "rocm": null,
        "vulkan": null
    });
    let backend: NativeRuntimeBackend = serde_json::from_value(old_json).unwrap();
    assert!(backend.cuda.as_ref().unwrap().driver_only.is_none());

    let encoded = serde_json::to_value(backend).unwrap();
    assert!(encoded["cuda"].get("driver_only").is_none());
}

#[test]
fn exact_model_and_selected_5090_accept_driver_only_without_toolkit_and_win_rank() {
    let host = profile();
    assert!(host.cuda.as_ref().unwrap().toolkit_majors.is_empty());
    assert_eq!(
        host.cuda.as_ref().unwrap().gpu_arches,
        BTreeSet::from(["sm_86".to_string(), "sm_120".to_string()])
    );
    let model = identity();
    let device = selected_device("sm_120", DRIVER_API, 2_000, 2_000);
    let evaluation = select(&request(&model, Some(&device))).unwrap();

    assert!(evaluation.compatible);
    assert_eq!(evaluation.artifact.id, "cuda-5090-specialized");
    assert!(evaluation.rank > 100);
}

#[test]
fn selected_device_architecture_controls_admission_not_host_arch_union() {
    let model = identity();
    let device = selected_device("sm_86", DRIVER_API, 2_000, 2_000);
    let evaluation = evaluate(
        &driver_only_runtime(),
        &request(&model, Some(&device)),
        Some(SKIPPY_ABI),
    );
    assert!(!evaluation.compatible);
    assert!(
        evaluation
            .rejection_reasons
            .contains(&CandidateRejection::CudaAdmission(
                CudaAdmissionRejection::ArchitectureUnsupported {
                    supported: vec!["sm_120".to_string()],
                    selected: "sm_86".to_string(),
                }
            ))
    );
    assert_eq!(
        select(&request(&model, Some(&device))).unwrap().artifact.id,
        "general-cpu"
    );
}

#[test]
fn existing_model_api_rejects_driver_only_runtime_without_selected_device() {
    let model = identity();
    let evaluation = evaluate_native_runtime_artifact_for_model(
        &driver_only_runtime(),
        &profile(),
        MESH_VERSION,
        Some(SKIPPY_ABI),
        &RuntimeSelection::Recommended,
        &model,
    );
    assert!(!evaluation.compatible);
    assert!(
        evaluation
            .rejection_reasons
            .contains(&CandidateRejection::CudaAdmission(
                CudaAdmissionRejection::SelectedDeviceMissing
            ))
    );
}

#[test]
fn insufficient_driver_api_or_device_memory_falls_back_to_general_runtime() {
    let model = identity();
    let cases = [
        (
            selected_device("sm_120", 12_999, 2_000, 2_000),
            CudaAdmissionRejection::DriverApiTooOld {
                required: DRIVER_API,
                available: 12_999,
            },
        ),
        (
            selected_device("sm_120", DRIVER_API, 999, 900),
            CudaAdmissionRejection::InsufficientTotalMemory {
                required: MIN_MEMORY,
                available: 999,
            },
        ),
        (
            selected_device("sm_120", DRIVER_API, 2_000, 999),
            CudaAdmissionRejection::InsufficientFreeMemory {
                required: MIN_MEMORY,
                available: 999,
            },
        ),
    ];

    for (device, reason) in cases {
        let evaluation = evaluate(
            &driver_only_runtime(),
            &request(&model, Some(&device)),
            Some(SKIPPY_ABI),
        );
        assert!(!evaluation.compatible);
        assert!(
            evaluation
                .rejection_reasons
                .contains(&CandidateRejection::CudaAdmission(reason))
        );
        assert_eq!(
            select(&request(&model, Some(&device))).unwrap().artifact.id,
            "general-cpu"
        );
    }
}

#[test]
fn exact_model_identity_does_not_bypass_abi_compatibility() {
    let model = identity();
    let device = selected_device("sm_120", DRIVER_API, 2_000, 2_000);
    let evaluation = evaluate(
        &driver_only_runtime(),
        &request(&model, Some(&device)),
        Some("different-abi"),
    );
    assert!(!evaluation.compatible);
    assert!(
        evaluation
            .rejection_reasons
            .iter()
            .any(|reason| { matches!(reason, CandidateRejection::SkippyAbiMismatch { .. }) })
    );
}

#[test]
fn malformed_driver_only_requirements_fail_manifest_and_pure_evaluation() {
    let mut invalid = Vec::new();

    let mut toolkit = driver_only_runtime();
    toolkit.backend.cuda.as_mut().unwrap().toolkit_major = 12;
    invalid.push(toolkit);

    let mut minimum_driver = driver_only_runtime();
    minimum_driver.backend.cuda.as_mut().unwrap().min_driver = Some("550.54".to_string());
    invalid.push(minimum_driver);

    let mut empty_serves = driver_only_runtime();
    empty_serves.serves.clear();
    invalid.push(empty_serves);

    let mut wrong_backend = driver_only_runtime();
    wrong_backend.backend.kind = NativeRuntimeBackendKind::Cpu;
    invalid.push(wrong_backend);

    let mut empty_arches = driver_only_runtime();
    empty_arches
        .backend
        .cuda
        .as_mut()
        .unwrap()
        .gpu_arches
        .clear();
    invalid.push(empty_arches);

    let model = identity();
    let device = selected_device("sm_120", DRIVER_API, 2_000, 2_000);
    for artifact in invalid {
        assert!(
            NativeRuntimeManifest {
                runtime: artifact.clone()
            }
            .validate()
            .is_err()
        );
        let evaluation = evaluate(&artifact, &request(&model, Some(&device)), Some(SKIPPY_ABI));
        assert!(!evaluation.compatible);
        assert!(
            evaluation
                .rejection_reasons
                .contains(&CandidateRejection::CudaAdmission(
                    CudaAdmissionRejection::InvalidRequirements
                ))
        );
    }
}

#[test]
fn toolkit_cuda_without_driver_only_still_needs_installed_toolkit() {
    let model = identity();
    let device = selected_device("sm_120", DRIVER_API, 2_000, 2_000);
    let toolkit_runtime = artifact(
        "cuda12-toolkit-runtime",
        20_000,
        NativeRuntimeBackend::cuda(12, vec!["sm_120".to_string()]),
    );
    let evaluation = evaluate(
        &toolkit_runtime,
        &request(&model, Some(&device)),
        Some(SKIPPY_ABI),
    );
    assert!(!evaluation.compatible);
    assert!(
        evaluation
            .rejection_reasons
            .contains(&CandidateRejection::CudaToolkitNotDetected { required: 12 })
    );
}
