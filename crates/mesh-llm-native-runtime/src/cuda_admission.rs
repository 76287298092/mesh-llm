use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaDriverOnlyRequirements {
    pub min_driver_api_version: u32,
    pub min_device_memory_bytes: u64,
}

impl CudaDriverOnlyRequirements {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1000..=99999).contains(&self.min_driver_api_version),
            "minimum CUDA driver API version must be between 1000 and 99999"
        );
        ensure!(
            self.min_device_memory_bytes > 0,
            "minimum CUDA device memory must be nonzero"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaSelectedDevice {
    pub ordinal: u32,
    pub uuid: String,
    pub compute_arch: String,
    pub driver_api_version: u32,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaAdmissionRejection {
    InvalidRequirements,
    SelectedDeviceMissing,
    InvalidDeviceEvidence,
    DriverApiTooOld {
        required: u32,
        available: u32,
    },
    ArchitectureUnsupported {
        supported: Vec<String>,
        selected: String,
    },
    InsufficientTotalMemory {
        required: u64,
        available: u64,
    },
    InsufficientFreeMemory {
        required: u64,
        available: u64,
    },
}

impl fmt::Display for CudaAdmissionRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequirements => {
                write!(formatter, "CUDA driver-only requirements are invalid")
            }
            Self::SelectedDeviceMissing => write!(formatter, "selected CUDA device is missing"),
            Self::InvalidDeviceEvidence => {
                write!(formatter, "selected CUDA device evidence is invalid")
            }
            Self::DriverApiTooOld {
                required,
                available,
            } => write!(
                formatter,
                "CUDA driver API {available} is below required version {required}"
            ),
            Self::ArchitectureUnsupported {
                supported,
                selected,
            } => write!(
                formatter,
                "selected CUDA architecture {selected} is unsupported (supported: {})",
                supported.join(", ")
            ),
            Self::InsufficientTotalMemory {
                required,
                available,
            } => write!(
                formatter,
                "CUDA total memory {available} bytes is below required {required} bytes"
            ),
            Self::InsufficientFreeMemory {
                required,
                available,
            } => write!(
                formatter,
                "CUDA free memory {available} bytes is below required {required} bytes"
            ),
        }
    }
}

pub(crate) fn evaluate(
    requirements: &CudaDriverOnlyRequirements,
    supported_arches: &[String],
    selected: Option<&CudaSelectedDevice>,
) -> Vec<CudaAdmissionRejection> {
    if requirements.validate().is_err() || validate_supported_arches(supported_arches).is_err() {
        return vec![CudaAdmissionRejection::InvalidRequirements];
    }
    let Some(selected) = selected else {
        return vec![CudaAdmissionRejection::SelectedDeviceMissing];
    };
    if !valid_device_evidence(selected) {
        return vec![CudaAdmissionRejection::InvalidDeviceEvidence];
    }

    let mut rejections = Vec::new();
    if selected.driver_api_version < requirements.min_driver_api_version {
        rejections.push(CudaAdmissionRejection::DriverApiTooOld {
            required: requirements.min_driver_api_version,
            available: selected.driver_api_version,
        });
    }
    if !supported_arches.contains(&selected.compute_arch) {
        rejections.push(CudaAdmissionRejection::ArchitectureUnsupported {
            supported: supported_arches.to_vec(),
            selected: selected.compute_arch.clone(),
        });
    }
    if selected.total_memory_bytes < requirements.min_device_memory_bytes {
        rejections.push(CudaAdmissionRejection::InsufficientTotalMemory {
            required: requirements.min_device_memory_bytes,
            available: selected.total_memory_bytes,
        });
    }
    if selected.free_memory_bytes < requirements.min_device_memory_bytes {
        rejections.push(CudaAdmissionRejection::InsufficientFreeMemory {
            required: requirements.min_device_memory_bytes,
            available: selected.free_memory_bytes,
        });
    }
    rejections
}

pub(crate) fn validate_supported_arches(arches: &[String]) -> Result<()> {
    ensure!(
        !arches.is_empty(),
        "supported CUDA architectures must not be empty"
    );
    ensure!(
        arches.iter().all(|arch| valid_compute_arch(arch)),
        "supported CUDA architectures must use the sm_<digits> form"
    );
    Ok(())
}

fn valid_device_evidence(device: &CudaSelectedDevice) -> bool {
    valid_text(&device.uuid, 128)
        && valid_compute_arch(&device.compute_arch)
        && (1000..=99999).contains(&device.driver_api_version)
        && device.total_memory_bytes > 0
        && device.free_memory_bytes <= device.total_memory_bytes
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_compute_arch(value: &str) -> bool {
    value.strip_prefix("sm_").is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use super::{CudaAdmissionRejection, CudaDriverOnlyRequirements, CudaSelectedDevice, evaluate};

    fn requirements(driver: u32, memory: u64) -> CudaDriverOnlyRequirements {
        CudaDriverOnlyRequirements {
            min_driver_api_version: driver,
            min_device_memory_bytes: memory,
        }
    }

    fn device(arch: &str, driver: u32, total: u64, free: u64) -> CudaSelectedDevice {
        CudaSelectedDevice {
            ordinal: 2,
            uuid: "GPU-test-uuid".to_string(),
            compute_arch: arch.to_string(),
            driver_api_version: driver,
            total_memory_bytes: total,
            free_memory_bytes: free,
        }
    }

    fn archs(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn invalid_requirements_and_architecture_lists_fail_before_device_selection() {
        for invalid in [
            requirements(999, 1),
            requirements(100_000, 1),
            requirements(1000, 0),
        ] {
            assert!(invalid.validate().is_err());
            assert_eq!(
                evaluate(&invalid, &archs(&["sm_120"]), None),
                [CudaAdmissionRejection::InvalidRequirements]
            );
        }
        for invalid_arches in [&[][..], &["sm_"][..], &["sm_120a"][..], &["SM_120"][..]] {
            assert_eq!(
                evaluate(&requirements(1000, 1), &archs(invalid_arches), None),
                [CudaAdmissionRejection::InvalidRequirements]
            );
        }
    }

    #[test]
    fn missing_selected_device_is_rejected() {
        assert_eq!(
            evaluate(&requirements(1000, 1), &archs(&["sm_120"]), None),
            [CudaAdmissionRejection::SelectedDeviceMissing]
        );
    }

    #[test]
    fn malformed_selected_device_evidence_is_rejected() {
        let base = device("sm_120", 1000, 16, 8);
        let mut invalid = Vec::new();
        let mut candidate = base.clone();
        candidate.uuid.clear();
        invalid.push(candidate);
        let mut candidate = base.clone();
        candidate.uuid = " GPU-test-uuid".to_string();
        invalid.push(candidate);
        let mut candidate = base.clone();
        candidate.uuid = "uuid\ncontrol".to_string();
        invalid.push(candidate);
        let mut candidate = base.clone();
        candidate.uuid = "u".repeat(129);
        invalid.push(candidate);
        for arch in ["sm_", "sm_120a", "sm_120-compat", "SM_120"] {
            let mut candidate = base.clone();
            candidate.compute_arch = arch.to_string();
            invalid.push(candidate);
        }
        for (driver, total, free) in [(999, 16, 8), (100_000, 16, 8), (1000, 0, 0), (1000, 16, 17)]
        {
            invalid.push(device("sm_120", driver, total, free));
        }
        for candidate in invalid {
            assert_eq!(
                evaluate(
                    &requirements(1000, 1),
                    &archs(&["sm_120"]),
                    Some(&candidate)
                ),
                [CudaAdmissionRejection::InvalidDeviceEvidence]
            );
        }
    }

    #[test]
    fn driver_only_admission_accepts_valid_boundary_values_without_toolkit_metadata() {
        for version in [1000, 99_999] {
            let requirements = requirements(version, 8);
            requirements.validate().unwrap();
            assert!(
                evaluate(
                    &requirements,
                    &archs(&["sm_120"]),
                    Some(&device("sm_120", version, 8, 8))
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn driver_architecture_and_memory_rejections_report_selected_device_evidence() {
        let rejected = evaluate(
            &requirements(1200, 10),
            &archs(&["sm_120"]),
            Some(&device("sm_86", 1100, 9, 8)),
        );
        assert!(rejected.contains(&CudaAdmissionRejection::DriverApiTooOld {
            required: 1200,
            available: 1100,
        }));
        assert!(
            rejected.contains(&CudaAdmissionRejection::ArchitectureUnsupported {
                supported: archs(&["sm_120"]),
                selected: "sm_86".to_string(),
            })
        );
        assert!(
            rejected.contains(&CudaAdmissionRejection::InsufficientTotalMemory {
                required: 10,
                available: 9,
            })
        );
        assert!(
            rejected.contains(&CudaAdmissionRejection::InsufficientFreeMemory {
                required: 10,
                available: 8,
            })
        );

        assert_eq!(
            evaluate(
                &requirements(1000, 8),
                &archs(&["sm_120"]),
                Some(&device("sm_120", 1000, 10, 7)),
            ),
            [CudaAdmissionRejection::InsufficientFreeMemory {
                required: 8,
                available: 7,
            }]
        );
    }

    #[test]
    fn exact_memory_thresholds_are_admitted() {
        assert!(
            evaluate(
                &requirements(1000, 8),
                &archs(&["sm_120"]),
                Some(&device("sm_120", 1000, 8, 8)),
            )
            .is_empty()
        );
    }
}
