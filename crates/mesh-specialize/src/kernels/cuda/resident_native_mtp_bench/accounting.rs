use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Serialize)]
pub(crate) enum EvidenceStatus {
    Missing,
    Failed { evidence: String },
    Qualified { evidence: String },
}

impl EvidenceStatus {
    const fn qualified(&self) -> bool {
        match self {
            Self::Qualified { .. } => true,
            Self::Missing | Self::Failed { .. } => false,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Missing => Ok(()),
            Self::Qualified { evidence } | Self::Failed { evidence } => {
                ensure!(
                    !evidence.trim().is_empty(),
                    "prerequisite evidence reference is empty"
                );
                Ok(())
            }
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Prerequisites {
    pub full_native_path: EvidenceStatus,
    pub target_batch: EvidenceStatus,
    pub meaningful_quality: EvidenceStatus,
}

impl Prerequisites {
    pub(super) fn validate(&self) -> Result<()> {
        self.full_native_path.validate()?;
        self.target_batch.validate()?;
        self.meaningful_quality.validate()
    }

    pub(super) const fn qualifies(&self, matched: bool) -> bool {
        matched
            && self.full_native_path.qualified()
            && self.target_batch.qualified()
            && self.meaningful_quality.qualified()
    }
}

#[derive(Serialize)]
pub(super) struct OutputCounts {
    pub emitted: usize,
    pub emitted_during_decode: usize,
    pub decode_tokens_per_second: Option<f64>,
}

pub(super) fn counts(tokens: &[u32], decode_seconds: f64) -> Result<OutputCounts> {
    let emitted_during_decode = tokens
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("complete run omitted its prefill output"))?;
    Ok(OutputCounts {
        emitted: tokens.len(),
        emitted_during_decode,
        decode_tokens_per_second: ratio(emitted_during_decode, decode_seconds),
    })
}

pub(super) fn ratio(numerator: usize, denominator: f64) -> Option<f64> {
    let numerator = f64::from(u32::try_from(numerator).ok()?);
    (denominator.is_finite() && denominator > 0.0).then_some(numerator / denominator)
}

#[derive(Serialize)]
pub(super) struct Attribution {
    pub reported_phase_seconds: f64,
    pub unassigned_decode_seconds: Option<f64>,
    pub exceeds_decode_interval: bool,
}

pub(super) fn attribution(
    decode: f64,
    phases: &[f64],
    nonoverlapping_host_intervals: bool,
) -> Attribution {
    let sum = phases.iter().sum::<f64>();
    let usable = decode.is_finite()
        && decode >= 0.0
        && phases
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0);
    Attribution {
        reported_phase_seconds: sum,
        unassigned_decode_seconds: (usable && nonoverlapping_host_intervals && sum <= decode)
            .then_some(decode - sum),
        exceeds_decode_interval: usable && sum > decode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_uses_actual_emissions_when_budget_is_not_the_observed_count() {
        let observed = [7, 8, 9, 10];
        let result = counts(&observed, 1.5).unwrap();
        assert_eq!((result.emitted, result.emitted_during_decode), (4, 3));
        assert!((result.decode_tokens_per_second.unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn overlapping_attribution_does_not_create_a_total_or_residual() {
        let result = attribution(3.0, &[2.0, 2.0], false);
        assert!(result.unassigned_decode_seconds.is_none());
        assert!(result.exceeds_decode_interval);
    }

    #[test]
    fn nonoverlapping_attribution_retains_unassigned_host_time() {
        let result = attribution(4.0, &[1.0, 0.5], true);
        assert!((result.unassigned_decode_seconds.unwrap() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn overlapping_phases_below_total_still_have_no_inferred_residual() {
        let result = attribution(4.0, &[1.0, 0.5], false);
        assert!(result.unassigned_decode_seconds.is_none());
    }

    #[test]
    fn absent_output_and_zero_time_cannot_produce_throughput() {
        assert!(counts(&[], 1.0).is_err());
        assert!(
            counts(&[1, 2], 0.0)
                .unwrap()
                .decode_tokens_per_second
                .is_none()
        );
    }

    #[test]
    fn failed_or_missing_prerequisites_prevent_qualification_despite_matched_runs() {
        for full_native_path in [
            EvidenceStatus::Missing,
            EvidenceStatus::Failed {
                evidence: "failed-run".into(),
            },
        ] {
            let prerequisites = Prerequisites {
                full_native_path,
                target_batch: EvidenceStatus::Qualified {
                    evidence: "batch-run".into(),
                },
                meaningful_quality: EvidenceStatus::Qualified {
                    evidence: "quality-run".into(),
                },
            };
            assert!(!prerequisites.qualifies(true));
        }
    }

    #[test]
    fn mismatched_runs_prevent_qualification_even_with_external_prerequisites() {
        let prerequisites = Prerequisites {
            full_native_path: EvidenceStatus::Qualified {
                evidence: "native-run".into(),
            },
            target_batch: EvidenceStatus::Qualified {
                evidence: "batch-run".into(),
            },
            meaningful_quality: EvidenceStatus::Qualified {
                evidence: "quality-run".into(),
            },
        };
        assert!(!prerequisites.qualifies(false));
        assert!(prerequisites.qualifies(true));
    }
}
