use std::ops::Range;

#[cfg(target_os = "linux")]
use super::NativeMtpQ8ProjectionResidentRequest;

pub const NATIVE_MTP_Q8_PROJECTION_CASE_COUNT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8ProjectionCaseRange {
    start: usize,
    end: usize,
}

impl Q8ProjectionCaseRange {
    pub fn new(start: usize, end: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            start < end && end <= NATIVE_MTP_Q8_PROJECTION_CASE_COUNT,
            "case range must satisfy 0 <= start < end <= 16"
        );
        Ok(Self { start, end })
    }

    pub const fn all() -> Self {
        Self {
            start: 0,
            end: NATIVE_MTP_Q8_PROJECTION_CASE_COUNT,
        }
    }

    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn end(self) -> usize {
        self.end
    }

    pub fn indices(self) -> Range<usize> {
        self.start..self.end
    }

    #[cfg(any(test, target_os = "linux"))]
    pub(super) fn cases(self) -> impl Iterator<Item = Case> {
        self.indices().map(case_at)
    }
}

#[cfg(any(test, target_os = "linux"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Projection {
    QueryKeyValue,
    MlpGateUp,
    AttentionOutput,
    MlpDown,
}

#[cfg(any(test, target_os = "linux"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DensePattern {
    AlternatingUnit,
    SignedMix,
}

#[cfg(any(test, target_os = "linux"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Case {
    pub(super) index: usize,
    pub(super) projection: Projection,
    pub(super) tokens: usize,
    pub(super) pattern: DensePattern,
}

#[cfg(target_os = "linux")]
pub(super) fn run(
    request: NativeMtpQ8ProjectionResidentRequest<'_>,
) -> anyhow::Result<serde_json::Value> {
    use super::cuda::native_mtp_q8_projection::{
        ProjectionKind, RealParentQualificationRequest, qualify_real_parent,
    };

    let cases = request
        .case_range
        .cases()
        .map(|case| {
            let kind = match case.projection {
                Projection::QueryKeyValue => ProjectionKind::QueryKeyValue,
                Projection::MlpGateUp => ProjectionKind::MlpGateUp,
                Projection::AttentionOutput => ProjectionKind::AttentionOutput,
                Projection::MlpDown => ProjectionKind::MlpDown,
            };
            let pattern = match case.pattern {
                DensePattern::AlternatingUnit => {
                    super::cuda::native_mtp_q8_projection::DensePattern::AlternatingUnit
                }
                DensePattern::SignedMix => {
                    super::cuda::native_mtp_q8_projection::DensePattern::SignedMix
                }
            };
            let trial_request = RealParentQualificationRequest {
                artifact: request.artifact,
                ptx: request.ptx,
                device: request.device,
                kind,
                tokens: case.tokens,
                pattern,
            };
            let (result, error) = match qualify_real_parent(trial_request) {
                Ok(result) => (result, None),
                Err(error) => (serde_json::Value::Null, Some(format!("{error:#}"))),
            };
            let case_passed =
                error.is_none() && result.get("all_passed") == Some(&serde_json::Value::Bool(true));
            serde_json::json!({
                "case_index": case.index,
                "projection": format!("{:?}", case.projection),
                "tokens": case.tokens,
                "dense_pattern": format!("{:?}", case.pattern),
                "all_passed": case_passed,
                "result": result,
                "error": error,
            })
        })
        .collect::<Vec<_>>();
    let all_passed = cases
        .iter()
        .all(|case| case.get("all_passed") == Some(&serde_json::Value::Bool(true)));
    let selected_case_indices = request.case_range.indices().collect::<Vec<_>>();
    Ok(serde_json::json!({
        "schema_version": 1,
        "kind": "native-mtp-q8-resident-projection-matrix-qualification-v1",
        "case_count": cases.len(),
        "selected_case_range": {
            "start": request.case_range.start(),
            "end": request.case_range.end(),
        },
        "selected_case_indices": selected_case_indices,
        "cases": cases,
        "all_passed": all_passed,
        "native_mtp_admitted": false,
        "model_executable": false,
        "full_model_executed": false,
        "timing_claim": false,
        "timing_collected": false,
        "model_prefill_tokens_per_second": null,
        "model_decode_tokens_per_second": null,
    }))
}

#[cfg(any(test, target_os = "linux"))]
fn case_at(index: usize) -> Case {
    let projection = match index / 4 {
        0 => Projection::QueryKeyValue,
        1 => Projection::MlpGateUp,
        2 => Projection::AttentionOutput,
        3 => Projection::MlpDown,
        _ => unreachable!("case indices are validated within 0..16"),
    };
    let tokens = match (index % 4) / 2 {
        0 => 1,
        _ => 5,
    };
    let pattern = match index % 2 {
        0 => DensePattern::AlternatingUnit,
        _ => DensePattern::SignedMix,
    };
    Case {
        index,
        projection,
        tokens,
        pattern,
    }
}

#[cfg(test)]
mod tests {
    use super::{DensePattern, Projection, Q8ProjectionCaseRange};

    #[test]
    fn range_accepts_only_nonempty_indices_within_the_matrix() {
        assert!(Q8ProjectionCaseRange::new(0, 1).is_ok());
        assert!(Q8ProjectionCaseRange::new(15, 16).is_ok());
        assert!(Q8ProjectionCaseRange::new(0, 0).is_err());
        assert!(Q8ProjectionCaseRange::new(4, 3).is_err());
        assert!(Q8ProjectionCaseRange::new(0, 17).is_err());
        assert!(Q8ProjectionCaseRange::new(16, 17).is_err());
    }

    #[test]
    fn default_range_preserves_all_cartesian_indices_in_order() {
        let cases = Q8ProjectionCaseRange::all()
            .cases()
            .map(|case| (case.index, case.projection, case.tokens, case.pattern))
            .collect::<Vec<_>>();
        let expected = vec![
            (
                0,
                Projection::QueryKeyValue,
                1,
                DensePattern::AlternatingUnit,
            ),
            (1, Projection::QueryKeyValue, 1, DensePattern::SignedMix),
            (
                2,
                Projection::QueryKeyValue,
                5,
                DensePattern::AlternatingUnit,
            ),
            (3, Projection::QueryKeyValue, 5, DensePattern::SignedMix),
            (4, Projection::MlpGateUp, 1, DensePattern::AlternatingUnit),
            (5, Projection::MlpGateUp, 1, DensePattern::SignedMix),
            (6, Projection::MlpGateUp, 5, DensePattern::AlternatingUnit),
            (7, Projection::MlpGateUp, 5, DensePattern::SignedMix),
            (
                8,
                Projection::AttentionOutput,
                1,
                DensePattern::AlternatingUnit,
            ),
            (9, Projection::AttentionOutput, 1, DensePattern::SignedMix),
            (
                10,
                Projection::AttentionOutput,
                5,
                DensePattern::AlternatingUnit,
            ),
            (11, Projection::AttentionOutput, 5, DensePattern::SignedMix),
            (12, Projection::MlpDown, 1, DensePattern::AlternatingUnit),
            (13, Projection::MlpDown, 1, DensePattern::SignedMix),
            (14, Projection::MlpDown, 5, DensePattern::AlternatingUnit),
            (15, Projection::MlpDown, 5, DensePattern::SignedMix),
        ];

        assert_eq!(cases, expected);
    }

    #[test]
    fn bounded_range_keeps_exact_original_case_indices_and_coordinates() {
        let cases = Q8ProjectionCaseRange::new(5, 11)
            .expect("valid case range")
            .cases()
            .map(|case| (case.index, case.projection, case.tokens, case.pattern))
            .collect::<Vec<_>>();
        let expected = vec![
            (5, Projection::MlpGateUp, 1, DensePattern::SignedMix),
            (6, Projection::MlpGateUp, 5, DensePattern::AlternatingUnit),
            (7, Projection::MlpGateUp, 5, DensePattern::SignedMix),
            (
                8,
                Projection::AttentionOutput,
                1,
                DensePattern::AlternatingUnit,
            ),
            (9, Projection::AttentionOutput, 1, DensePattern::SignedMix),
            (
                10,
                Projection::AttentionOutput,
                5,
                DensePattern::AlternatingUnit,
            ),
        ];

        assert_eq!(cases, expected);
    }
}
