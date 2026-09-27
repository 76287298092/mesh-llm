use anyhow::{Result, ensure};
use serde_json::{Value, json};

const MAX_ELEMENTS: usize = 67_108_864;
const MAX_NORMALIZED_L2: f64 = 0.01;
const MIN_COSINE_SIMILARITY: f64 = 0.9999;

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }
}

/// Compare one complete layer's FP32 output with a reference vector.
///
/// The fixed acceptance gates are normalized L2 error at most 1% and cosine
/// similarity at least 0.9999. This measures a single-layer engineering budget;
/// it does not establish model or logit parity.
pub fn compare(actual: &[f32], expected: &[f32]) -> Result<Value> {
    ensure!(
        !actual.is_empty() && actual.len() <= MAX_ELEMENTS && actual.len() == expected.len(),
        "layer comparison extent mismatch"
    );
    ensure!(
        actual.iter().chain(expected).all(|value| value.is_finite()),
        "layer comparison values must be finite"
    );

    let mut error_energy = CompensatedSum::default();
    let mut actual_energy = CompensatedSum::default();
    let mut reference_energy = CompensatedSum::default();
    let mut dot = CompensatedSum::default();
    let mut max_abs_error = 0.0_f64;
    let mut exact_f32_elements = 0_usize;
    for (&actual_f32, &expected_f32) in actual.iter().zip(expected) {
        exact_f32_elements += usize::from(actual_f32.to_bits() == expected_f32.to_bits());
        let (actual_value, expected_value) = (f64::from(actual_f32), f64::from(expected_f32));
        let difference = actual_value - expected_value;
        error_energy.add(difference * difference);
        actual_energy.add(actual_value * actual_value);
        reference_energy.add(expected_value * expected_value);
        dot.add(actual_value * expected_value);
        max_abs_error = max_abs_error.max(difference.abs());
    }

    let error_energy = error_energy.sum.max(0.0);
    let actual_energy = actual_energy.sum.max(0.0);
    let reference_energy = reference_energy.sum.max(0.0);
    let (normalized_l2, cosine_similarity) = match (actual_energy > 0.0, reference_energy > 0.0) {
        (false, false) => (Some(0.0), 1.0),
        (true, false) => (None, 0.0),
        (false, true) => (Some(1.0), 0.0),
        (true, true) => {
            let normalized_l2 = (error_energy / reference_energy).sqrt();
            let denominator = actual_energy.sqrt() * reference_energy.sqrt();
            let cosine_similarity = (dot.sum / denominator).clamp(-1.0, 1.0);
            (Some(normalized_l2), cosine_similarity)
        }
    };
    let rms_error = (error_energy / actual.len() as f64).sqrt();
    let passed = normalized_l2.is_some_and(|value| value <= MAX_NORMALIZED_L2)
        && cosine_similarity >= MIN_COSINE_SIMILARITY;

    Ok(json!({
        "passed": passed,
        "elements": actual.len(),
        "normalized_l2": normalized_l2,
        "cosine_similarity": cosine_similarity,
        "max_abs_error": max_abs_error,
        "rms_error": rms_error,
        "exact_f32_elements": exact_f32_elements,
        "tolerance": {
            "normalized_l2_max": MAX_NORMALIZED_L2,
            "cosine_min": MIN_COSINE_SIMILARITY
        },
        "scope": "single-layer engineering error budget; not model/logit parity"
    }))
}

/// Compare aggregate layer output and equal-width independent partitions.
pub fn compare_partitioned(
    actual: &[f32],
    expected: &[f32],
    partition_width: usize,
) -> Result<Value> {
    ensure!(
        partition_width > 0,
        "layer comparison partition width must be positive"
    );
    ensure!(
        actual.len() == expected.len() && actual.len().is_multiple_of(partition_width),
        "layer comparison partition extent mismatch"
    );
    let aggregate = compare(actual, expected)?;
    let mut partitions = Vec::with_capacity(actual.len() / partition_width);
    for (actual_partition, expected_partition) in actual
        .chunks_exact(partition_width)
        .zip(expected.chunks_exact(partition_width))
    {
        partitions.push(compare(actual_partition, expected_partition)?);
    }
    let all_passed = aggregate["passed"] == true
        && partitions
            .iter()
            .all(|partition| partition["passed"] == true);
    Ok(json!({
        "all_passed": all_passed,
        "aggregate": aggregate,
        "partition_width": partition_width,
        "partitions": partitions
    }))
}

#[cfg(test)]
mod tests {
    use super::{compare, compare_partitioned};
    use serde_json::Value;

    fn metric(report: &Value, name: &str) -> f64 {
        report[name].as_f64().unwrap()
    }

    #[test]
    fn exact_signed_values_and_zero_vectors_report_bitwise_matches() {
        let exact = compare(&[1.0, -2.0, -0.0], &[1.0, -2.0, -0.0]).unwrap();
        assert_eq!(exact["passed"], true);
        assert_eq!(exact["exact_f32_elements"], 3);
        assert_eq!(metric(&exact, "normalized_l2"), 0.0);
        assert!((metric(&exact, "cosine_similarity") - 1.0).abs() <= 1e-15);

        let signed_zero = compare(&[0.0], &[-0.0]).unwrap();
        assert_eq!(signed_zero["passed"], true);
        assert_eq!(signed_zero["exact_f32_elements"], 0);
        assert_eq!(metric(&signed_zero, "max_abs_error"), 0.0);
    }

    #[test]
    fn zero_reference_is_explicit_and_zero_actual_has_unit_relative_error() {
        let both_zero = compare(&[-0.0, 0.0], &[0.0, -0.0]).unwrap();
        assert_eq!(both_zero["passed"], true);
        assert_eq!(metric(&both_zero, "normalized_l2"), 0.0);
        assert_eq!(metric(&both_zero, "cosine_similarity"), 1.0);

        let missing_reference = compare(&[1.0], &[0.0]).unwrap();
        assert_eq!(missing_reference["passed"], false);
        assert!(missing_reference["normalized_l2"].is_null());
        assert_eq!(metric(&missing_reference, "cosine_similarity"), 0.0);

        let missing_actual = compare(&[0.0], &[1.0]).unwrap();
        assert_eq!(missing_actual["passed"], false);
        assert_eq!(metric(&missing_actual, "normalized_l2"), 1.0);
        assert_eq!(metric(&missing_actual, "cosine_similarity"), 0.0);
    }

    #[test]
    fn half_percent_scaling_passes_but_two_percent_scaling_fails_l2_gate() {
        let expected = [-2.0, -1.0, 0.5, 4.0];
        let within: Vec<_> = expected.iter().map(|value| value * 1.005).collect();
        let within_report = compare(&within, &expected).unwrap();
        assert_eq!(within_report["passed"], true);
        assert!(metric(&within_report, "normalized_l2") <= 0.01);
        assert!(metric(&within_report, "cosine_similarity") >= 0.9999);

        let outside: Vec<_> = expected.iter().map(|value| value * 1.02).collect();
        let outside_report = compare(&outside, &expected).unwrap();
        assert_eq!(outside_report["passed"], false);
        assert!(metric(&outside_report, "normalized_l2") > 0.01);
        assert!(metric(&outside_report, "cosine_similarity") >= 0.9999);
    }

    #[test]
    fn orthogonal_and_single_element_corruption_fail() {
        let orthogonal = compare(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert_eq!(orthogonal["passed"], false);
        assert_eq!(metric(&orthogonal, "cosine_similarity"), 0.0);

        let expected = vec![1.0_f32; 256];
        let mut actual = expected.clone();
        actual[73] = 2.5;
        let corrupt = compare(&actual, &expected).unwrap();
        assert_eq!(corrupt["passed"], false);
        assert_eq!(corrupt["exact_f32_elements"], 255);
        assert_eq!(metric(&corrupt, "max_abs_error"), 1.5);
    }

    #[test]
    fn near_l2_budget_angle_stays_above_cosine_floor() {
        let report = compare(&[1.0, 0.0099], &[1.0, 0.0]).unwrap();
        assert!(metric(&report, "normalized_l2") <= 0.01);
        assert!(metric(&report, "cosine_similarity") > 0.9999);
        assert_eq!(report["passed"], true);
    }

    #[test]
    fn one_bad_partition_fails_when_the_aggregate_passes() {
        let expected = vec![1.0_f32; 1024];
        let mut actual = expected.clone();
        actual[517] += 0.02;
        let report = compare_partitioned(&actual, &expected, 1).unwrap();
        assert_eq!(report["aggregate"]["passed"], true);
        assert_eq!(report["partitions"][517]["passed"], false);
        assert_eq!(report["all_passed"], false);
    }

    #[test]
    fn rejects_mismatched_empty_and_nonfinite_vectors() {
        assert!(compare(&[], &[]).is_err());
        assert!(compare(&[1.0], &[1.0, 2.0]).is_err());
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f32::INFINITY]).is_err());
    }

    #[test]
    fn rejects_invalid_partition_width_and_extent() {
        assert!(compare_partitioned(&[1.0], &[1.0], 0).is_err());
        assert!(compare_partitioned(&[1.0, 2.0], &[1.0, 2.0], 3).is_err());
    }
}
