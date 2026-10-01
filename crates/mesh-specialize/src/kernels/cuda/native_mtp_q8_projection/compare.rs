use super::reference::ProjectionReference;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn compare(
    expected: &ProjectionReference,
    outputs: &[Vec<u16>; 2],
    rows: usize,
    tokens: usize,
) -> Result<Value> {
    let count = rows
        .checked_mul(tokens)
        .ok_or_else(|| anyhow::anyhow!("Q8 comparison extent overflow"))?;
    ensure!(rows > 0, "Q8 projection comparison has no output rows");
    ensure!(
        expected.output_bf16.len() == count && outputs.iter().all(|output| output.len() == count),
        "Q8 projection output/reference extent mismatch"
    );
    let mut nonfinite = 0_usize;
    let mut repeat_mismatches = 0_usize;
    let mut exact_mismatches = 0_usize;
    let mut failures = Vec::new();
    for (index, (&first, &second)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
        nonfinite += usize::from(first & 0x7f80 == 0x7f80) + usize::from(second & 0x7f80 == 0x7f80);
        repeat_mismatches += usize::from(first != second);
        let oracle = expected.output_bf16[index];
        exact_mismatches += usize::from(first != oracle) + usize::from(second != oracle);
        if (first != oracle || second != oracle) && failures.len() < 16 {
            failures.push(json!({
                "token": index / rows,
                "row": index % rows,
                "expected_bf16": oracle,
                "observed_bf16": [first, second],
            }));
        }
    }
    Ok(json!({
        "all_passed": nonfinite == 0 && repeat_mismatches == 0 && exact_mismatches == 0,
        "outputs_per_repeat": count,
        "repeats": outputs.len(),
        "nonfinite_outputs": nonfinite,
        "repeat_mismatches": repeat_mismatches,
        "exact_bf16_mismatches": exact_mismatches,
        "failures_first_16": failures,
        "mathematical_f64_diagnostic_only": true,
        "mathematical_error_bound_is_gpu_tolerance": false,
    }))
}

#[cfg(test)]
mod tests {
    use super::compare;
    use crate::kernels::cuda::native_mtp_q8_projection::reference::ProjectionReference;

    #[test]
    fn all_row_gate_when_last_output_is_poisoned_rejects_it() {
        let expected = expected(4 * 5);
        let first = vec![0x3f80; 20];
        let mut second = first.clone();
        second[19] = 0x7fc1;

        let report = compare(&expected, &[first, second], 4, 5).expect("comparison report");

        assert_eq!(report["all_passed"], false);
        assert_eq!(report["nonfinite_outputs"], 1);
        assert_eq!(report["repeat_mismatches"], 1);
        assert_eq!(report["exact_bf16_mismatches"], 1);
        assert_eq!(report["failures_first_16"][0]["token"], 4);
        assert_eq!(report["failures_first_16"][0]["row"], 3);
    }

    #[test]
    fn all_row_gate_when_both_repeats_share_wrong_last_word_rejects_it() {
        let expected = expected(4 * 5);
        let mut first = vec![0x3f80; 20];
        let mut second = first.clone();
        first[19] = 0x3f81;
        second[19] = 0x3f81;

        let report = compare(&expected, &[first, second], 4, 5).expect("comparison report");

        assert_eq!(report["all_passed"], false);
        assert_eq!(report["repeat_mismatches"], 0);
        assert_eq!(report["exact_bf16_mismatches"], 2);
    }

    #[test]
    fn mismatch_report_when_many_rows_fail_caps_the_report_at_sixteen() {
        let expected = expected(20);
        let first = vec![0x3f81; 20];
        let second = vec![0x3f81; 20];

        let report = compare(&expected, &[first, second], 20, 1).expect("comparison report");

        assert_eq!(report["all_passed"], false);
        assert_eq!(report["exact_bf16_mismatches"], 40);
        assert_eq!(
            report["failures_first_16"].as_array().map(Vec::len),
            Some(16)
        );
        assert_eq!(report["failures_first_16"][0]["row"], 0);
        assert_eq!(report["failures_first_16"][15]["row"], 15);
    }

    fn expected(count: usize) -> ProjectionReference {
        ProjectionReference {
            scheduled_f32: vec![1.0; count],
            output_bf16: vec![0x3f80; count],
            mathematical_f64: vec![1.0; count],
            mathematical_error_bound: vec![0.0; count],
        }
    }
}
