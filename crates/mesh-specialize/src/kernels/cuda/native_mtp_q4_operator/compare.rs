use anyhow::{Result, ensure};
use serde_json::{Value, json};

use super::launch::Output;
use crate::native_mtp_q4_gemv_reference::Q4ProjectionReference;

const MAX_SCALED_ERROR: f64 = 2.0e-4;

pub(super) fn run(name: &str, actual: &Output, expected: &Q4ProjectionReference) -> Result<Value> {
    ensure!(
        actual.raw_f32.len() == expected.raw_f64.len()
            && actual.logits_bf16.len() == expected.raw_f64.len()
            && expected.logits_bf16.len() == expected.raw_f64.len(),
        "native Q4 output counts differ from selected rows"
    );
    let mut max_abs_error = 0.0_f64;
    let mut max_scaled_error = 0.0_f64;
    let mut bf16_mismatches = 0_usize;
    for ((&actual_f32, &actual_bf16), (&expected_f64, &expected_bf16)) in actual
        .raw_f32
        .iter()
        .zip(&actual.logits_bf16)
        .zip(expected.raw_f64.iter().zip(&expected.logits_bf16))
    {
        ensure!(
            actual_f32.is_finite() && expected_f64.is_finite(),
            "native Q4 head produced a nonfinite value"
        );
        let error = (f64::from(actual_f32) - expected_f64).abs();
        max_abs_error = max_abs_error.max(error);
        max_scaled_error = max_scaled_error.max(error / expected_f64.abs().max(1.0));
        if actual_bf16 != expected_bf16 {
            bf16_mismatches += 1;
        }
    }
    let selected_row = first_argmax(&actual.logits_bf16)?;
    let scaled_error_passed = max_scaled_error <= MAX_SCALED_ERROR;
    let output_count = actual.raw_f32.len();
    let max_scaled_error_limit = MAX_SCALED_ERROR;
    Ok(json!({
        "case": name,
        "outputs": output_count,
        "max_abs_error": max_abs_error,
        "max_scaled_error": max_scaled_error,
        "max_scaled_error_limit": max_scaled_error_limit,
        "bf16_mismatches_to_fp64_oracle_rounding": bf16_mismatches,
        "fp64_scaled_error_passed": scaled_error_passed,
        "selected_proposal_row": selected_row,
        "all_passed": scaled_error_passed,
    }))
}

pub(super) fn first_argmax(logits: &[u16]) -> Result<usize> {
    ensure!(!logits.is_empty(), "native Q4 proposal logits are empty");
    let mut best_row = 0;
    let mut best_value = f32::from_bits(u32::from(logits[0]) << 16);
    ensure!(
        best_value.is_finite(),
        "native Q4 proposal logit is nonfinite"
    );
    for (row, &bits) in logits.iter().enumerate().skip(1) {
        let value = f32::from_bits(u32::from(bits) << 16);
        ensure!(value.is_finite(), "native Q4 proposal logit is nonfinite");
        if value > best_value {
            best_row = row;
            best_value = value;
        }
    }
    Ok(best_row)
}
