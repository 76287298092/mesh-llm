use anyhow::{Result, ensure};
use serde_json::{Value, json};

const MAX_SCALED_ERROR: f64 = 1.0e-4;

pub(super) fn run(name: &str, output_bytes: &[u8], expected: &[f64]) -> Result<Value> {
    ensure!(output_bytes.len() == expected.len() * 4, "native Q8 output byte count differs from N");
    let actual = output_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let mut max_abs_error = 0.0_f64;
    let mut max_scaled_error = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        ensure!(actual.is_finite() && expected.is_finite(), "native Q8 GEMV produced a nonfinite result");
        let error = (f64::from(actual) - expected).abs();
        max_abs_error = max_abs_error.max(error);
        max_scaled_error = max_scaled_error.max(error / expected.abs().max(1.0));
    }
    Ok(json!({
        "case": name,
        "outputs": actual.len(),
        "max_abs_error": max_abs_error,
        "max_scaled_error": max_scaled_error,
        "max_scaled_error_limit": MAX_SCALED_ERROR,
        "all_passed": max_scaled_error <= MAX_SCALED_ERROR,
    }))
}
