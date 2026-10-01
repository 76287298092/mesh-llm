use crate::entry_reference::round_bf16;

#[derive(serde::Serialize)]
pub(super) struct ErrorMetrics {
    pub(super) relative_l2: f64,
    pub(super) max_scaled_error: f64,
}

pub(super) fn compare_values(
    actual: &[f32],
    expected: &[f64],
    absolute_sums: &[f64],
) -> ErrorMetrics {
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (f64::from(left) - right).powi(2))
        .sum::<f64>();
    let squared_norm = expected.iter().map(|value| value * value).sum::<f64>();
    let maximum = actual
        .iter()
        .zip(expected)
        .zip(absolute_sums)
        .map(|((&left, &right), &absolute)| (f64::from(left) - right).abs() / absolute.max(1.0))
        .fold(0.0_f64, f64::max);
    ErrorMetrics {
        relative_l2: (squared_error / squared_norm.max(1.0e-30)).sqrt(),
        max_scaled_error: maximum,
    }
}

pub(super) fn exact_rounding(raw: &[f32], rounded: &[u16]) -> bool {
    raw.iter()
        .zip(rounded)
        .all(|(&value, &bits)| round_bf16(value) == bits)
}

pub(super) fn fp32_l2(actual: &[f32], expected: &[f32]) -> f64 {
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (f64::from(left) - f64::from(right)).powi(2))
        .sum::<f64>();
    let squared_norm = expected
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    (squared_error / squared_norm.max(1.0e-30)).sqrt()
}
