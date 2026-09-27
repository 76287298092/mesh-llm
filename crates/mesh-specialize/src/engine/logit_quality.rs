//! Distribution diagnostics for identical teacher-forced input prefixes.
//! These measurements deliberately do not define a model-quality pass threshold.
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub fn compare(candidate: &[u16], control: &[u16]) -> Result<Value> {
    ensure!(
        !candidate.is_empty() && candidate.len() == control.len(),
        "logit extent mismatch"
    );
    let candidate = decode(candidate)?;
    let control = decode(control)?;
    let c_log_z = log_partition(&candidate);
    let r_log_z = log_partition(&control);
    let c_top = top(&candidate);
    let r_top = top(&control);
    let mut kl = 0.0;
    let mut total_variation = 0.0;
    for (&c, &r) in candidate.iter().zip(&control) {
        let log_p = r - r_log_z;
        let log_q = c - c_log_z;
        let p = log_p.exp();
        kl += p * (log_p - log_q);
        total_variation += (p - log_q.exp()).abs();
    }
    Ok(json!({
        "diagnostic_only":true,
        "kl_control_to_candidate_nats":kl.max(0.0),
        "total_variation":total_variation * 0.5,
        "greedy_agreement":c_top == r_top,
        "control_top_token":r_top,
        "candidate_top_token":c_top,
        "control_top_probability":(control[r_top]-r_log_z).exp(),
        "candidate_probability_of_control_top":(candidate[r_top]-c_log_z).exp(),
        "control_top_margin":margin(&control, r_top),
        "candidate_top_margin":margin(&candidate, c_top),
    }))
}
fn decode(bits: &[u16]) -> Result<Vec<f64>> {
    let values = bits
        .iter()
        .map(|&v| f64::from(f32::from_bits(u32::from(v) << 16)))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite logits");
    Ok(values)
}
fn log_partition(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}
fn top(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
fn margin(values: &[f64], best: usize) -> Option<f64> {
    (values.len() > 1).then(|| {
        values[best]
            - values
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != best)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max)
    })
}
#[cfg(test)]
mod tests {
    use super::compare;
    fn bits(values: &[f32]) -> Vec<u16> {
        values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
    }
    #[test]
    fn identical_and_shifted_distributions() {
        let result = compare(&bits(&[1.0, 2.0]), &bits(&[0.0, 1.0])).unwrap();
        assert!(result["kl_control_to_candidate_nats"].as_f64().unwrap() < 1e-14);
        assert!(result["total_variation"].as_f64().unwrap() < 1e-14);
        assert_eq!(result["greedy_agreement"], true);
    }
    #[test]
    fn reversed_logits_match_analytic_binary_distribution() {
        let result = compare(&bits(&[1.0, 0.0]), &bits(&[0.0, 1.0])).unwrap();
        let expected = (1.0_f64.exp() - 1.0) / (1.0_f64.exp() + 1.0);
        assert!(
            (result["kl_control_to_candidate_nats"].as_f64().unwrap() - expected).abs() < 1e-14
        );
        assert!((result["total_variation"].as_f64().unwrap() - expected).abs() < 1e-14);
        assert_eq!(result["greedy_agreement"], false);
        assert!(compare(&[0x7f80], &[0]).is_err());
        assert!(compare(&[], &[]).is_err());
    }
}
