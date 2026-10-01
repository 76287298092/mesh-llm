//! Independent FP64 log-softmax oracle for teacher-forced scoring rows.
//!
//! For one row of BF16 logits this computes, entirely in FP64 with the host
//! `exp`/`ln`: the log-sum-exp, the target log-probability and the top-64
//! `(id, logprob)` pairs ordered by descending logit with the lower id first on
//! ties. Positive and negative zero compare equal, so they tie by id. It shares
//! no code with `kernels/nvptx/row_logprob_topk.rs`.

/// Number of retained candidates per scored row.
pub const TOP_K: usize = 64;

/// One scored row in full FP64 precision.
#[derive(Clone, Debug, PartialEq)]
pub struct RowScore {
    pub target: u32,
    pub target_logprob: f64,
    pub logsumexp: f64,
    pub top_ids: Vec<u32>,
    pub top_logprobs: Vec<f64>,
}

/// Widen a BF16 bit pattern exactly.
pub fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// Score one row. `top_ids` holds `min(64, logits.len())` entries.
pub fn score_row(logits: &[u16], target: u32) -> Result<RowScore, String> {
    if logits.is_empty() {
        return Err("empty logit row".into());
    }
    let target_index = target as usize;
    if target_index >= logits.len() {
        return Err(format!(
            "target {target} outside vocabulary {}",
            logits.len()
        ));
    }
    let values = logits
        .iter()
        .map(|&bits| bf16_to_f64(bits))
        .collect::<Vec<_>>();
    if let Some(position) = values.iter().position(|value| !value.is_finite()) {
        return Err(format!("nonfinite logit at {position}"));
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = values
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    let log_sum = sum.ln();
    let logprob = |value: f64| (value - maximum) - log_sum;

    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        values[right]
            .partial_cmp(&values[left])
            .expect("finite logits are ordered")
            .then(left.cmp(&right))
    });
    order.truncate(TOP_K.min(values.len()));
    Ok(RowScore {
        target,
        target_logprob: logprob(values[target_index]),
        logsumexp: maximum + log_sum,
        top_ids: order.iter().map(|&index| index as u32).collect(),
        top_logprobs: order.iter().map(|&index| logprob(values[index])).collect(),
    })
}

/// Absolute tolerance used when comparing a device FP32 value with this oracle.
///
/// The device uses FP32 exponentials with FP64 accumulation and stores FP32,
/// so the bound is a small absolute term plus a few FP32 ulps of the value.
pub fn tolerance(reference: f64) -> f64 {
    1.0e-5 + 4.0e-7 * reference.abs()
}

/// Compare device outputs against this oracle. Ids must match exactly.
pub fn matches_device(
    reference: &RowScore,
    target_logprob: f32,
    logsumexp: f32,
    top_ids: &[u32],
    top_logprobs: &[f32],
) -> Result<f64, String> {
    if top_ids.len() < reference.top_ids.len() || top_logprobs.len() < reference.top_ids.len() {
        return Err("device top-k extent is short".into());
    }
    if top_ids[..reference.top_ids.len()] != reference.top_ids[..] {
        return Err("device top-k ids differ from the FP64 oracle".into());
    }
    let mut worst = 0.0_f64;
    let pairs = [
        (f64::from(target_logprob), reference.target_logprob),
        (f64::from(logsumexp), reference.logsumexp),
    ]
    .into_iter()
    .chain(
        top_logprobs
            .iter()
            .zip(&reference.top_logprobs)
            .map(|(&device, &oracle)| (f64::from(device), oracle)),
    );
    for (device, oracle) in pairs {
        let error = (device - oracle).abs();
        if error.is_nan() || error > tolerance(oracle) {
            return Err(format!(
                "device value {device:e} differs from oracle {oracle:e} by {error:e}"
            ));
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::{TOP_K, bf16_to_f64, matches_device, score_row};

    const ONE: u16 = 0x3f80;
    const TWO: u16 = 0x4000;
    const THREE: u16 = 0x4040;
    const MAX_FINITE: u16 = 0x7f7f;
    const MINUS_HUNDRED: u16 = 0xc2c8;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn two_equal_logits_split_mass_and_tie_by_id() {
        let score = score_row(&[0, 0], 1).unwrap();
        close(score.logsumexp, 2.0_f64.ln());
        close(score.target_logprob, -(2.0_f64.ln()));
        assert_eq!(score.top_ids, [0, 1]);
        close(score.top_logprobs[0], -(2.0_f64.ln()));
    }

    #[test]
    fn ties_order_lower_id_first_and_merge_zero_signs() {
        let score = score_row(&[ONE, THREE, THREE, ONE, TWO], 4).unwrap();
        assert_eq!(score.top_ids, [1, 2, 4, 0, 3]);
        let lse = 3.0 + (2.0 + (-1.0_f64).exp() + 2.0 * (-2.0_f64).exp()).ln();
        close(score.logsumexp, lse);
        close(score.target_logprob, 2.0 - lse);
        let zeros = score_row(&[0x8000, 0x0000, 0x8000], 0).unwrap();
        assert_eq!(zeros.top_ids, [0, 1, 2]);
    }

    #[test]
    fn extreme_finite_logits_stay_finite() {
        let score = score_row(&[0x0000, MAX_FINITE, MINUS_HUNDRED], 0).unwrap();
        let maximum = bf16_to_f64(MAX_FINITE);
        assert_eq!(score.logsumexp, maximum);
        assert_eq!(score.top_ids, [1, 0, 2]);
        assert_eq!(score.top_logprobs[0], 0.0);
        assert_eq!(score.target_logprob, -maximum);
        assert!((score.target_logprob as f32).is_finite());
        assert!(score.top_logprobs.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn target_outside_top_k_is_still_scored() {
        // Logit i is i/8 for ids 1..100; id 0 carries -100 and is the target.
        let mut logits = (0..100_u32)
            .map(|index| ((index as f32) / 8.0).to_bits() >> 16)
            .map(|bits| bits as u16)
            .collect::<Vec<_>>();
        logits[0] = MINUS_HUNDRED;
        let score = score_row(&logits, 0).unwrap();
        assert_eq!(score.top_ids.len(), TOP_K);
        assert!(!score.top_ids.contains(&0));
        assert_eq!(score.top_ids[0], 99);
        assert_eq!(score.top_ids[TOP_K - 1], 36);
        let sum: f64 =
            (1..100).map(|i| (f64::from(i) / 8.0).exp()).sum::<f64>() + (-100.0_f64).exp();
        close(score.logsumexp, sum.ln());
        close(score.target_logprob, -100.0 - sum.ln());
    }

    #[test]
    fn rejects_nonfinite_rows_and_bad_targets() {
        assert!(score_row(&[ONE, 0x7f80], 0).is_err());
        assert!(score_row(&[ONE, 0x7fc0], 0).is_err());
        assert!(score_row(&[ONE], 1).is_err());
        assert!(score_row(&[], 0).is_err());
    }

    #[test]
    fn device_comparison_requires_exact_ids_and_bounded_values() {
        let score = score_row(&[ONE, THREE, TWO], 2).unwrap();
        let lps = score
            .top_logprobs
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<_>>();
        let target = score.target_logprob as f32;
        let lse = score.logsumexp as f32;
        assert!(matches_device(&score, target, lse, &[1, 2, 0], &lps).is_ok());
        assert!(matches_device(&score, target, lse, &[2, 1, 0], &lps).is_err());
        assert!(matches_device(&score, target + 1.0e-3, lse, &[1, 2, 0], &lps).is_err());
    }
}
