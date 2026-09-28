//! Independent oracle and predeclared acceptance gates for BF16 split attention.
//! Calls the existing complete-logical-dot FP64 oracle, never split/GPU arithmetic.
use crate::{
    attention_online_reference as oracle,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub const HEAD_RELATIVE_L2: f64 = 1.0e-3;

pub fn run(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    rows: usize,
    past: usize,
    capacity: usize,
) -> Result<oracle::Attention> {
    ensure!((1..=8).contains(&rows), "split oracle supports M=1..8");
    oracle::run(
        q,
        k,
        v,
        &oracle::Shape {
            rows,
            past,
            capacity,
            query_heads: 24,
            kv_heads: 4,
            width: 256,
            scale: 0.0625,
        },
    )
}

/// Keep the established component gate, add a per-row/head raw FP32 L2 gate,
/// and verify BF16 is exactly RNE(raw). BF16 quantization alone can exceed 1e-3,
/// so the raw L2 threshold is not incorrectly applied to quantized output.
pub fn compare(raw: &[f32], bits: &[u16], expected: &oracle::Attention) -> Result<Value> {
    ensure!(
        raw.len() == expected.unrounded.len()
            && bits.len() == raw.len()
            && expected.output.len() == raw.len()
            && expected.value_bounds.len() == raw.len()
            && !raw.is_empty()
            && raw.len().is_multiple_of(256),
        "attention output extent mismatch"
    );
    let mut failures = 0;
    let mut rounding_failures = 0;
    let mut max_abs = 0.0_f32;
    let mut max_l2 = 0.0_f64;
    let mut head_failures = 0;
    let mut bf16_differences = 0;
    for head in 0..raw.len() / 256 {
        let mut square_error = 0.0_f64;
        let mut square_reference = 0.0_f64;
        for channel in 0..256 {
            let i = head * 256 + channel;
            let reference = expected.unrounded[i];
            let error = (raw[i] - reference).abs();
            let budget = oracle::COMPONENT_ABS_BUDGET
                + oracle::COMPONENT_REL_BUDGET * expected.value_bounds[i];
            if !raw[i].is_finite() || error > budget {
                failures += 1;
            }
            if !bf16_to_f32(bits[i]).is_finite() || bits[i] != round_bf16(raw[i]) {
                rounding_failures += 1;
            }
            if bits[i] != expected.output[i] {
                bf16_differences += 1;
            }
            max_abs = max_abs.max(error);
            square_error += f64::from(error).powi(2);
            square_reference += f64::from(reference).powi(2);
        }
        // Near-zero heads are covered by the absolute component gate instead.
        let l2 = square_error.sqrt() / square_reference.sqrt().max(1.0e-6);
        if square_reference.sqrt() >= 1.0e-6 && (!l2.is_finite() || l2 > HEAD_RELATIVE_L2) {
            head_failures += 1;
        }
        if l2.is_finite() {
            max_l2 = max_l2.max(l2);
        }
    }
    Ok(
        json!({"all_passed":failures==0 && rounding_failures==0 && head_failures==0,
        "component_failures":failures,"rounding_failures":rounding_failures,"head_l2_failures":head_failures,
        "max_abs_error":max_abs,"max_head_relative_l2":max_l2,"bf16_differences":bf16_differences,
        "component_abs_budget":oracle::COMPONENT_ABS_BUDGET,"component_value_relative_budget":oracle::COMPONENT_REL_BUDGET,
        "head_relative_l2_budget":HEAD_RELATIVE_L2}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_gqa_causal_tail_has_hand_computed_means() {
        let rows = 3;
        let past = 1;
        let capacity = 7;
        let q = vec![0; rows * 24 * 256];
        let mut k = vec![0x7fc0; capacity * 4 * 256];
        let mut v = k.clone();
        for token in 0..past + rows {
            for head in 0..4 {
                for c in 0..256 {
                    let i = (token * 4 + head) * 256 + c;
                    k[i] = 0;
                    v[i] = round_bf16((token * 2 + head * 16) as f32);
                }
            }
        }
        let result = run(&q, &k, &v, rows, past, capacity).unwrap();
        for row in 0..rows {
            for head in 0..24 {
                let expected = (past + row + head / 6 * 16) as f32;
                for c in 0..256 {
                    assert_eq!(result.unrounded[(row * 24 + head) * 256 + c], expected);
                }
            }
        }
        assert_eq!(
            compare(&result.unrounded, &result.output, &result).unwrap()["all_passed"],
            true
        );
    }

    #[test]
    fn single_key_and_nonfinite_results() {
        let q = vec![round_bf16(2.0); 24 * 256];
        let k = vec![round_bf16(-3.0); 4 * 256];
        let v = vec![round_bf16(-0.5); 4 * 256];
        let result = run(&q, &k, &v, 1, 0, 1).unwrap();
        assert!(result.unrounded.iter().all(|&x| x == -0.5));
        let mut raw = result.unrounded.clone();
        raw[0] = f32::NAN;
        assert_eq!(
            compare(&raw, &result.output, &result).unwrap()["all_passed"],
            false
        );
        assert!(run(&q, &k, &v, 9, 0, 9).is_err());
    }
}
