//! Independent CPU arithmetic reference for the native 5120 x 10240 Q8 FC.
//!
//! Arithmetic attribution: NInfer, Apache-2.0, pinned revision e31bc99b,
//! `src/ops/linear/q8/q8_a16_sliced_k_mma.cuh`, groups at lines 267-315
//! and reduction at lines 333-372. No device implementation is reproduced.
//! Scalar FP32 group summation does NOT establish tensor-core internal FP32
//! summation bit identity. The FP64 result and its mathematical error bound are
//! independent evidence, not permission to loosen any existing comparison gate.

use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context, ensure};
use std::collections::BTreeSet;

pub const OUTPUT_ROWS: usize = 5120;
pub const K: usize = 10240;
const GROUP: usize = 32;
const SPLITS: usize = 8;
const WARP_K: usize = 64;
const BLOCK_K: usize = 512;
const ITERATIONS: usize = 20;

pub struct Q8SlicedKReference {
    /// Token-major, then selected-row order, before BF16 rounding.
    pub scheduled_f32: Vec<f32>,
    pub output_bf16: Vec<u16>,
    pub mathematical_f64: Vec<f64>,
    /// Absolute FP64 summation error bound only, not a GPU acceptance budget.
    pub mathematical_error_bound: Vec<f64>,
}

/// Runs T1 or T5 without converting packed Q8 to a dense weight matrix.
///
/// Fixtures may contain 1..=5120 complete parent rows and select any unique
/// subset. Full FC verification supplies all 5120 parent rows. K, padding,
/// groups, and split geometry never shrink with the fixture row count.
/// Signed codes include -128, which the pinned arithmetic can represent even
/// though the canonical exporter uses only -127..=127.
///
/// # Errors
/// Rejects inconsistent planes, nonzero alignment padding, invalid row maps,
/// nonfinite activations, negative/nonfinite scales, and nonzero codes in a
/// zero-scale group. Rejects nonfinite scheduled intermediate/output values.
pub fn run(
    object: &[u8],
    view: &Q8MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<Q8SlicedKReference> {
    let scale_offset = validate(object, view, input_bf16)?;
    let capacity = input_bf16.len() / K * view.shape[0];
    let mut result = Q8SlicedKReference {
        scheduled_f32: Vec::with_capacity(capacity),
        output_bf16: Vec::with_capacity(capacity),
        mathematical_f64: Vec::with_capacity(capacity),
        mathematical_error_bound: Vec::with_capacity(capacity),
    };
    for token in input_bf16.as_chunks::<K>().0 {
        for &row in &view.source_rows {
            let mut partial = [0.0_f32; SPLITS];
            for iteration in 0..ITERATIONS {
                for (split, acc) in partial.iter_mut().enumerate() {
                    for group in 0..2 {
                        let start = iteration * BLOCK_K + split * WARP_K + group * GROUP;
                        let mut dot = 0.0_f32;
                        for lane in 0..GROUP {
                            let code = i8::from_ne_bytes([object[row * K + start + lane]]);
                            let activation = bf16(token[start + lane]);
                            dot = f32::from(code).mul_add(activation, dot);
                        }
                        ensure!(dot.is_finite(), "Q8 scalar group dot overflowed FP32");
                        let scale = scale_at(
                            object,
                            scale_offset + (row * (K / GROUP) + start / GROUP) * 2,
                        )?;
                        *acc = dot.mul_add(scale, *acc);
                        ensure!(acc.is_finite(), "Q8 split accumulator overflowed FP32");
                    }
                }
            }
            let p01 = partial[0] + partial[1];
            let p23 = partial[2] + partial[3];
            let p45 = partial[4] + partial[5];
            let p67 = partial[6] + partial[7];
            let sum = ((p01 + p23) + p45) + p67;
            ensure!(sum.is_finite(), "Q8 split reduction overflowed FP32");
            let mut mathematical = 0.0_f64;
            let mut absolute = 0.0_f64;
            for (k, &bits) in token.iter().enumerate() {
                let code = i8::from_ne_bytes([object[row * K + k]]);
                let scale = scale_at(object, scale_offset + (row * (K / GROUP) + k / GROUP) * 2)?;
                let term = f64::from(code) * f64::from(bf16(bits)) * f64::from(scale);
                mathematical += term;
                absolute += term.abs();
            }
            // Each term is exact in FP64. gamma bounds the K additions; the
            // second denominator covers rounding in the absolute-term sum.
            let operations = 10240.0_f64;
            let gamma =
                operations * (f64::EPSILON / 2.0) / (1.0 - operations * (f64::EPSILON / 2.0));
            result.scheduled_f32.push(sum);
            result.output_bf16.push(round_bf16(sum));
            result.mathematical_f64.push(mathematical);
            result
                .mathematical_error_bound
                .push(2.0 * gamma * absolute / (1.0 - gamma));
        }
    }
    Ok(result)
}

fn validate(object: &[u8], view: &Q8MatrixView, input: &[u16]) -> anyhow::Result<usize> {
    ensure!(
        (1..=OUTPUT_ROWS).contains(&view.shape[0]) && view.shape[1] == K,
        "Q8 FC shape is invalid"
    );
    ensure!(
        view.padded_k == K && view.group_size == GROUP,
        "Q8 FC K/group padding is invalid"
    );
    ensure!(
        input.len() == K || input.len() == 5 * K,
        "Q8 FC requires T1 or T5 input"
    );
    ensure!(
        input.iter().all(|&bits| bf16(bits).is_finite()),
        "Q8 FC activation is nonfinite"
    );
    ensure!(view.codes.offset == 0, "Q8 codes must begin at offset zero");
    let code_bytes = usize::try_from(view.codes.bytes).context("Q8 code extent exceeds usize")?;
    ensure!(
        code_bytes > 0 && code_bytes.is_multiple_of(K),
        "Q8 code rows are incomplete"
    );
    let parent_rows = code_bytes / K;
    ensure!(parent_rows <= OUTPUT_ROWS, "Q8 FC parent rows exceed 5120");
    let scale_count = parent_rows * (K / GROUP);
    ensure!(
        view.scale_count == scale_count,
        "Q8 scale count differs from geometry"
    );
    let scale_offset = code_bytes + (256 - code_bytes % 256) % 256;
    ensure!(
        usize::try_from(view.scale_bits.offset)? == scale_offset,
        "Q8 scale alignment is invalid"
    );
    ensure!(
        usize::try_from(view.scale_bits.bytes)? == scale_count * 2,
        "Q8 scale plane length is invalid"
    );
    ensure!(
        object.len() == scale_offset + scale_count * 2,
        "Q8 object length is invalid"
    );
    ensure!(
        object[code_bytes..scale_offset]
            .iter()
            .all(|&byte| byte == 0),
        "Q8 alignment padding is nonzero"
    );
    ensure!(
        view.source_rows.len() == view.shape[0],
        "Q8 selected row count is invalid"
    );
    ensure!(
        view.source_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == view.shape[0],
        "Q8 selected rows repeat"
    );
    ensure!(
        view.source_rows.iter().all(|&row| row < parent_rows),
        "Q8 selected row is absent"
    );
    for row in 0..parent_rows {
        for group in 0..K / GROUP {
            let scale = scale_at(object, scale_offset + (row * (K / GROUP) + group) * 2)?;
            let start = row * K + group * GROUP;
            ensure!(
                scale != 0.0 || object[start..start + GROUP].iter().all(|&code| code == 0),
                "zero Q8 scale requires zero codes"
            );
        }
    }
    Ok(scale_offset)
}

fn scale_at(object: &[u8], offset: usize) -> anyhow::Result<f32> {
    let bits = u16::from_le_bytes([object[offset], object[offset + 1]]);
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    ensure!(
        bits & 0x8000 == 0 && exponent != 31,
        "Q8 FP16 scale is negative or nonfinite"
    );
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    let bytes = rounded.to_le_bytes();
    u16::from_le_bytes([bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packages::qwen3_8_27b::native_mtp_views::BytePlane;

    fn fixture() -> (Vec<u8>, Q8MatrixView, Vec<u16>) {
        let mut object = vec![0; K + K / GROUP * 2];
        for pair in object[K..].as_chunks_mut::<2>().0 {
            pair.copy_from_slice(&0x3c00_u16.to_le_bytes());
        }
        let view = Q8MatrixView {
            object_id: "sliced-k-fixture".into(),
            shape: [1, K],
            padded_k: K,
            group_size: GROUP,
            codes: BytePlane {
                offset: 0,
                bytes: 10240,
            },
            scale_bits: BytePlane {
                offset: 10240,
                bytes: 640,
            },
            scale_count: K / GROUP,
            source_rows: vec![0],
        };
        (object, view, vec![0x3f80; K])
    }

    #[test]
    fn signed_extremes_when_neighbor_scales_differ() {
        let (mut object, view, input) = fixture();
        object[0] = 0x80;
        object[32] = 0x7f;
        object[K + 2..K + 4].copy_from_slice(&0x4000_u16.to_le_bytes());
        let output = run(&object, &view, &input).expect("valid signed Q8");
        assert_eq!(output.scheduled_f32, [126.0]);
        assert_eq!(output.mathematical_f64, [126.0]);
        assert_eq!(output.output_bf16, [0x42fc]);
    }

    #[test]
    fn pair_reduction_when_all_eight_splits_cancel() {
        let (mut object, view, mut input) = fixture();
        for (split, bits) in [0x4b80, 0x3f80, 0xcb80, 0, 0x4040, 0, 0xc000, 0]
            .into_iter()
            .enumerate()
        {
            object[split * WARP_K] = 1;
            input[split * WARP_K] = bits;
        }
        let output = run(&object, &view, &input).expect("valid cancellation fixture");
        assert_eq!(output.scheduled_f32, [1.0]);
        assert_eq!(output.mathematical_f64, [2.0]);
        assert_eq!(output.output_bf16, [0x3f80]);
        assert!(output.mathematical_error_bound[0] < 0.001);
    }

    #[test]
    fn distinct_tokens_when_t1_and_t5_use_the_last_iteration() {
        let (mut object, view, mut input) = fixture();
        object[K - 1] = 1;
        input[K - 1] = 0x3f00;
        let single = run(&object, &view, &input).expect("T1");
        let mut batch = vec![0; 5 * K];
        for (token, bits) in [0x3f00, 0x3f80, 0xbf80, 0x4000, 0xc000]
            .into_iter()
            .enumerate()
        {
            batch[token * K + K - 1] = bits;
        }
        let output = run(&object, &view, &batch).expect("T5");
        assert_eq!(output.output_bf16, [0x3f00, 0x3f80, 0xbf80, 0x4000, 0xc000]);
        assert_eq!(single.output_bf16[0], output.output_bf16[0]);
    }

    #[test]
    fn bf16_even_rounding_when_fc_results_are_ties() {
        let (mut object, view, mut input) = fixture();
        object[0] = 1;
        object[1] = 1;
        input[1] = 0x3b80;
        let even = run(&object, &view, &input).expect("even tie");
        assert_eq!(even.output_bf16, [0x3f80]);
        input[0] = 0x3f81;
        let odd = run(&object, &view, &input).expect("odd tie");
        assert_eq!(odd.output_bf16, [0x3f82]);
    }

    #[test]
    fn group_order_when_iterations_cancel_a_small_neighbor() {
        let (mut object, view, mut input) = fixture();
        for (k, bits) in [(0, 0x4b80), (32, 0x3f80), (512, 0xcb80)] {
            object[k] = 1;
            input[k] = bits;
        }
        let output = run(&object, &view, &input).expect("ordered groups");
        assert_eq!(output.scheduled_f32, [0.0]);
        assert_eq!(output.mathematical_f64, [1.0]);
    }

    #[test]
    fn checked_boundary_when_planes_scales_or_inputs_are_invalid() {
        let (object, view, input) = fixture();
        assert!(run(&object[..object.len() - 1], &view, &input).is_err());
        assert!(run(&object, &view, &input[..K - 1]).is_err());
        let mut invalid_input = input.clone();
        invalid_input[0] = 0x7f80;
        assert!(run(&object, &view, &invalid_input).is_err());
        let mut zero_scale = object.clone();
        zero_scale[K..K + 2].fill(0);
        zero_scale[0] = 1;
        assert!(run(&zero_scale, &view, &input).is_err());
        for bits in [0x8000_u16, 0x7c00, 0x7e00] {
            let mut invalid = object.clone();
            invalid[K..K + 2].copy_from_slice(&bits.to_le_bytes());
            assert!(run(&invalid, &view, &input).is_err());
        }
    }
}
