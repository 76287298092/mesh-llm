//! SPDX-License-Identifier: Apache-2.0
//! Independent CPU reference for bounded native MTP Q8 identity-row projections.
//!
//! Arithmetic source: NInfer contributors, Apache-2.0, pinned revision
//! e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d,
//! `src/ops/linear/q8/q8_a16_sliced_k_mma.cuh`, groups at lines 267-315 and split
//! reduction at lines 331-372. This implementation decodes logical bytes directly
//! and uses no GPU packing or execution helpers.

use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context, ensure};

const GROUP: usize = 32;
const WARP_K: usize = 64;

pub struct ProjectionReference {
    pub scheduled_f32: Vec<f32>,
    pub output_bf16: Vec<u16>,
    pub mathematical_f64: Vec<f64>,
    pub mathematical_error_bound: Vec<f64>,
}

pub struct ProjectionReferenceRequest<'a> {
    pub parent: &'a [u8],
    pub view: &'a Q8MatrixView,
    pub input_bf16: &'a [u16],
    pub split_warps: usize,
}

/// Apply the source split-K FP32 schedule to complete identity-row parent data.
///
/// `split_warps` is four or eight, matching the MTP C4/C8 physical schedules.
/// Inputs are BF16-exact dyadics with magnitude at most two. Signed Q8 codes have
/// magnitude at most 128, so `32 * 128 * 2 = 8192` bounds each G32 dot in FP32.
/// FP64 results and error bounds are diagnostics only, never GPU tolerances.
///
/// # Errors
/// Rejects invalid identity views, packed planes, scales, input extents, nonfinite
/// inputs, and schedules that do not divide K into complete 64-wide warp slices.
pub fn run(request: ProjectionReferenceRequest<'_>) -> anyhow::Result<ProjectionReference> {
    let ProjectionReferenceRequest {
        parent,
        view,
        input_bf16,
        split_warps,
    } = request;
    let (rows, k, scale_offset, tokens) = validate(parent, view, input_bf16, split_warps)?;
    let count = rows
        .checked_mul(tokens)
        .context("Q8 projection result extent overflow")?;
    let mut result = ProjectionReference {
        scheduled_f32: Vec::with_capacity(count),
        output_bf16: Vec::with_capacity(count),
        mathematical_f64: Vec::with_capacity(count),
        mathematical_error_bound: Vec::with_capacity(count),
    };
    let block_k = split_warps * WARP_K;
    for token in input_bf16.chunks_exact(k) {
        for row in 0..rows {
            let mut partials = [0.0_f32; 8];
            for iteration_k in (0..k).step_by(block_k) {
                for (split, partial) in partials.iter_mut().enumerate().take(split_warps) {
                    let slice_k = iteration_k + split * WARP_K;
                    for group in 0..2 {
                        let group_start = slice_k + group * GROUP;
                        let mut dot = 0.0_f32;
                        for lane in 0..GROUP {
                            let code = i8::from_ne_bytes([parent[row * k + group_start + lane]]);
                            let activation = bf16(token[group_start + lane]);
                            dot = f32::from(code).mul_add(activation, dot);
                        }
                        let scale_index = row * (k / GROUP) + group_start / GROUP;
                        let scale = half(parent, scale_offset + scale_index * 2)?;
                        *partial = dot.mul_add(scale, *partial);
                    }
                }
            }
            let mut sum = partials[0] + partials[1];
            let mut split = 2;
            while split < split_warps {
                sum += partials[split] + partials[split + 1];
                split += 2;
            }
            ensure!(
                sum.is_finite(),
                "Q8 scheduled projection produced a nonfinite value"
            );
            let mut mathematical = 0.0_f64;
            let mut absolute_sum = 0.0_f64;
            for (column, &activation_bits) in token.iter().enumerate() {
                let code = i8::from_ne_bytes([parent[row * k + column]]);
                let scale_index = row * (k / GROUP) + column / GROUP;
                let scale = f64::from(half(parent, scale_offset + scale_index * 2)?);
                let term = f64::from(code) * f64::from(bf16(activation_bits)) * scale;
                mathematical += term;
                absolute_sum += term.abs();
            }
            result.scheduled_f32.push(sum);
            result.output_bf16.push(round_bf16(sum));
            result.mathematical_f64.push(mathematical);
            result
                .mathematical_error_bound
                .push(error_bound(f64::from(u32::try_from(k)?), absolute_sum));
        }
    }
    Ok(result)
}

fn validate(
    parent: &[u8],
    view: &Q8MatrixView,
    input: &[u16],
    split_warps: usize,
) -> anyhow::Result<(usize, usize, usize, usize)> {
    let [rows, k] = view.shape;
    ensure!(
        rows > 0 && k > 0 && view.padded_k == k && view.group_size == GROUP,
        "Q8 projection shape or group width is invalid"
    );
    ensure!(
        matches!(split_warps, 4 | 8) && k.is_multiple_of(split_warps * WARP_K),
        "Q8 projection K does not fit the selected sliced-K schedule"
    );
    ensure!(
        input.len().is_multiple_of(k) && matches!(input.len() / k, 1 | 5),
        "Q8 projection requires T1 or T5 complete inputs"
    );
    ensure!(
        input.iter().all(|&word| matches!(
            word,
            0x0000 | 0x8000 | 0x3f80 | 0xbf80 | 0x3f00 | 0xbf00 | 0x4000 | 0xc000
        )),
        "Q8 projection activation is outside the exact bounded fixture set"
    );
    ensure!(
        view.source_rows.iter().copied().eq(0..rows),
        "Q8 projection reference requires identity physical rows"
    );
    let code_bytes = rows
        .checked_mul(k)
        .context("Q8 projection code extent overflow")?;
    let scale_count = rows
        .checked_mul(k / GROUP)
        .context("Q8 projection scale count overflow")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("Q8 projection scale extent overflow")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("Q8 projection scale offset overflow")?;
    ensure!(
        view.codes.offset == 0 && view.codes.bytes == u64::try_from(code_bytes)?,
        "Q8 projection code plane extent mismatch"
    );
    ensure!(
        view.scale_count == scale_count
            && view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?,
        "Q8 projection scale plane extent mismatch"
    );
    ensure!(
        parent.len()
            == scale_offset
                .checked_add(scale_bytes)
                .context("Q8 parent extent overflow")?,
        "Q8 projection parent byte extent mismatch"
    );
    ensure!(
        parent[code_bytes..scale_offset]
            .iter()
            .all(|&byte| byte == 0),
        "Q8 projection code alignment padding is nonzero"
    );
    for row in 0..rows {
        for group in 0..k / GROUP {
            let group_offset = row * k + group * GROUP;
            let scale = half(parent, scale_offset + (row * (k / GROUP) + group) * 2)?;
            ensure!(
                scale.is_finite() && scale >= 0.0,
                "Q8 projection scale is negative or nonfinite"
            );
            if scale == 0.0 {
                ensure!(
                    parent[group_offset..group_offset + GROUP]
                        .iter()
                        .all(|&code| code == 0),
                    "zero Q8 projection scale requires zero codes"
                );
            }
        }
    }
    Ok((rows, k, scale_offset, input.len() / k))
}

fn half(parent: &[u8], offset: usize) -> anyhow::Result<f32> {
    let bytes = parent
        .get(offset..offset + 2)
        .context("Q8 FP16 scale is out of bounds")?;
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
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
    let bits = value
        .to_bits()
        .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1));
    let bytes = bits.to_le_bytes();
    u16::from_le_bytes([bytes[2], bytes[3]])
}

fn error_bound(operations: f64, absolute_sum: f64) -> f64 {
    let unit_roundoff = f64::EPSILON / 2.0;
    let gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff);
    2.0 * gamma * absolute_sum / (1.0 - gamma)
}

#[cfg(test)]
mod tests {
    use super::{ProjectionReferenceRequest, round_bf16, run};
    use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView};

    #[test]
    fn four_split_schedule_when_signed_contributions_cover_each_pair_matches_hand_sum() {
        let (parent, view, input) = fixture(1, 256, 1);
        let mut parent = parent;
        let mut input = input;
        for (split, activation) in [0x3f80, 0x4000, 0xbf80, 0x3f00].into_iter().enumerate() {
            parent[split * 64] = 1;
            input[split * 64] = activation;
        }

        let result = run(ProjectionReferenceRequest {
            parent: &parent,
            view: &view,
            input_bf16: &input,
            split_warps: 4,
        })
        .expect("bounded projection");

        assert_eq!(result.scheduled_f32, [2.5]);
        assert_eq!(result.output_bf16, [round_bf16(2.5)]);
        assert_eq!(result.mathematical_f64, [2.5]);
    }

    #[test]
    fn token_four_last_row_when_only_final_k_is_live_matches_bf16_factor() {
        let (mut parent, view, mut input) = fixture(2, 256, 5);
        parent[256 + 255] = 1;
        input[4 * 256 + 255] = 0x4000;

        let result = run(ProjectionReferenceRequest {
            parent: &parent,
            view: &view,
            input_bf16: &input,
            split_warps: 4,
        })
        .expect("bounded T5 projection");

        assert_eq!(result.output_bf16, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0x4000]);
    }

    fn fixture(rows: usize, k: usize, tokens: usize) -> (Vec<u8>, Q8MatrixView, Vec<u16>) {
        let code_bytes = rows * k;
        let scale_offset = code_bytes + (256 - code_bytes % 256) % 256;
        let scale_count = rows * (k / 32);
        let mut parent = vec![0_u8; scale_offset + scale_count * 2];
        for scale in parent[scale_offset..].as_chunks_mut::<2>().0 {
            scale.copy_from_slice(&0x3c00_u16.to_le_bytes());
        }
        let view = Q8MatrixView {
            object_id: "multi-split-reference".into(),
            shape: [rows, k],
            padded_k: k,
            group_size: 32,
            codes: BytePlane {
                offset: 0,
                bytes: u64::try_from(code_bytes).expect("code extent"),
            },
            scale_bits: BytePlane {
                offset: u64::try_from(scale_offset).expect("scale offset"),
                bytes: u64::try_from(scale_count * 2).expect("scale extent"),
            },
            scale_count,
            source_rows: (0..rows).collect(),
        };
        (parent, view, vec![0; k * tokens])
    }
}
