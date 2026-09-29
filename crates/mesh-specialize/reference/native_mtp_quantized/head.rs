use super::error::NativeMtpDecodeError;
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    packages::qwen3_8_27b::native_mtp_views::{ProposalTokenMap, Q4MatrixView},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpHeadStep {
    /// One BF16 proposal logit for each indexed shortlist row.
    pub logits_bf16: Vec<u16>,
    /// Winning row in the local proposal-head domain.
    pub proposal_row: usize,
    /// Target-vocabulary ID selected through the signed proposal token map.
    pub target_token: u32,
}

pub fn q4_g64_fp16_mtp_head_step(
    head: &Q4MatrixView,
    object_bytes: &[u8],
    hidden_bf16: &[u16],
    proposal_tokens: &ProposalTokenMap,
) -> Result<MtpHeadStep, NativeMtpDecodeError> {
    let [rows, logical_k] = head.shape;
    if rows != head.source_rows.len() || rows > proposal_tokens.len() {
        return Err(NativeMtpDecodeError::ProposalMapExtent {
            expected: rows,
            actual: proposal_tokens.len(),
        });
    }
    if hidden_bf16.len() != logical_k {
        return Err(NativeMtpDecodeError::ActivationExtent {
            expected: logical_k,
            actual: hidden_bf16.len(),
        });
    }
    let mut activations = Vec::with_capacity(logical_k);
    for (index, &bits) in hidden_bf16.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(NativeMtpDecodeError::NonFiniteActivation { index });
        }
        activations.push(value);
    }
    let mut logits_bf16 = Vec::with_capacity(rows);
    let mut proposal_row = 0;
    let mut best_logit = f32::NEG_INFINITY;
    for row in 0..rows {
        let weights = super::views::decode_q4_view_row(object_bytes, head, row)?;
        let logit = round_bf16(dot(&weights, &activations)?);
        let value = bf16_to_f32(logit);
        logits_bf16.push(logit);
        if value > best_logit {
            best_logit = value;
            proposal_row = row;
        }
    }
    let target_token = proposal_tokens
        .target_id(proposal_row)
        .ok_or(NativeMtpDecodeError::ProposalMapExtent {
            expected: proposal_row + 1,
            actual: proposal_tokens.len(),
        })?
        .value();
    Ok(MtpHeadStep {
        logits_bf16,
        proposal_row,
        target_token,
    })
}

fn dot(weights: &[f32], activations: &[f32]) -> Result<f32, NativeMtpDecodeError> {
    if activations.len() > weights.len() {
        return Err(NativeMtpDecodeError::ActivationExtent {
            expected: weights.len(),
            actual: activations.len(),
        });
    }
    let mut sum = 0.0_f32;
    for (&weight, &activation) in weights.iter().zip(activations) {
        sum = weight.mul_add(activation, sum);
    }
    Ok(sum)
}
