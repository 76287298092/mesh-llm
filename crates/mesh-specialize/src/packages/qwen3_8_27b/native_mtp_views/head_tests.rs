use super::cpu_reference::{MtpHeadStep, q4_g64_fp16_mtp_head_step};
use super::{BytePlane, NativeMtpViews, ProposalTokenMap, Q4MatrixView, TargetTokenId};

fn scale_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn q4_code_row(code: u8) -> Vec<u8> {
    vec![(code & 0x0f) | ((code & 0x0f) << 4); 64]
}

#[test]
fn q4_proposal_head_step_returns_bf16_logits_and_target_id() {
    let codes = [q4_code_row(1), q4_code_row(2), q4_code_row(0)].concat();
    let scales = scale_bytes(&[0x3c00; 6]);
    let mut object = vec![0; 256 + scales.len()];
    object[..codes.len()].copy_from_slice(&codes);
    object[256..].copy_from_slice(&scales);
    let mut views = empty_views();
    views.proposal_head = Q4MatrixView {
        object_id: "proposal".into(),
        shape: [3, 128],
        padded_k: 128,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(codes.len()).expect("code length fits u64"),
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: u64::try_from(scales.len()).expect("scale length fits u64"),
        },
        scale_count: 6,
        source_rows: vec![0, 1, 2],
    };
    views.proposal_tokens = ProposalTokenMap {
        target_ids: vec![TargetTokenId(9), TargetTokenId(7), TargetTokenId(5)],
    };
    let hidden = vec![0x3f80; 128];
    let result: MtpHeadStep = q4_g64_fp16_mtp_head_step(
        &views.proposal_head,
        &object,
        &hidden,
        &views.proposal_tokens,
    )
    .expect("Q4 head projects, rounds, and maps its winning row");
    assert_eq!(result.logits_bf16, [0x4300, 0x4380, 0x0000]);
    assert_eq!(result.proposal_row, 1);
    assert_eq!(result.target_token, 7);
}

fn empty_views() -> NativeMtpViews {
    use super::{Bf16NormView, NativeMtpNormViews, Q8MatrixView};

    let q8 = Q8MatrixView {
        object_id: String::new(),
        shape: [1, 128],
        padded_k: 128,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: 128,
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: 8,
        },
        scale_count: 4,
        source_rows: vec![0],
    };
    let norm = Bf16NormView {
        object_id: String::new(),
        elements: 1,
        bytes: 2,
    };
    NativeMtpViews {
        fc: q8.clone(),
        query_gate: q8.clone(),
        key: q8.clone(),
        value: q8.clone(),
        attention_output: q8.clone(),
        mlp_gate: q8.clone(),
        mlp_up: q8.clone(),
        mlp_down: q8,
        norms: NativeMtpNormViews {
            embedding: norm.clone(),
            hidden: norm.clone(),
            final_norm: norm.clone(),
            input: norm.clone(),
            post_attention: norm.clone(),
            query: norm.clone(),
            key: norm,
        },
        proposal_head: Q4MatrixView {
            object_id: String::new(),
            shape: [1, 128],
            padded_k: 128,
            group_size: 64,
            codes: BytePlane {
                offset: 0,
                bytes: 64,
            },
            scale_bits: BytePlane {
                offset: 256,
                bytes: 4,
            },
            scale_count: 2,
            source_rows: vec![0],
        },
        proposal_tokens: ProposalTokenMap {
            target_ids: vec![TargetTokenId(0)],
        },
    }
}
