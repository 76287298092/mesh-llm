use super::super::driver::{Context, Module};
use super::{compare, launch, schedule_reference, validate};
use crate::{
    native_mtp_q4_gemv_reference,
    packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView},
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

#[path = "fixture_dense.rs"]
mod dense;

struct Fixture {
    name: &'static str,
    view: Q4MatrixView,
    object_bytes: Vec<u8>,
    input_bf16: Vec<u16>,
    proposal_tokens: Vec<u32>,
}

pub(super) fn run_synthetic(context: &Context, module: &Module<'_>) -> Result<Value> {
    let mut cases = Vec::new();
    for fixture in [
        fixture(128, false)?,
        fixture(160, false)?,
        fixture(163, false)?,
        fixture(256, false)?,
        fixture(5120, false)?,
        fixture(128, true)?,
        dense::build()?,
    ] {
        let validated =
            validate::validate(&fixture.object_bytes, &fixture.view, &fixture.input_bf16)?;
        let expected = native_mtp_q4_gemv_reference::run(
            &fixture.object_bytes,
            &fixture.view,
            &fixture.input_bf16,
        )?;
        let schedule_expected =
            schedule_reference::run(&fixture.object_bytes, &fixture.input_bf16, &validated)?;
        let input = launch::Input {
            object_bytes: &fixture.object_bytes,
            input_bf16: &fixture.input_bf16,
            validated: &validated,
        };
        let output = launch::run(context, module, &input)?;
        let repeated_output = launch::run(context, module, &input)?;
        let repeated_raw_bits_match = output
            .raw_f32
            .iter()
            .map(|value| value.to_bits())
            .eq(repeated_output.raw_f32.iter().map(|value| value.to_bits()));
        let repeated_bf16_bits_match = output.logits_bf16 == repeated_output.logits_bf16;
        ensure!(
            repeated_raw_bits_match && repeated_bf16_bits_match,
            "native Q4 outputs changed between repeats for {}",
            fixture.name
        );
        let mut comparison = compare::run(fixture.name, &output, &expected)?;
        let schedule_bits_match = output
            .raw_f32
            .iter()
            .zip(&schedule_expected.raw_f32)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits());
        let schedule_bf16_bits_match = output.logits_bf16 == schedule_expected.logits_bf16;
        comparison["schedule_raw_bits_match"] = json!(schedule_bits_match);
        comparison["schedule_bf16_bits_match"] = json!(schedule_bf16_bits_match);
        comparison["schedule_oracle"] =
            json!("GemvR4W1 lane order and warp-reduction FP32 schedule");
        let winner = comparison["selected_proposal_row"]
            .as_u64()
            .and_then(|row| usize::try_from(row).ok())
            .context("native Q4 selected row is not an index")?;
        let expected_winner = reference_first_argmax(&expected.logits_bf16)?;
        let target_token = *fixture
            .proposal_tokens
            .get(winner)
            .context("native Q4 selected row is absent from proposal-token map")?;
        let expected_target_token = *fixture
            .proposal_tokens
            .get(expected_winner)
            .context("reference proposal row is absent from proposal-token map")?;
        comparison["target_token"] = json!(target_token);
        comparison["reference_proposal_row"] = json!(expected_winner);
        comparison["reference_target_token"] = json!(expected_target_token);
        comparison["winner_matches_reference"] = json!(winner == expected_winner);
        comparison["target_token_matches_reference"] = json!(target_token == expected_target_token);
        comparison["repeated_raw_bits_match"] = json!(repeated_raw_bits_match);
        comparison["repeated_bf16_bits_match"] = json!(repeated_bf16_bits_match);
        let numerical_passed = comparison["all_passed"] == true;
        let all_passed = numerical_passed
            && winner == expected_winner
            && target_token == expected_target_token
            && repeated_raw_bits_match
            && repeated_bf16_bits_match
            && schedule_bits_match
            && schedule_bf16_bits_match;
        comparison["all_passed"] = json!(all_passed);
        cases.push(comparison);
    }
    Ok(json!({
        "kind": "native-mtp-gemv-r4-w1-q4-head-operator-check-v1",
        "kernel_resources": module.function("native_mtp_q4_head_gemv")?.resources()?,
        "arithmetic": "Ninfer GemvR4W1: packed-word eight-code lane, FP16-mantissa decode, ordered FP32 FMA, five-step warp reduction, BF16 RNE",
        "raw_scaled_error_limit": 2.0e-4,
        "fp64_oracle": "independent mathematical projection; scaled-error threshold unchanged",
        "schedule_oracle": "separate exact FP32 lane accumulation/reduction reference; exact bits required",
        "bf16_schedule_logits_require_exact_bits": true,
        "bf16_fp64_rounding_is_diagnostic_only": true,
        "all_passed": cases.iter().all(|case| case["all_passed"] == true),
        "cases": cases,
        "scope": "synthetic indexed proposal-head projection only; MTP integration and GPU qualification are not established",
    }))
}

fn reference_first_argmax(logits: &[u16]) -> Result<usize> {
    ensure!(!logits.is_empty(), "reference proposal logits are empty");
    let mut winner = 0;
    let mut best = f32::from_bits(u32::from(logits[0]) << 16);
    for (row, &bits) in logits.iter().enumerate().skip(1) {
        let value = f32::from_bits(u32::from(bits) << 16);
        if value > best {
            winner = row;
            best = value;
        }
    }
    Ok(winner)
}

#[cfg(test)]
pub(super) fn fixture_for_test(width: usize) -> Result<(Vec<u8>, Q4MatrixView, Vec<u16>)> {
    let fixture = fixture(width, false)?;
    Ok((fixture.object_bytes, fixture.view, fixture.input_bf16))
}

fn fixture(width: usize, tied: bool) -> Result<Fixture> {
    let padded_k = width
        .checked_add(127)
        .context("fixture padded K overflows")?
        / 128
        * 128;
    let parent_rows = 3_usize;
    let code_bytes = parent_rows
        .checked_mul(padded_k / 2)
        .context("fixture code extent overflows")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("fixture scale offset overflows")?;
    let groups_per_row = padded_k / 64;
    let scale_count = parent_rows
        .checked_mul(groups_per_row)
        .context("fixture scale count overflows")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("fixture scale extent overflows")?;
    let mut object_bytes = vec![0; scale_offset + scale_bytes];
    for parent_row in 0..parent_rows {
        for k in 0..width {
            let code = if tied {
                0_i8
            } else {
                match (parent_row, k) {
                    (2, 0) => -8_i8,
                    (2, 1) => -1,
                    (2, 64) => 3,
                    (2, 128) => 2,
                    (0, 0) => 7,
                    (0, 1) => 1,
                    (0, 128) => 1,
                    _ => 0,
                }
            };
            let nibble = u8::from_ne_bytes([code.to_ne_bytes()[0]]) & 0x0f;
            let packed_index = parent_row * (padded_k / 2) + k / 2;
            if k.is_multiple_of(2) {
                object_bytes[packed_index] |= nibble;
            } else {
                object_bytes[packed_index] |= nibble << 4;
            }
        }
        for group in 0..groups_per_row {
            let scale = if tied {
                0
            } else if parent_row == 2 && group == 0 {
                0x0001_u16
            } else if group * 64 < width {
                0x3c00_u16
            } else {
                0
            };
            let offset = scale_offset + (parent_row * groups_per_row + group) * 2;
            object_bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    let mut input_bf16 = vec![0; width];
    if tied {
        input_bf16.fill(0x3f80);
    } else {
        input_bf16[0] = 0x3f80;
        input_bf16[1] = 0xbf80;
        input_bf16[64] = 0x3f80;
        if width > 128 {
            input_bf16[128] = 0xbf80;
        }
        if width > 162 {
            input_bf16[162] = 0x3f80;
        }
    }
    let view = Q4MatrixView {
        object_id: "synthetic-native-mtp-q4".into(),
        shape: [2, width],
        padded_k,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(code_bytes)?,
        },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset)?,
            bytes: u64::try_from(scale_bytes)?,
        },
        scale_count,
        source_rows: vec![2, 0],
    };
    let name = match (width, tied) {
        (128, false) => "k128",
        (160, false) => "k160-tail",
        (163, false) => "k163-vector-tail",
        (256, false) => "k256-two-splits",
        (5120, false) => "k5120",
        (128, true) => "k128-first-row-tie",
        _ => anyhow::bail!("unsupported synthetic fixture width"),
    };
    ensure!(
        view.shape[1] == input_bf16.len(),
        "fixture activation width mismatch"
    );
    Ok(Fixture {
        name,
        view,
        object_bytes,
        input_bf16,
        proposal_tokens: vec![91_337, 17],
    })
}
