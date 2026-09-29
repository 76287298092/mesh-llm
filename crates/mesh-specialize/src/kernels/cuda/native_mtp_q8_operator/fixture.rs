use super::{compare, launch, validate};
use crate::{native_mtp_q8_gemv_reference, packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView}};
use super::super::driver::{Context, Module};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

struct Fixture {
    name: &'static str,
    view: Q8MatrixView,
    object_bytes: Vec<u8>,
    input_bf16: Vec<u16>,
}

pub(super) fn run_synthetic(context: &Context, module: &Module<'_>) -> Result<Value> {
    let mut cases = Vec::new();
    for width in [128, 160, 256, 5120, 10240, 17408] {
        let fixture = fixture(width)?;
        let validated = validate::validate(&fixture.object_bytes, &fixture.view, &fixture.input_bf16)?;
        let expected = native_mtp_q8_gemv_reference::run(
            &fixture.object_bytes,
            &fixture.view,
            &fixture.input_bf16,
        )?;
        let output = launch::run(
            context,
            module,
            &fixture.object_bytes,
            &fixture.input_bf16,
            &validated,
        )?;
        let repeated_output = launch::run(
            context,
            module,
            &fixture.object_bytes,
            &fixture.input_bf16,
            &validated,
        )?;
        ensure!(
            output == repeated_output,
            "native Q8 GEMV output changed between repeats for {}",
            fixture.name
        );
        cases.push(compare::run(fixture.name, &output, &expected)?);
    }
    Ok(json!({
        "kind": "native-mtp-row-split-k128-q8-gemv-operator-check-v1",
        "kernel_resources": module.function("native_mtp_q8_gemv")?.resources()?,
        "arithmetic": "signed Q8 codes, little-endian FP16 group-32 scales, BF16 activations, FP32 warp reduction",
        "all_passed": cases.iter().all(|case| case["all_passed"] == true),
        "cases": cases,
        "scope": "synthetic single-vector matrix-row operator only; native MTP integration and GPU qualification are not established",
    }))
}

#[cfg(test)]
pub(super) fn fixture_for_test(width: usize) -> Result<(Vec<u8>, Q8MatrixView, Vec<u16>)> {
    let fixture = fixture(width)?;
    Ok((fixture.object_bytes, fixture.view, fixture.input_bf16))
}

fn fixture(width: usize) -> Result<Fixture> {
    let padded_k = width.checked_add(127).context("fixture padded K overflows")? / 128 * 128;
    let parent_rows = 3_usize;
    let code_bytes = parent_rows.checked_mul(padded_k).context("fixture code extent overflows")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("fixture scale offset overflows")?;
    let scale_count = parent_rows
        .checked_mul(padded_k / 32)
        .context("fixture scale count overflows")?;
    let scale_bytes = scale_count.checked_mul(2).context("fixture scale extent overflows")?;
    let mut object_bytes = vec![0; scale_offset + scale_bytes];
    let scale_words = [0x3800_u16, 0x3c00, 0x4000, 0x3400];
    for parent_row in 0..parent_rows {
        for k in 0..width {
            let code = if k == 0 {
                127_i8
            } else if k == 1 {
                -127_i8
            } else {
                i8::try_from((k * 13 + parent_row * 7 + k / 32) % 31)? - 15
            };
            object_bytes[parent_row * padded_k + k] = code.to_ne_bytes()[0];
        }
        for group in 0..(padded_k / 32) {
            let scale = if group * 32 < width {
                scale_words[(parent_row + group) % scale_words.len()]
            } else {
                0
            };
            let offset = scale_offset + (parent_row * (padded_k / 32) + group) * 2;
            object_bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    let activation_pattern = [0xbf80_u16, 0xbf00, 0x0000, 0x3f00, 0x3f80, 0x3fc0];
    let input_bf16: Vec<u16> = (0..width)
        .map(|index| activation_pattern[index % activation_pattern.len()])
        .collect();
    let view = Q8MatrixView {
        object_id: "synthetic-native-mtp-q8".into(),
        shape: [2, width],
        padded_k,
        group_size: 32,
        codes: BytePlane { offset: 0, bytes: u64::try_from(code_bytes)? },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset)?,
            bytes: u64::try_from(scale_bytes)?,
        },
        scale_count,
        source_rows: vec![2, 0],
    };
    let name = match width {
        128 => "k128",
        160 => "k160-tail",
        256 => "k256-two-splits",
        5120 => "k5120",
        10240 => "k10240",
        17408 => "k17408",
        _ => anyhow::bail!("unsupported synthetic fixture width"),
    };
    ensure!(view.shape[1] == input_bf16.len(), "fixture activation width mismatch");
    Ok(Fixture { name, view, object_bytes, input_bf16 })
}
