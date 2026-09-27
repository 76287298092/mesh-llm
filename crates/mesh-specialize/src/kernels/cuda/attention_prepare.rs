//! Per-head Q/K normalization and partial RoPE with resident projection inputs.
use super::driver::{Buffer, Context, Module};
use crate::{
    attention_prepare_reference as reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) device: &'a Buffer<'ctx>,
    pub(super) words: &'a [u16],
    pub(super) weight: &'a [u16],
    pub(super) cos: &'a [u16],
    pub(super) sin: &'a [u16],
    pub(super) shape: &'a reference::Shape,
}

pub(super) fn check(context: &Context, module: &Module<'_>, input: Input<'_, '_>) -> Result<Value> {
    let shape = input.shape;
    let epsilon = 1e-6_f32;
    let expected = reference::run(
        input.words,
        input.weight,
        input.cos,
        input.sin,
        shape,
        epsilon,
    )?;
    let count = expected.output.len();
    let weight = upload_words(context, input.weight)?;
    let cos = upload_words(context, input.cos)?;
    let sin = upload_words(context, input.sin)?;
    let output = upload(context, &vec![0xa5; count * 2])?;
    let normalized = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let gate = upload(context, &vec![0xa5; count * 2])?;
    let mut pointers = [
        input.device.pointer(),
        weight.pointer(),
        cos.pointer(),
        sin.pointer(),
        output.pointer(),
        normalized.pointer(),
        unrounded.pointer(),
        gate.pointer(),
    ];
    let mut dims = [
        u32::try_from(shape.rows)?,
        u32::try_from(shape.heads)?,
        u32::try_from(shape.width)?,
        u32::try_from(shape.rotary_dim)?,
        u32::from(shape.with_gate),
    ];
    let mut epsilon_arg = epsilon;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dims.iter_mut().map(|p| (p as *mut u32).cast()));
    args.push((&mut epsilon_arg as *mut f32).cast());
    // SAFETY: The independent reference validates all shapes and host extents.
    // Input is the corresponding resident device projection. Output allocations
    // are disjoint, and every head uses 256 threads through every barrier.
    unsafe {
        module.function("attention_qk_prepare")?.launch(
            [u32::try_from(shape.rows * shape.heads)?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let actual = words(&output, count)?;
    let actual_norm = words(&normalized, count)?;
    let actual_fp32 = floats(&unrounded, count)?;
    let actual_gate = words(&gate, count)?;
    let expected_rotation = reference::rotate(&actual_norm, input.cos, input.sin, shape)?;
    let rotation_exact = actual == expected_rotation;
    let gate_exact = if shape.with_gate {
        actual_gate == expected.gate
    } else {
        actual_gate.iter().all(|&v| v == 0xa5a5)
    };
    let mut max_norm_error = 0.0_f32;
    let mut norm_mismatches = 0;
    let mut rounding_mismatches = 0;
    for (index, (&value, &reference)) in actual_fp32.iter().zip(&expected.unrounded).enumerate() {
        ensure!(value.is_finite(), "nonfinite attention norm result");
        let error = (value - reference).abs();
        max_norm_error = max_norm_error.max(error);
        norm_mismatches += usize::from(error > 1e-6 + 3e-6 * reference.abs());
        rounding_mismatches += usize::from(actual_norm[index] != round_bf16(value));
    }
    let decode = |words: &[u16]| words.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
    let full_reference = crate::layer_comparison_reference::compare_partitioned(
        &decode(&actual),
        &decode(&expected.output),
        shape.width,
    )?;
    let scalar_norm_differences = actual_norm
        .iter()
        .zip(&expected.normalized)
        .filter(|(a, b)| a != b)
        .count();
    let scalar_output_differences = actual
        .iter()
        .zip(&expected.output)
        .filter(|(a, b)| a != b)
        .count();
    let passed = norm_mismatches == 0
        && rounding_mismatches == 0
        && rotation_exact
        && gate_exact
        && full_reference["all_passed"] == true;
    Ok(
        json!({"all_passed":passed,"shape":{"rows":shape.rows,"heads":shape.heads,"width":shape.width,"rotary_dim":shape.rotary_dim,"with_gate":shape.with_gate},
        "elements":count,"norm_max_abs_error":max_norm_error,"norm_mismatches":norm_mismatches,"rounding_mismatches":rounding_mismatches,
        "norm_tolerance":"1e-6 + 3e-6*abs(reference)","rotation_from_actual_norm_exact":rotation_exact,"gate_copy_or_untouched_exact":gate_exact,
        "norm_bf16_scalar_differences":scalar_norm_differences,"output_bf16_scalar_differences":scalar_output_differences,"independent_operation_reference":full_reference,
        "device_projection_input_resident":true,"scope":"Q/K normalization, partial RoPE and Q gate split only; no attention scores or KV cache"}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for (rows, heads, width, rotary_dim, with_gate, zero) in [
        (2, 3, 8, 4, true, false),
        (17, 2, 256, 64, false, false),
        (3, 1, 258, 64, true, false),
        (1, 1, 1024, 64, false, false),
        (1, 2, 8, 4, true, true),
    ] {
        let shape = reference::Shape {
            rows,
            heads,
            width,
            rotary_dim,
            with_gate,
        };
        let count = rows * heads * width * if with_gate { 2 } else { 1 };
        let input: Vec<_> = (0..count)
            .map(|i| {
                round_bf16(if zero {
                    0.0
                } else {
                    ((i * 37 % 257) as i32 - 128) as f32 / 64.0
                })
            })
            .collect();
        let weight: Vec<_> = (0..width)
            .map(|i| {
                round_bf16(if i % 11 == 0 {
                    -1.0
                } else {
                    ((i * 7 % 31) as i32 - 15) as f32 / 32.0
                })
            })
            .collect();
        let positions: Vec<_> = (0..rows)
            .map(|i| [0, 1, 262143, 131071, 131072][i % 5])
            .collect();
        let (cos, sin) = reference::text_rope_tables(&positions, rotary_dim, 1e7)?;
        let device = upload_words(context, &input)?;
        let mut report = check(
            context,
            module,
            Input {
                device: &device,
                words: &input,
                weight: &weight,
                cos: &cos,
                sin: &sin,
                shape: &shape,
            },
        )?;
        report["positions"] = json!(positions);
        report["zero_input"] = json!(zero);
        reports.push(report);
    }
    Ok(reports)
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = context.allocate(bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn upload_words<'a>(context: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    upload(
        context,
        &words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect())
}
fn floats(buffer: &Buffer<'_>, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 4];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
