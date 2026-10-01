use super::super::driver::{Buffer, Context, Module};
use super::launch::{self, Outputs, Request};
use super::metrics::{self, compare_values};
use super::types::{Case, Weight};
use crate::nvfp4_swiglu_a16_reference as reference;
use crate::{entry_reference::bf16_to_f32, nvfp4_linear_reference::Matrix};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

const RAW_RELATIVE_L2_LIMIT: f64 = 0.01;
const RAW_MAX_SCALED_ERROR_LIMIT: f64 = 1.0e-4;
const ACTIVATION_RELATIVE_L2_LIMIT: f64 = 0.01;

pub(super) fn run(context: &Context, module: &Module<'_>, case: Case<'_>) -> Result<Value> {
    let Case {
        source,
        input,
        width,
        gate,
        up,
    } = case;
    ensure!(
        (16..=32768).contains(&width) && width.is_multiple_of(16),
        "invalid fused NVFP4 A16 K geometry"
    );
    ensure!(
        input.len() == width && input.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "invalid fused NVFP4 A16 BF16 input"
    );
    ensure!(
        gate.divisor.is_finite()
            && gate.divisor > 0.0
            && up.divisor.is_finite()
            && up.divisor > 0.0,
        "invalid fused NVFP4 A16 weight divisor"
    );
    let channels = gate.scales.len() / (width / 16);
    ensure!(
        (1..=32768).contains(&channels)
            && gate.packed.len() == channels * width / 2
            && gate.scales.len() == channels * width / 16
            && up.packed.len() == channels * width / 2
            && up.scales.len() == channels * width / 16,
        "fused NVFP4 A16 gate/up extents differ"
    );
    let expected = reference::run(reference::SwiGluInput {
        activation_bf16: input,
        gate: matrix(&gate, channels),
        up: matrix(&up, channels),
        width,
    })?;
    let input_bytes: Vec<_> = input.iter().flat_map(|word| word.to_le_bytes()).collect();
    let input_device = upload(context, &input_bytes)?;
    let outputs = Outputs::new(context, channels)?;
    launch::run(Request {
        context,
        module,
        input: &input_device,
        weights: [&gate, &up],
        outputs: &outputs,
        dimensions: [channels, width],
    })?;
    compare_outputs(source, &expected, &outputs)
}

fn matrix<'a>(weight: &Weight<'a>, channels: usize) -> Matrix<'a> {
    Matrix {
        packed: weight.packed,
        scales: weight.scales,
        rows: channels,
        global: weight.divisor,
    }
}

fn compare_outputs(
    source: &str,
    expected: &reference::SwiGluResult,
    outputs: &Outputs<'_>,
) -> Result<Value> {
    let gate_raw = download_f32(&outputs.gate_raw)?;
    let up_raw = download_f32(&outputs.up_raw)?;
    let gate_bf16 = download_bf16(&outputs.gate)?;
    let up_bf16 = download_bf16(&outputs.up)?;
    let activation_raw = download_f32(&outputs.activation_raw)?;
    let activation = download_bf16(&outputs.activation)?;
    ensure!(
        gate_raw
            .iter()
            .chain(&up_raw)
            .chain(&activation_raw)
            .all(|value| value.is_finite())
            && gate_bf16
                .iter()
                .chain(&up_bf16)
                .chain(&activation)
                .all(|&bits| bf16_to_f32(bits).is_finite()),
        "fused NVFP4 A16 produced nonfinite output"
    );
    let gate = compare_values(&gate_raw, &expected.gate_dot, &expected.gate_absolute_sum);
    let up = compare_values(&up_raw, &expected.up_dot, &expected.up_absolute_sum);
    let activation_expected = expected
        .activation_raw_fp32
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let activation_error = compare_values(
        &activation_raw,
        &activation_expected,
        &expected.activation_absolute_sum,
    );
    let gate_rounding = metrics::exact_rounding(&gate_raw, &gate_bf16);
    let up_rounding = metrics::exact_rounding(&up_raw, &up_bf16);
    let activation_rounding = metrics::exact_rounding(&activation_raw, &activation);
    let activation_l2 = metrics::fp32_l2(&activation_raw, &expected.activation_raw_fp32);
    let activation_bf16_matches_reference = activation == expected.activation_bf16;
    let activation_differences = activation
        .iter()
        .zip(&expected.activation_bf16)
        .filter(|(left, right)| left != right)
        .count();
    let all_passed = gate.relative_l2 <= RAW_RELATIVE_L2_LIMIT
        && gate.max_scaled_error <= RAW_MAX_SCALED_ERROR_LIMIT
        && up.relative_l2 <= RAW_RELATIVE_L2_LIMIT
        && up.max_scaled_error <= RAW_MAX_SCALED_ERROR_LIMIT
        && activation_error.relative_l2 <= RAW_RELATIVE_L2_LIMIT
        && activation_error.max_scaled_error <= RAW_MAX_SCALED_ERROR_LIMIT
        && gate_rounding
        && up_rounding
        && activation_rounding
        && activation_l2 <= ACTIVATION_RELATIVE_L2_LIMIT;
    Ok(json!({
        "source": source,
        "gate": gate,
        "up": up,
        "activation_raw": activation_error,
        "gate_bf16_rounding_exact": gate_rounding,
        "up_bf16_rounding_exact": up_rounding,
        "activation_bf16_rounding_exact": activation_rounding,
        "activation_bf16_matches_fp64_reference": activation_bf16_matches_reference,
        "activation_bf16_differences_vs_fp64": activation_differences,
        "activation_raw_relative_l2": activation_l2,
        "raw_relative_l2_limit": RAW_RELATIVE_L2_LIMIT,
        "raw_max_scaled_error_limit": RAW_MAX_SCALED_ERROR_LIMIT,
        "activation_relative_l2_limit": ACTIVATION_RELATIVE_L2_LIMIT,
        "all_passed": all_passed,
    }))
}

fn download_bf16(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

fn download_f32(buffer: &Buffer<'_>) -> Result<Vec<f32>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
