//! Resident attention output/MLP chain and independent complete-layer comparison.
use super::{
    attention_gate,
    driver::{Buffer, Context, Module},
    mlp, projections, residual_norm,
};
use crate::{
    entry_reference::bf16_to_f32,
    kernels::AttentionInput,
    layer_comparison_reference, projection_reference,
    qwen_attention_layer_reference::{self, Stage},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) struct Input<'a, 'ctx> {
    pub(super) attention: &'a Buffer<'ctx>,
    pub(super) attention_words: &'a [u16],
    pub(super) gate: &'a Buffer<'ctx>,
    pub(super) gate_words: &'a [u16],
    pub(super) residual: &'a Buffer<'ctx>,
    pub(super) residual_words: &'a [u16],
    pub(super) k_cache: &'a [u16],
    pub(super) v_cache: &'a [u16],
}

pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: Input<'_, '_>,
    weights: &AttentionInput,
    tokens: &[u32],
    positions: &[u32],
    stages: &mut Vec<Stage>,
) -> Result<Value> {
    let gated = attention_gate::check(
        context,
        module,
        attention_gate::Input {
            attention: input.attention,
            attention_words: input.attention_words,
            gate: input.gate,
            gate_words: input.gate_words,
        },
    )?;
    let projected = output_projection(context, module, &gated, weights, tokens.len())?;
    let post_norm = residual_norm::check(
        context,
        module,
        residual_norm::Input {
            residual: input.residual,
            residual_words: input.residual_words,
            branch: &projected.device,
            branch_words: &projected.words,
        },
        &weights.post_attention_norm,
        [tokens.len(), weights.entry.width],
    )?;
    let mlp = mlp::check(
        context,
        module,
        &post_norm,
        &weights.mlp,
        [tokens.len(), weights.entry.width],
    )?;
    stages.push(Stage {
        name: "attention_gate",
        words: gated.words,
        width: weights.query_heads * weights.head_width,
    });
    stages.push(Stage {
        name: "output_projection",
        words: projected.words,
        width: weights.entry.width,
    });
    stages.push(Stage {
        name: "post_residual",
        words: post_norm.residual_words,
        width: weights.entry.width,
    });
    stages.push(Stage {
        name: "post_norm",
        words: post_norm.words,
        width: weights.entry.width,
    });
    stages.extend(mlp.stages);
    let whole = whole_layer(
        weights,
        tokens,
        positions,
        &mlp.words,
        input.k_cache,
        input.v_cache,
        stages,
    )?;
    Ok(
        json!({"all_passed":gated.report["all_passed"]==true && projected.report["passed"]==true && post_norm.report["all_passed"]==true && mlp.report["all_passed"]==true && whole["all_passed"]==true,
        "gate":gated.report,"output_projection":projected.report,"post_attention":post_norm.report,"mlp":mlp.report,"whole_layer_reference":whole,
        "device_intermediates_resident":true,"full_model_executed":false,"logits_compared":false}),
    )
}

struct Projected<'a> {
    device: Buffer<'a>,
    words: Vec<u16>,
    report: Value,
}

fn output_projection<'a>(
    context: &'a Context,
    module: &Module<'_>,
    gated: &attention_gate::Checked<'_>,
    input: &AttentionInput,
    rows: usize,
) -> Result<Projected<'a>> {
    let width = input.query_heads * input.head_width;
    let quantized = projection_reference::quantize(&gated.words, rows, width)?;
    let codes = upload(context, &vec![127; rows * width])?;
    let scales = upload(context, &vec![0xff; rows * 4])?;
    projections::quantize(
        &module.function("fp8_quantize_bf16")?,
        &gated.output,
        &codes,
        &scales,
        rows,
        width,
    )?;
    context.synchronize()?;
    let mut actual_codes = vec![0; rows * width];
    codes.download(&mut actual_codes)?;
    let mut actual_scales = vec![0; rows * 4];
    scales.download(&mut actual_scales)?;
    let actual_scales: Vec<_> = actual_scales
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(
        actual_codes == quantized.codes && actual_scales == quantized.scales,
        "attention output FP8 quantization mismatch"
    );
    let p = &input.output_projection;
    let weights = upload(context, &p.weights)?;
    let weight_scales = upload(context, &p.scales)?;
    let output = projections::run_linear(
        context,
        &module.function("fp8_linear_wide")?,
        &[&codes, &weights, &scales, &weight_scales],
        [rows, p.channels, width],
    )?;
    let (words, unrounded) = output.read(rows * p.channels)?;
    let scales: Vec<_> = p
        .scales
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let expected = projection_reference::linear(&quantized, &p.weights, &scales, width)?;
    let mut report = projections::compare(&words, &unrounded, &expected)?;
    report["input_quantization_exact"] = json!(true);
    report["projection_name"] = json!(p.name);
    report["shape_mnk"] = json!([rows, p.channels, width]);
    Ok(Projected {
        device: output.bf16,
        words,
        report,
    })
}

fn whole_layer(
    input: &AttentionInput,
    tokens: &[u32],
    positions: &[u32],
    output: &[u16],
    k_cache: &[u16],
    v_cache: &[u16],
    stages: &[Stage],
) -> Result<Value> {
    let expected = qwen_attention_layer_reference::run(input, tokens, positions)?;
    let mut stage_reports = Vec::new();
    ensure!(
        stages.len() == expected.stages.len(),
        "attention diagnostic stage count mismatch"
    );
    for expected_stage in &expected.stages {
        let stage = stages
            .iter()
            .find(|s| s.name == expected_stage.name)
            .ok_or_else(|| anyhow::anyhow!("missing attention stage {}", expected_stage.name))?;
        ensure!(
            stage.width == expected_stage.width,
            "attention diagnostic width mismatch"
        );
        let mut report = compare(&stage.words, &expected_stage.words, stage.width)?;
        report["name"] = json!(stage.name);
        report["diagnostics_only"] = json!(true);
        stage_reports.push(report);
    }
    let count = tokens.len() * input.kv_heads * input.head_width;

    ensure!(
        k_cache.len() >= count && v_cache.len() >= count,
        "attention cache prefix missing"
    );
    let hidden = compare(output, &expected.output, input.entry.width)?;
    let k = compare(&k_cache[..count], &expected.k_cache, input.head_width)?;
    let v = compare(&v_cache[..count], &expected.v_cache, input.head_width)?;
    Ok(
        json!({"all_passed":hidden["all_passed"]==true && k["all_passed"]==true && v["all_passed"]==true,
        "hidden":hidden,"k_cache":k,"v_cache":v,"stage_diagnostics":stage_reports,
        "reference_inputs":"original checkpoint weights, token IDs and text positions only; no GPU intermediate substitution",
        "budget":"each token hidden vector and cached token/head vector: normalized L2 <=0.01 and cosine >=0.9999; aggregate also required",
        "scope":"synthetic embedding input through layer 3 only; selected BF16/FP32 arithmetic profile; no preceding layers or logits"}),
    )
}

fn compare(actual: &[u16], expected: &[u16], width: usize) -> Result<Value> {
    let actual_f32: Vec<_> = actual.iter().map(|&v| bf16_to_f32(v)).collect();
    let expected_f32: Vec<_> = expected.iter().map(|&v| bf16_to_f32(v)).collect();
    let mut report =
        layer_comparison_reference::compare_partitioned(&actual_f32, &expected_f32, width)?;
    report["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    Ok(report)
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

pub(super) fn projection_fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for [rows, channels, width] in [[1, 1, 1], [3, 9, 35], [1, 1, 35]] {
        let mut input: Vec<_> = (0..rows * width)
            .map(|i| crate::entry_reference::round_bf16(((i * 11 % 29) as f32 - 14.0) / 8.0))
            .collect();
        let cancellation = rows == 1 && width == 35;
        if cancellation {
            input.fill(0);
            for (index, value) in [
                (0, 448.0),
                (1, 1.0),
                (2, 0.0625),
                (3, 1.0 / 512.0),
                (31, 448.0),
            ] {
                input[index] = crate::entry_reference::round_bf16(value);
            }
        }
        let quantized = projection_reference::quantize(&input, rows, width)?;

        let mut weights: Vec<_> = (0..channels * width)
            .map(|i| ((i * 17 % 127) as u8) | if i % 3 == 0 { 128 } else { 0 })
            .collect();
        if cancellation {
            weights.fill(0);
            for (index, code) in [(0, 0x7e), (1, 0x38), (2, 0x18), (3, 0x01), (31, 0xfe)] {
                weights[index] = code;
            }
        }
        let scales =
            vec![
                crate::entry_reference::round_bf16(if cancellation { 1.0 } else { 1.0 / 256.0 });
                channels
            ];
        let expected = projection_reference::linear(&quantized, &weights, &scales, width)?;
        let input_device = upload(context, &quantized.codes)?;
        let weight_device = upload(context, &weights)?;
        let input_scales = upload(
            context,
            &quantized
                .scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let weight_scales = upload(
            context,
            &scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        for kernel in ["fp8_linear", "fp8_linear_wide"] {
            let output = projections::run_linear(
                context,
                &module.function(kernel)?,
                &[&input_device, &weight_device, &input_scales, &weight_scales],
                [rows, channels, width],
            )?;
            let (words, unrounded) = output.read(rows * channels)?;
            let comparison = projections::compare(&words, &unrounded, &expected)?;
            let exact_bf16 = words == expected.normalized;
            ensure!(
                kernel != "fp8_linear_wide" || exact_bf16,
                "refined FP8 fixture BF16 mismatch"
            );
            reports.push(json!({"all_passed":comparison["passed"]==true,"kernel":kernel,"cancellation_fixture":cancellation,"bf16_reference_exact":exact_bf16,"shape_mnk":[rows,channels,width],"comparison":comparison}));
        }
    }
    Ok(reports)
}
