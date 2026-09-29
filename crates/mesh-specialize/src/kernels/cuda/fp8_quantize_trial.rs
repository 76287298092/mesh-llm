use super::{
    driver::{Buffer, Context, Module},
    projections,
};
use crate::{entry_reference::round_bf16, projection_reference};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
enum Pattern {
    Zero,
    Equal,
    Denormal,
    PositiveMax,
    NegativeMax,
    Mixed,
    Nan,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Equal => "all-equal",
            Self::Denormal => "denormal",
            Self::PositiveMax => "positive-max-bf16",
            Self::NegativeMax => "negative-max-bf16",
            Self::Mixed => "mixed-signed-max",
            Self::Nan => "nan-rejected-by-reference",
        }
    }

    fn value(self, index: usize) -> u16 {
        match self {
            Self::Zero => 0,
            Self::Equal => round_bf16(1.5),
            Self::Denormal => {
                if index.is_multiple_of(2) {
                    1
                } else {
                    0x8001
                }
            }
            Self::PositiveMax => 0x7f7f,
            Self::NegativeMax => 0xff7f,
            Self::Mixed => {
                if index.is_multiple_of(2) {
                    0x7f7f
                } else {
                    0xff7f
                }
            }
            Self::Nan => 0x7fc1,
        }
    }
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

fn run_case(
    context: &Context,
    module: &Module<'_>,
    rows: usize,
    width: usize,
    consumers: usize,
    pattern: Pattern,
) -> Result<Value> {
    let input_words: Vec<_> = (0..rows * width)
        .map(|index| pattern.value(index % width))
        .collect();
    let reference = projection_reference::quantize(&input_words, rows, width);
    if matches!(pattern, Pattern::Nan) {
        ensure!(
            reference.is_err(),
            "independent quantizer accepted BF16 NaN"
        );
        return Ok(json!({
            "rows": rows,
            "width": width,
            "pattern": pattern.name(),
            "consumers": consumers,
            "nan_policy": "rejected before device launch by the current finite-input contract",
            "all_passed": true,
        }));
    }
    let expected = reference?;
    let input_bytes: Vec<_> = input_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let input = upload(context, &input_bytes)?;
    let channels = 5;
    let mut weights = Vec::with_capacity(consumers);
    for projection in 0..consumers {
        let codes = (0..channels * width)
            .map(|index| {
                let magnitude = u8::try_from((index * 37 + projection * 11) % 127)?;
                Ok(magnitude | if index.is_multiple_of(2) { 0 } else { 128 })
            })
            .collect::<Result<Vec<_>>>()?;
        let weight = upload(context, &codes)?;
        let scale_words = (0..channels)
            .flat_map(|index| [0x3f80_u16, 0x3f00, 0x3f40, 0x3e80, 0x3f60][index].to_le_bytes())
            .collect::<Vec<_>>();
        let weight_scales = upload(context, &scale_words)?;
        weights.push((weight, weight_scales));
    }
    let mut baseline = Vec::with_capacity(consumers);
    let mut baseline_outputs = Vec::with_capacity(consumers);
    for _ in 0..consumers {
        let codes = upload(context, &vec![0xa5; rows * width])?;
        let scales = upload(context, &vec![0xa5; rows * 4])?;
        projections::quantize(
            &module.function("fp8_quantize_bf16")?,
            &input,
            &codes,
            &scales,
            rows,
            width,
        )?;
        context.synchronize()?;
        let (weight, weight_scales) = &weights[baseline.len()];
        let output = projections::run_linear_geometry(
            context,
            &module.function("fp8_linear_exact")?,
            &[&codes, weight, &scales, weight_scales],
            [rows, channels, width],
            [
                [
                    u32::try_from(channels.div_ceil(4))?,
                    u32::try_from(rows)?,
                    1,
                ],
                [128, 1, 1],
            ],
        )?;
        baseline_outputs.push(output.read(rows * channels)?);
        baseline.push((codes, scales));
    }
    let shared_codes = upload(context, &vec![0xa5; rows * width])?;
    let shared_scales = upload(context, &vec![0xa5; rows * 4])?;
    projections::quantize(
        &module.function("fp8_quantize_bf16")?,
        &input,
        &shared_codes,
        &shared_scales,
        rows,
        width,
    )?;
    context.synchronize()?;
    let mut shared_code_bytes = vec![0; rows * width];
    shared_codes.download(&mut shared_code_bytes)?;
    let mut shared_scale_bytes = vec![0; rows * 4];
    shared_scales.download(&mut shared_scale_bytes)?;
    let expected_scales: Vec<_> = expected
        .scales
        .iter()
        .flat_map(|scale| scale.to_le_bytes())
        .collect();
    let mut baseline_codes = vec![0; rows * width];
    let mut baseline_scales = vec![0; rows * 4];
    baseline[0].0.download(&mut baseline_codes)?;
    baseline[0].1.download(&mut baseline_scales)?;
    for (codes, scales) in &baseline {
        let mut actual_codes = vec![0; rows * width];
        let mut actual_scales = vec![0; rows * 4];
        codes.download(&mut actual_codes)?;
        scales.download(&mut actual_scales)?;
        ensure!(
            actual_codes == baseline_codes
                && actual_scales == baseline_scales
                && actual_codes == shared_code_bytes
                && actual_scales == shared_scale_bytes,
            "baseline FP8 quantization changed across repeated consumers"
        );
    }
    ensure!(
        shared_code_bytes == expected.codes && shared_scale_bytes == expected_scales,
        "shared FP8 quantization differs from the independent oracle"
    );
    for (index, (weight, weight_scales)) in weights.iter().enumerate() {
        let output = projections::run_linear_geometry(
            context,
            &module.function("fp8_linear_exact")?,
            &[&shared_codes, weight, &shared_scales, weight_scales],
            [rows, channels, width],
            [
                [
                    u32::try_from(channels.div_ceil(4))?,
                    u32::try_from(rows)?,
                    1,
                ],
                [128, 1, 1],
            ],
        )?;
        let actual = output.read(rows * channels)?;
        let expected = &baseline_outputs[index];
        ensure!(
            expected.0 == actual.0
                && expected
                    .1
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(actual.1.iter().map(|value| value.to_bits())),
            "shared FP8 projection output differs from separately quantized baseline: rows={rows} width={width} pattern={} consumer={index}",
            pattern.name()
        );
    }
    Ok(json!({
        "rows": rows,
        "width": width,
        "pattern": pattern.name(),
        "consumers": consumers,
        "codes_bitwise_equal": true,
        "scales_bitwise_equal": true,
        "independent_oracle_equal": true,
        "projection_outputs_bitwise_equal": true,
        "baseline_launches": consumers,
        "reuse_input_launches": 1,
        "launches_saved": consumers - 1,
        "all_passed": true,
    }))
}

pub(in crate::kernels) fn run(ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "FP8 quantization check requires SM120a PTX"
    );
    let context = Context::new(device)?;
    ensure!(
        (context.info().major, context.info().minor) == (12, 0),
        "FP8 quantization check requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let mut cases = Vec::new();
    for (rows, width) in [
        (1, 2048),
        (1, 5120),
        (1, 6144),
        (1, 10240),
        (1, 12288),
        (1, 17408),
        (1, 32768),
        (4, 5120),
        (4, 17408),
    ] {
        for consumers in [3, 2] {
            let patterns = [
                Pattern::Zero,
                Pattern::Equal,
                Pattern::Denormal,
                Pattern::PositiveMax,
                Pattern::NegativeMax,
                Pattern::Mixed,
                Pattern::Nan,
            ];
            for pattern in patterns {
                cases.push(run_case(
                    &context, &module, rows, width, consumers, pattern,
                )?);
            }
        }
    }
    Ok(json!({
        "kind": "fp8-exact-activation-reuse-v1",
        "device": context.info(),
        "widths": [2048, 5120, 6144, 10240, 12288, 17408, 32768],
        "all_passed": cases.iter().all(|case| case["all_passed"] == true),
        "cases": cases,
        "quantizer": "unchanged fp8_quantize_bf16: per-row amax/448, zero scale one, software E4M3FN RNE",
        "scope": "operator equality only; model parity and throughput require separate parent qualification",
    }))
}
