//! Resident embedding/norm feeding FP8 QKV/Z and BF16 A/B tensor-core projections.
use super::{
    causal_conv4,
    driver::{Buffer, Context, Function, Module},
    embedding_norm, gdn_output, gdn_prepare, gdn_recurrent, mlp, mlp_activation, nvfp4_linear,
    nvfp4_quantize, residual_add, residual_norm,
};
use crate::{
    entry_reference, gdn_recurrent_reference,
    kernels::{
        Bf16Projection, CausalConv4Weights, EmbeddingNormInput, Fp8Projection, ProjectionInput,
    },
    projection_reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(in crate::kernels) fn run(ptx: &str, device: i32, input: &ProjectionInput) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "projection PTX must target SM120a"
    );
    validate(input)?;
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "projection trial requires SM120"
    );
    let before = context.memory()?;
    let module = Module::load(&context, ptx)?;
    let nvfp4_quantization_fixtures = nvfp4_quantize::fixtures(&context, &module)?;
    let nvfp4_linear_fixtures = nvfp4_linear::fixtures(&context, &module)?;
    let mlp_activation_fixtures = mlp_activation::fixtures(&context, &module)?;
    let residual_add_fixtures = residual_add::fixtures(&context, &module)?;
    let quantization_fixtures = quantization_fixtures(&context, &module)?;
    let convolution_fixtures = causal_conv4::fixtures(&context, &module)?;
    let residual_norm_fixtures = residual_norm::fixtures(&context, &module)?;
    let gated_norm_fixtures = gdn_output::fixtures(&context, &module)?;
    let gdn_fixtures = gdn_prepare::fixtures(&context, &module)?;
    let recurrent_fixtures = gdn_recurrent::fixtures(&context, &module)?;
    let fixture = fixture()?;
    let fixture_cases = run_input(&context, &module, &fixture)?;
    let cases = run_input(&context, &module, input)?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":9,"kind":"qwen-layer-zero-reference-trial","device":info,
        "all_passed":cases.iter().chain(&fixture_cases).all(|c|c["passed"]==true),
        "cases":cases,"fixture_cases":fixture_cases,"quantization_fixtures":quantization_fixtures,
        "convolution_fixtures":convolution_fixtures,
        "nvfp4_linear_fixtures":nvfp4_linear_fixtures,"mlp_activation_fixtures":mlp_activation_fixtures,"residual_add_fixtures":residual_add_fixtures,"nvfp4_quantization_fixtures":nvfp4_quantization_fixtures,"residual_norm_fixtures":residual_norm_fixtures,"gated_norm_fixtures":gated_norm_fixtures,"gdn_fixtures":gdn_fixtures,"recurrent_fixtures":recurrent_fixtures,
        "memory_before":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_after":{"free_bytes":after.0,"total_bytes":after.1},
        "quantize_resources":module.function("fp8_quantize_bf16")?.resources()?,
        "linear_resources":module.function("fp8_linear")?.resources()?,
        "bf16_linear_resources":module.function("bf16_linear")?.resources()?,
        "convolution_resources":module.function("causal_conv4_bf16")?.resources()?,
        "qk_norm_resources":module.function("gdn_qk_norm")?.resources()?,
        "nvfp4_linear_resources":module.function("nvfp4_linear")?.resources()?,"mlp_activation_resources":module.function("mlp_silu_product")?.resources()?,"residual_add_resources":module.function("residual_add_bf16")?.resources()?,"nvfp4_quantize_resources":module.function("nvfp4_quantize_bf16")?.resources()?,"residual_norm_resources":module.function("residual_norm_bf16")?.resources()?,"gated_norm_resources":module.function("gdn_gated_rms_norm")?.resources()?,"gate_resources":module.function("gdn_gates")?.resources()?,"recurrent_resources":module.function("gdn_recurrent")?.resources()?,
        "activation_profile":"E4M3FN dynamic token scale in FP32; amax/448, zero scale replaced with 1; RNE finite saturation",
        "gpu_chain":"embedding/norm -> FP8 quantization -> QKV -> convolution/SiLU -> Q/K normalization; normalized BF16 -> A/B -> beta/log-decay/decay gates -> recurrent matrix update -> gated RMSNorm using resident Z -> FP8 output projection -> BF16 residual add and zero-centered post-attention norm -> MLP NVFP4 gate/up -> BF16 SiLU product -> NVFP4 down -> second BF16 residual; device intermediates stay resident",
        "timing_collected":false,"full_model_executed":false}),
    )
}

fn validate(input: &ProjectionInput) -> Result<()> {
    let entry = &input.entry;
    ensure!(
        (1..=16).contains(&entry.batches.len())
            && (1..=8).contains(&input.projections.len())
            && (1..=8).contains(&input.bf16_projections.len()),
        "invalid trial case count"
    );
    for tokens in &entry.batches {
        entry_reference::embedding_norm(
            &entry.table,
            tokens,
            &entry.weight,
            entry.width,
            entry.epsilon,
        )?;
    }
    for p in &input.projections {
        ensure!(
            (1..=262144).contains(&p.channels),
            "invalid projection channels"
        );
        ensure!(
            p.weights.len() == p.channels * entry.width && p.scales.len() == p.channels * 2,
            "projection extent mismatch"
        );
        ensure!(
            p.weights.iter().all(|v| v & 127 != 127),
            "nonfinite checkpoint FP8 code"
        );
        ensure!(
            p.scales.as_chunks::<2>().0.iter().all(|bytes| {
                let scale = entry_reference::bf16_to_f32(u16::from_le_bytes(*bytes));
                scale.is_finite() && scale > 0.0
            }),
            "invalid projection scale"
        );
    }
    for p in &input.bf16_projections {
        ensure!(
            (1..=262144).contains(&p.channels),
            "invalid BF16 projection channels"
        );
        ensure!(
            p.weights.len() == p.channels * entry.width * 2,
            "BF16 projection extent mismatch"
        );
        ensure!(
            p.weights
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| { entry_reference::bf16_to_f32(u16::from_le_bytes(*b)).is_finite() }),
            "nonfinite checkpoint BF16 weight"
        );
    }
    if let Some(conv) = &input.convolution {
        let projection = input
            .projections
            .get(conv.projection)
            .ok_or_else(|| anyhow::anyhow!("convolution projection index outside input"))?;
        ensure!(
            projection.channels <= 32768,
            "too many convolution channels"
        );
        ensure!(
            conv.weights.len() == projection.channels * 4 * 2,
            "convolution weight extent mismatch"
        );
        ensure!(
            conv.weights
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| { entry_reference::bf16_to_f32(u16::from_le_bytes(*b)).is_finite() }),
            "nonfinite convolution weight"
        );
    }
    if let Some(gdn) = &input.gdn {
        gdn_prepare::validate_connections(input, gdn)?;
    }
    if let Some(output) = &input.gdn_output {
        gdn_output::validate(input, output)?;
    }
    if let Some(weights) = &input.mlp {
        ensure!(
            input.post_attention_norm.is_some(),
            "MLP requires post-attention norm"
        );
        mlp::validate(weights, input.entry.width)?;
    }
    if let Some(norm) = &input.post_attention_norm {
        ensure!(
            input.gdn_output.is_some(),
            "post-attention norm needs GDN output"
        );
        residual_norm::validate(norm, input.entry.width)?;
    }
    Ok(())
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
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

fn run_input(
    context: &Context,
    module: &Module<'_>,
    input: &ProjectionInput,
) -> Result<Vec<Value>> {
    validate(input)?;
    let table = upload(context, &input.entry.table)?;
    let norm = upload(context, &input.entry.weight)?;
    let projections: Vec<_> = input
        .projections
        .iter()
        .map(|p| Ok((upload(context, &p.weights)?, upload(context, &p.scales)?)))
        .collect::<Result<_>>()?;
    let bf16_projections: Vec<_> = input
        .bf16_projections
        .iter()
        .map(|p| upload(context, &p.weights))
        .collect::<Result<_>>()?;
    let mut cases = Vec::new();
    for tokens in &input.entry.batches {
        let mut convolution = None;
        let mut z_output = None;
        let mut bf16_outputs = Vec::new();
        let expected = entry_reference::embedding_norm(
            &input.entry.table,
            tokens,
            &input.entry.weight,
            input.entry.width,
            input.entry.epsilon,
        )?;
        let token_bytes: Vec<_> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ids = upload(context, &token_bytes)?;
        let count = tokens.len() * input.entry.width;
        let residual = upload(context, &vec![0xa5; count * 2])?;
        let normalized = upload(context, &vec![0xa5; count * 2])?;
        let unrounded = upload(context, &vec![0xff; count * 4])?;
        let function = module.function("embedding_norm_bf16")?;
        embedding_norm::launch(
            &function,
            &[&table, &ids, &norm, &residual, &normalized, &unrounded],
            tokens.len(),
            input.entry.width,
            input.entry.epsilon,
        )?;
        context.synchronize()?;
        ensure!(
            words(&residual, count)? == expected.residual,
            "chained embedding mismatch"
        );
        ensure!(
            words(&normalized, count)? == expected.normalized,
            "chained normalized BF16 differs from scalar reference"
        );
        let expected_quant =
            projection_reference::quantize(&expected.normalized, tokens.len(), input.entry.width)?;
        let codes = upload(context, &vec![127; count])?;
        let scales = upload(context, &vec![0xff; tokens.len() * 4])?;
        quantize(
            &module.function("fp8_quantize_bf16")?,
            &normalized,
            &codes,
            &scales,
            tokens.len(),
            input.entry.width,
        )?;
        context.synchronize()?;
        let mut actual_codes = vec![0; count];
        codes.download(&mut actual_codes)?;
        ensure!(
            actual_codes == expected_quant.codes,
            "GPU FP8 activation codes differ from scalar reference"
        );
        ensure!(
            floats(&scales, tokens.len())? == expected_quant.scales,
            "GPU FP8 activation scales differ from scalar reference"
        );
        for (index, (projection, (w, sw))) in input.projections.iter().zip(&projections).enumerate()
        {
            let shape = [tokens.len(), projection.channels, input.entry.width];
            let device_output = run_linear(
                context,
                &module.function("fp8_linear")?,
                &[&codes, w, &scales, sw],
                shape,
            )?;
            let actual = device_output.read(tokens.len() * projection.channels)?;
            let weight_scales: Vec<_> = projection
                .scales
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            let reference = projection_reference::linear(
                &expected_quant,
                &projection.weights,
                &weight_scales,
                input.entry.width,
            )?;
            let mut report = compare(&actual.0, &actual.1, &reference)?;
            report["projection"] = json!(projection.name);
            report["weight_format"] = json!("E4M3FN with BF16 per-channel scale");
            report["shape_mnk"] = json!(shape);
            report["tokens"] = json!(tokens);
            report["activation_codes_exact"] = json!(true);
            report["activation_scales_exact"] = json!(true);
            report["free_device_bytes_with_allocations"] = json!(context.memory()?.0);
            if let Some(conv) = &input.convolution
                && conv.projection == index
            {
                let result = causal_conv4::check(
                    context,
                    module,
                    &device_output.bf16,
                    &actual.0,
                    &conv.weights,
                    [tokens.len(), projection.channels],
                )?;
                ensure!(
                    result.report["all_passed"] == true,
                    "chained convolution failed"
                );
                report["convolution"] = result.report.clone();
                convolution = Some(result);
            }
            cases.push(report);
            if input
                .gdn_output
                .as_ref()
                .is_some_and(|o| o.z_projection == index)
            {
                z_output = Some((device_output, actual.0));
            }
        }
        for (projection, w) in input.bf16_projections.iter().zip(&bf16_projections) {
            let shape = [tokens.len(), projection.channels, input.entry.width];
            let device_output = run_linear(
                context,
                &module.function("bf16_linear")?,
                &[&normalized, w],
                shape,
            )?;
            let actual = device_output.read(tokens.len() * projection.channels)?;
            let weights: Vec<_> = projection
                .weights
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            let reference = projection_reference::linear_bf16(
                &expected.normalized,
                &weights,
                tokens.len(),
                input.entry.width,
            )?;
            let mut report = compare(&actual.0, &actual.1, &reference)?;
            report["projection"] = json!(projection.name);
            report["weight_format"] = json!("BF16");
            report["activation_format"] = json!("resident normalized BF16, no quantization");
            report["shape_mnk"] = json!(shape);
            report["tokens"] = json!(tokens);
            report["free_device_bytes_with_allocations"] = json!(context.memory()?.0);
            cases.push(report);
            bf16_outputs.push((device_output, actual.0));
        }
        if input.gdn.is_some() {
            let convolution = convolution
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing resident convolution"))?;
            cases.push(check_gdn(
                context,
                module,
                input,
                GdnInputs {
                    convolution,
                    bf16_outputs: &bf16_outputs,
                    tokens,
                    z: z_output.as_ref(),
                    residual: &residual,
                    residual_words: &expected.residual,
                },
            )?);
        }
    }
    Ok(cases)
}

struct GdnInputs<'a, 'ctx> {
    convolution: &'a causal_conv4::CheckedConvolution<'ctx>,
    bf16_outputs: &'a [(LinearOutput<'ctx>, Vec<u16>)],
    tokens: &'a [u32],
    z: Option<&'a (LinearOutput<'ctx>, Vec<u16>)>,
    residual: &'a Buffer<'ctx>,
    residual_words: &'a [u16],
}

fn check_gdn(
    context: &Context,
    module: &Module<'_>,
    input: &ProjectionInput,
    chain: GdnInputs<'_, '_>,
) -> Result<Value> {
    let GdnInputs {
        convolution,
        bf16_outputs,
        tokens,
        z,
        residual,
        residual_words,
    } = chain;
    let gdn = input
        .gdn
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing GDN weights"))?;
    let a = &bf16_outputs[gdn.a_projection];
    let b = &bf16_outputs[gdn.b_projection];
    let prepared = gdn_prepare::check(
        context,
        module,
        gdn_prepare::Input {
            qkv: &convolution.output,
            qkv_words: &convolution.words,
            a: &a.0.bf16,
            a_words: &a.1,
            b: &b.0.bf16,
            b_words: &b.1,
        },
        gdn,
        tokens.len(),
    )?;
    let recurrent = gdn_recurrent::check(
        context,
        module,
        gdn_recurrent::Input {
            q: &prepared.q,
            k: &prepared.k,
            qkv: &convolution.output,
            beta: &prepared.beta,
            decay: &prepared.decay,
            host: gdn_recurrent_reference::Input {
                q: &prepared.host.q,
                k: &prepared.host.k,
                qkv: &convolution.words,
                beta: &prepared.host.beta,
                decay: &prepared.host.decay,
            },
        },
        &gdn_recurrent_reference::Shape {
            rows: tokens.len(),
            key_heads: gdn.key_heads,
            value_heads: gdn.value_heads,
            width: gdn.width,
        },
    )?;
    let output = if let Some(weights) = &input.gdn_output {
        let z = z.ok_or_else(|| anyhow::anyhow!("missing resident Z"))?;
        Some(gdn_output::check(
            context,
            module,
            gdn_output::Input {
                x: &recurrent.output,
                x_words: &recurrent.words,
                z: &z.0.bf16,
                z_words: &z.1,
            },
            weights,
            [tokens.len(), gdn.value_heads, gdn.width],
        )?)
    } else {
        None
    };
    let post_attention = if let Some(norm) = &input.post_attention_norm {
        let output = output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing resident attention output"))?;
        Some(residual_norm::check(
            context,
            module,
            residual_norm::Input {
                residual,
                residual_words,
                branch: &output.output,
                branch_words: &output.words,
            },
            norm,
            [tokens.len(), input.entry.width],
        )?)
    } else {
        None
    };
    let mlp_report = if let Some(weights) = &input.mlp {
        let norm = post_attention
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing resident post-attention norm"))?;
        Some(mlp::check(
            context,
            module,
            norm,
            weights,
            [tokens.len(), input.entry.width],
        )?)
    } else {
        None
    };
    let whole_layer = if let Some(mlp) = &mlp_report {
        Some(compare_layer(
            input,
            tokens,
            &mlp.words,
            &convolution.history,
            &recurrent.state,
        )?)
    } else {
        None
    };
    let passed = whole_layer.as_ref().is_none_or(|r| r["all_passed"] == true);
    Ok(
        json!({"operation":"layer_zero_components","tokens":tokens,"passed":passed,
        "preparation":prepared.report,"recurrence":recurrent.report,"output":output.map(|v|v.report),"post_attention":post_attention.map(|v|v.report),"mlp":mlp_report.map(|v|v.report),"whole_layer_reference":whole_layer}),
    )
}

pub(super) fn quantize(
    function: &Function<'_, '_>,
    input: &Buffer<'_>,
    codes: &Buffer<'_>,
    scales: &Buffer<'_>,
    rows: usize,
    width: usize,
) -> Result<()> {
    let mut pointers = [input.pointer(), codes.pointer(), scales.pointer()];
    let mut width = u32::try_from(width)?;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.push((&mut width as *mut u32).cast());
    // SAFETY: BF16 input and output extents are validated; allocations are disjoint,
    // all rows use 256 threads, and the caller synchronizes before reading outputs.
    unsafe { function.launch([u32::try_from(rows)?, 1, 1], [256, 1, 1], 0, &mut args) }
}

pub(super) struct LinearOutput<'a> {
    pub(super) bf16: Buffer<'a>,
    unrounded: Buffer<'a>,
}
impl LinearOutput<'_> {
    pub(super) fn read(&self, count: usize) -> Result<(Vec<u16>, Vec<f32>)> {
        Ok((words(&self.bf16, count)?, floats(&self.unrounded, count)?))
    }
}

pub(super) fn run_linear<'a>(
    context: &'a Context,
    function: &Function<'_, '_>,
    buffers: &[&Buffer<'_>],
    shape: [usize; 3],
) -> Result<LinearOutput<'a>> {
    let [m, n, k] = shape;
    let count = m * n;
    let output = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers: Vec<_> = buffers.iter().map(|b| b.pointer()).collect();
    pointers.extend([output.pointer(), unrounded.pointer()]);
    let mut dimensions = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|p| (p as *mut u32).cast()));
    // SAFETY: Callers supply four input pointers for fp8_linear or two for
    // bf16_linear, followed here by BF16/FP32 outputs and three u32 dimensions.
    // Validated extents, masked tails and allocation lifetimes cover every access.
    unsafe {
        function.launch(
            [
                u32::try_from(n.div_ceil(8))?,
                u32::try_from(m.div_ceil(16))?,
                1,
            ],
            [32, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    Ok(LinearOutput {
        bf16: output,
        unrounded,
    })
}

pub(super) fn compare(
    bf16: &[u16],
    actual: &[f32],
    reference: &projection_reference::LinearReference,
) -> Result<Value> {
    ensure!(
        actual.len() == reference.unrounded.len() && bf16.len() == actual.len(),
        "projection result extent mismatch"
    );
    let mut maximum = 0.0_f32;
    let mut mismatches = 0;
    let mut rounding = 0;
    let mut bf16_differences = 0;
    for (i, (&a, &b)) in actual.iter().zip(&reference.unrounded).enumerate() {
        ensure!(
            a.is_finite() && entry_reference::bf16_to_f32(bf16[i]).is_finite(),
            "nonfinite projection output"
        );
        let error = (a - b).abs();
        maximum = maximum.max(error);
        mismatches += usize::from(f64::from(error) > 1e-6 + 2e-6 * reference.absolute_sums[i]);
        rounding += usize::from(bf16[i] != entry_reference::round_bf16(a));
        bf16_differences += usize::from(bf16[i] != reference.normalized[i]);
    }
    Ok(
        json!({"elements":actual.len(),"max_abs_error":maximum,"numerical_mismatches":mismatches,"rounding_mismatches":rounding,"bf16_reference_differences":bf16_differences,"tolerance":"1e-6 + 2e-6 * scaled sum(abs(products))","passed":mismatches==0 && rounding==0}),
    )
}

fn fixture() -> Result<ProjectionInput> {
    let width = 71;
    let vocabulary = 19;
    let channels = 13;
    let table: Vec<_> = (0..vocabulary * width)
        .flat_map(|i| {
            entry_reference::round_bf16(((i * 13 % 31) as f32 - 15.0) / 8.0).to_le_bytes()
        })
        .collect();
    let weight = vec![0; width * 2];
    let weights = (0..channels * width)
        .map(|i| projection_reference::encode((i * 7 % 15) as f32 - 7.0))
        .collect::<Result<_>>()?;
    let scales = (0..channels)
        .flat_map(|i| entry_reference::round_bf16(0.25 * (i % 4 + 1) as f32).to_le_bytes())
        .collect();
    Ok(ProjectionInput {
        entry: EmbeddingNormInput {
            table,
            weight,
            width,
            epsilon: 1e-6,
            batches: vec![(0..17).collect()],
        },
        projections: vec![Fp8Projection {
            name: "signed-tail-fixture".into(),
            weights,
            scales,
            channels,
        }],
        bf16_projections: vec![Bf16Projection {
            name: "signed-bf16-tail-fixture".into(),
            weights: (0..channels * width)
                .flat_map(|i| {
                    entry_reference::round_bf16(((i * 11 % 43) as f32 - 21.0) / 16.0).to_le_bytes()
                })
                .collect(),
            channels,
        }],
        convolution: Some(CausalConv4Weights {
            projection: 0,
            weights: (0..channels * 4)
                .flat_map(|i| {
                    entry_reference::round_bf16(((i * 5 % 17) as f32 - 8.0) / 16.0).to_le_bytes()
                })
                .collect(),
        }),
        gdn: None,
        gdn_output: None,
        post_attention_norm: None,
        mlp: None,
    })
}

fn quantization_fixtures(context: &Context, module: &Module<'_>) -> Result<Value> {
    let mut levels = Vec::new();
    for code in 0..=126 {
        let value = projection_reference::decode(code);
        for sign in [1.0, -1.0] {
            levels.push(entry_reference::round_bf16(value * sign));
            if code < 126 {
                levels.push(entry_reference::round_bf16(
                    (value + projection_reference::decode(code + 1)) * 0.5 * sign,
                ));
            }
        }
    }
    let width = levels.len();
    let mut input = levels;
    input.extend(vec![0; width]);
    input.extend((0..width).map(|i| if i % 2 == 0 { 1_u16 } else { 0x8001 }));
    let expected = projection_reference::quantize(&input, 3, width)?;
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = upload(context, &bytes)?;
    let codes = upload(context, &vec![127; expected.codes.len()])?;
    let scales = upload(context, &[0xff; 12])?;
    quantize(
        &module.function("fp8_quantize_bf16")?,
        &input,
        &codes,
        &scales,
        3,
        width,
    )?;
    context.synchronize()?;
    let mut actual = vec![0; expected.codes.len()];
    codes.download(&mut actual)?;
    ensure!(
        actual == expected.codes,
        "FP8 code/tie/zero/tiny fixture mismatch"
    );
    ensure!(
        floats(&scales, 3)? == expected.scales,
        "FP8 fixture scale mismatch"
    );
    Ok(
        json!({"rows":3,"width":width,"elements":actual.len(),"codes_exact":true,"scales_exact":true,"coverage":"every finite signed E4M3 value and adjacent midpoint, signed zero, all-zero row, minimum BF16 subnormal"}),
    )
}

fn compare_layer(
    input: &ProjectionInput,
    tokens: &[u32],
    output: &[u16],
    history: &[u16],
    state: &[f32],
) -> Result<Value> {
    let expected = crate::qwen_gdn_layer_reference::run(input, tokens)?;
    let gdn = input
        .gdn
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing GDN comparison shape"))?;
    let decode = |words: &[u16]| {
        words
            .iter()
            .copied()
            .map(entry_reference::bf16_to_f32)
            .collect::<Vec<_>>()
    };
    let mut hidden = crate::layer_comparison_reference::compare_partitioned(
        &decode(output),
        &decode(&expected.output),
        input.entry.width,
    )?;
    hidden["bf16_bit_differences"] = json!(
        output
            .iter()
            .zip(&expected.output)
            .filter(|(a, b)| a != b)
            .count()
    );
    let history_width = (2 * gdn.key_heads + gdn.value_heads) * gdn.width;
    let mut history_report = crate::layer_comparison_reference::compare_partitioned(
        &decode(history),
        &decode(&expected.convolution_history),
        history_width,
    )?;
    history_report["bf16_bit_differences"] = json!(
        history
            .iter()
            .zip(&expected.convolution_history)
            .filter(|(a, b)| a != b)
            .count()
    );
    let recurrent = crate::layer_comparison_reference::compare_partitioned(
        state,
        &expected.recurrent_state,
        gdn.width * gdn.width,
    )?;
    let passed = hidden["all_passed"] == true
        && history_report["all_passed"] == true
        && recurrent["all_passed"] == true;
    Ok(
        json!({"all_passed":passed,"hidden":hidden,"convolution_history":history_report,"recurrent_state":recurrent,
        "full_layer_reference_compared":true,
        "reference_inputs":"artifact weights, original tokens and zero initial states only; no device outputs",
        "reference_profile":"independent scalar operation chain, f64 dot/norm/exp oracles and ordered-FP32 recurrence; chosen FP32 activation quantization profile",
        "full_model_or_logits_compared":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_comparison_rejects_wrong_rounding_large_error_and_nonfinite_output() {
        let reference = projection_reference::LinearReference {
            unrounded: vec![1.0],
            normalized: vec![0x3f80],
            absolute_sums: vec![2.0],
        };
        assert_eq!(
            compare(&[0x3f80], &[1.0], &reference).unwrap()["passed"],
            true
        );
        assert_eq!(
            compare(&[0x3f81], &[1.0], &reference).unwrap()["passed"],
            false
        );
        assert_eq!(
            compare(&[0x4000], &[2.0], &reference).unwrap()["passed"],
            false
        );
        assert!(compare(&[0x7fc0], &[f32::NAN], &reference).is_err());
        assert!(compare(&[], &[1.0], &reference).is_err());
    }
}
