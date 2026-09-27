//! Chained resident embedding, dynamic FP8 quantization and FP8 tensor-core linear checks.
use super::{
    driver::{Buffer, Context, Function, Module},
    embedding_norm,
};
use crate::{
    entry_reference, fp8_reference,
    kernels::{EmbeddingNormInput, Fp8Projection, ProjectionInput},
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
    let quantization_fixtures = quantization_fixtures(&context, &module)?;
    let fixture = fixture()?;
    let fixture_cases = run_input(&context, &module, &fixture)?;
    let cases = run_input(&context, &module, input)?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"qwen-fp8-projection-trial","device":info,
        "all_passed":cases.iter().chain(&fixture_cases).all(|c|c["passed"]==true),
        "cases":cases,"fixture_cases":fixture_cases,"quantization_fixtures":quantization_fixtures,
        "memory_before":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_after":{"free_bytes":after.0,"total_bytes":after.1},
        "quantize_resources":module.function("fp8_quantize_bf16")?.resources()?,
        "linear_resources":module.function("fp8_linear")?.resources()?,
        "activation_profile":"E4M3FN dynamic token scale in FP32; amax/448, zero scale replaced with 1; RNE finite saturation",
        "gpu_chain":"embedding/norm -> FP8 quantization -> FP8 MMA; no host replacement of intermediate device data",
        "timing_collected":false,"full_model_executed":false}),
    )
}

fn validate(input: &ProjectionInput) -> Result<()> {
    let entry = &input.entry;
    ensure!(
        (1..=16).contains(&entry.batches.len()) && (1..=8).contains(&input.projections.len()),
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
    let mut cases = Vec::new();
    for tokens in &input.entry.batches {
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
            fp8_reference::quantize(&expected.normalized, tokens.len(), input.entry.width)?;
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
        for (projection, (w, sw)) in input.projections.iter().zip(&projections) {
            let shape = [tokens.len(), projection.channels, input.entry.width];
            let actual = run_linear(
                context,
                &module.function("fp8_linear")?,
                [&codes, w, &scales, sw],
                shape,
            )?;
            let weight_scales: Vec<_> = projection
                .scales
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            let reference = fp8_reference::linear(
                &expected_quant,
                &projection.weights,
                &weight_scales,
                input.entry.width,
            )?;
            let mut report = compare(&actual.0, &actual.1, &reference)?;
            report["projection"] = json!(projection.name);
            report["shape_mnk"] = json!(shape);
            report["tokens"] = json!(tokens);
            report["activation_codes_exact"] = json!(true);
            report["activation_scales_exact"] = json!(true);
            report["free_device_bytes_with_allocations"] = json!(context.memory()?.0);
            cases.push(report);
        }
    }
    Ok(cases)
}

fn quantize(
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

fn run_linear(
    context: &Context,
    function: &Function<'_, '_>,
    buffers: [&Buffer<'_>; 4],
    shape: [usize; 3],
) -> Result<(Vec<u16>, Vec<f32>)> {
    let [m, n, k] = shape;
    let count = m * n;
    let output = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers = [
        buffers[0].pointer(),
        buffers[1].pointer(),
        buffers[2].pointer(),
        buffers[3].pointer(),
        output.pointer(),
        unrounded.pointer(),
    ];
    let mut dimensions = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|p| (p as *mut u32).cast()));
    // SAFETY: Six pointers and three dimensions match fp8_linear's ABI. Every
    // tensor has validated dimensions, tail loads/stores are masked by the kernel,
    // and allocations stay alive through synchronization and download.
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
    Ok((words(&output, count)?, floats(&unrounded, count)?))
}

fn compare(
    bf16: &[u16],
    actual: &[f32],
    reference: &fp8_reference::LinearReference,
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
        .map(|i| fp8_reference::encode((i * 7 % 15) as f32 - 7.0))
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
    })
}

fn quantization_fixtures(context: &Context, module: &Module<'_>) -> Result<Value> {
    let mut levels = Vec::new();
    for code in 0..=126 {
        let value = fp8_reference::decode(code);
        for sign in [1.0, -1.0] {
            levels.push(entry_reference::round_bf16(value * sign));
            if code < 126 {
                levels.push(entry_reference::round_bf16(
                    (value + fp8_reference::decode(code + 1)) * 0.5 * sign,
                ));
            }
        }
    }
    let width = levels.len();
    let mut input = levels;
    input.extend(vec![0; width]);
    input.extend((0..width).map(|i| if i % 2 == 0 { 1_u16 } else { 0x8001 }));
    let expected = fp8_reference::quantize(&input, 3, width)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_comparison_rejects_wrong_rounding_large_error_and_nonfinite_output() {
        let reference = fp8_reference::LinearReference {
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
