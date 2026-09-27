//! Resident GDN gated normalization followed by its FP8 output projection.
use super::{
    driver::{Buffer, Context, Module},
    projections,
};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    gated_norm_reference as reference,
    kernels::{GdnOutputWeights, ProjectionInput},
    projection_reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) x: &'a Buffer<'ctx>,
    pub(super) x_words: &'a [u16],
    pub(super) z: &'a Buffer<'ctx>,
    pub(super) z_words: &'a [u16],
}

pub(super) fn validate(input: &ProjectionInput, weights: &GdnOutputWeights) -> Result<()> {
    let gdn = input
        .gdn
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GDN output needs recurrence"))?;
    let z = input
        .projections
        .get(weights.z_projection)
        .ok_or_else(|| anyhow::anyhow!("GDN Z projection index invalid"))?;
    let inner = gdn.value_heads * gdn.width;
    ensure!(z.channels == inner, "GDN Z projection shape mismatch");
    ensure!(
        input
            .convolution
            .as_ref()
            .is_none_or(|c| c.projection != weights.z_projection),
        "GDN Z and QKV projection must differ"
    );
    ensure!(
        weights.norm.len() == gdn.width * 2 && weights.epsilon.is_finite() && weights.epsilon > 0.0,
        "GDN norm shape/epsilon invalid"
    );
    ensure!(
        decode(&weights.norm)
            .iter()
            .all(|&x| bf16_to_f32(x).is_finite()),
        "GDN norm nonfinite weight"
    );
    let p = &weights.projection;
    ensure!(
        p.channels == input.entry.width
            && p.weights.len() == p.channels * inner
            && p.scales.len() == p.channels * 2,
        "GDN output projection extent mismatch"
    );
    ensure!(
        p.weights.iter().all(|x| x & 127 != 127),
        "GDN output projection nonfinite FP8 weight"
    );
    ensure!(
        decode(&p.scales).iter().all(|&x| {
            let v = bf16_to_f32(x);
            v.is_finite() && v > 0.0
        }),
        "GDN output projection invalid scale"
    );
    Ok(())
}

pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: Input<'_, '_>,
    weights: &GdnOutputWeights,
    shape: [usize; 3],
) -> Result<Value> {
    let [rows, heads, width] = shape;
    let norm = normalize(
        context,
        module,
        input,
        &decode(&weights.norm),
        rows * heads,
        width,
        weights.epsilon,
    )?;
    let inner = heads * width;
    let expected_quant = projection_reference::quantize(&norm.words, rows, inner)?;
    let codes = upload(context, &vec![127; rows * inner])?;
    let scales = upload(context, &vec![0xff; rows * 4])?;
    projections::quantize(
        &module.function("fp8_quantize_bf16")?,
        &norm.device,
        &codes,
        &scales,
        rows,
        inner,
    )?;
    context.synchronize()?;
    let mut actual_codes = vec![0; rows * inner];
    codes.download(&mut actual_codes)?;
    ensure!(
        actual_codes == expected_quant.codes,
        "GDN output activation FP8 codes differ"
    );
    ensure!(
        floats(&scales, rows)? == expected_quant.scales,
        "GDN output activation scales differ"
    );
    let weight = upload(context, &weights.projection.weights)?;
    let weight_scale = upload(context, &weights.projection.scales)?;
    let channels = weights.projection.channels;
    let result = projections::run_linear(
        context,
        &module.function("fp8_linear")?,
        &[&codes, &weight, &scales, &weight_scale],
        [rows, channels, inner],
    )?;
    let actual = result.read(rows * channels)?;
    let expected = projection_reference::linear(
        &expected_quant,
        &weights.projection.weights,
        &decode(&weights.projection.scales),
        inner,
    )?;
    let mut report = projections::compare(&actual.0, &actual.1, &expected)?;
    ensure!(
        report["passed"] == true,
        "GDN output projection numerical mismatch: {report}"
    );
    report["shape_mnk"] = json!([rows, channels, inner]);
    report["activation_codes_exact"] = json!(true);
    report["activation_scales_exact"] = json!(true);
    Ok(
        json!({"all_passed":true,"gated_norm":norm.report,"output_projection":report,
        "device_inputs_resident":true,"model_executable":false,
        "scope":"layer-zero attention components chained from resident embedding/norm; no independent full-layer or model logit parity yet"}),
    )
}

struct Normalized<'a> {
    device: Buffer<'a>,
    words: Vec<u16>,
    report: Value,
}

fn normalize<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: Input<'_, '_>,
    weight: &[u16],
    groups: usize,
    width: usize,
    epsilon: f32,
) -> Result<Normalized<'a>> {
    let expected = reference::run(input.x_words, input.z_words, weight, groups, width, epsilon)?;
    let count = groups * width;
    let weight_device = upload(context, &word_bytes(weight))?;
    let output = upload(context, &vec![0xa5; count * 2])?;
    let normalized = upload(context, &vec![0xff; count * 4])?;
    let weighted = upload(context, &vec![0xa5; count * 2])?;
    let silu = upload(context, &vec![0xff; count * 4])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers = [
        input.x.pointer(),
        input.z.pointer(),
        weight_device.pointer(),
        output.pointer(),
        normalized.pointer(),
        weighted.pointer(),
        silu.pointer(),
        unrounded.pointer(),
    ];
    let mut dimensions = [u32::try_from(groups)?, u32::try_from(width)?];
    let mut epsilon = epsilon;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|x| (x as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|x| (x as *mut u32).cast()));
    args.push((&mut epsilon as *mut f32).cast());
    // SAFETY: Eight disjoint allocations, two u32 extents and FP32 epsilon match
    // the ABI. Independent scalar validation covers extents/domains. All 256
    // threads participate in reductions; synchronization precedes reads/drop.
    unsafe {
        module.function("gdn_gated_rms_norm")?.launch(
            [dimensions[0], 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let actual = reference::GatedNorm {
        normalized: floats(&normalized, count)?,
        weighted: words(&weighted, count)?,
        silu: floats(&silu, count)?,
        unrounded: floats(&unrounded, count)?,
        output: words(&output, count)?,
    };
    let report = compare(&actual, &expected, weight)?;
    Ok(Normalized {
        device: output,
        words: actual.output,
        report,
    })
}

fn compare(
    actual: &reference::GatedNorm,
    expected: &reference::GatedNorm,
    weight: &[u16],
) -> Result<Value> {
    let count = expected.output.len();
    ensure!(
        !weight.is_empty() && count > 0 && count.is_multiple_of(weight.len()),
        "gated norm comparison shape mismatch"
    );
    ensure!(
        [
            actual.output.len(),
            actual.normalized.len(),
            actual.weighted.len(),
            actual.silu.len(),
            actual.unrounded.len(),
            expected.normalized.len(),
            expected.weighted.len(),
            expected.silu.len(),
            expected.unrounded.len()
        ]
        .iter()
        .all(|&n| n == count),
        "gated norm comparison extent mismatch"
    );
    let mut norm_error = 0.0_f64;
    let mut silu_error = 0.0_f64;
    let mut final_error = 0.0_f64;
    let mut norm_bf16_differences = 0;
    let mut weight_differences = 0;
    let mut output_differences = 0;
    for i in 0..count {
        let norm = actual.normalized[i];
        let gate = actual.silu[i];
        let out = actual.unrounded[i];
        ensure!(
            [
                norm,
                gate,
                out,
                bf16_to_f32(actual.output[i]),
                bf16_to_f32(actual.weighted[i])
            ]
            .iter()
            .all(|v| v.is_finite()),
            "gated norm nonfinite output"
        );
        let ne = (f64::from(norm) - f64::from(expected.normalized[i])).abs();
        let ge = (f64::from(gate) - f64::from(expected.silu[i])).abs();
        ensure!(
            ne <= 2e-6 + 2e-6 * f64::from(expected.normalized[i]).abs(),
            "gated norm FP32 mismatch at {i}"
        );
        ensure!(
            ge <= 3e-6 + 5e-6 * f64::from(expected.silu[i]).abs(),
            "gated norm SiLU mismatch at {i}"
        );
        let weighted =
            round_bf16(bf16_to_f32(round_bf16(norm)) * bf16_to_f32(weight[i % weight.len()]));
        ensure!(
            weighted == actual.weighted[i],
            "gated norm intermediate BF16 rounding mismatch at {i}"
        );
        let final_value = bf16_to_f32(weighted) * gate;
        ensure!(
            out.to_bits() == final_value.to_bits(),
            "gated norm final multiply mismatch at {i}"
        );
        ensure!(
            actual.output[i] == round_bf16(out),
            "gated norm final BF16 rounding mismatch at {i}"
        );
        norm_error = norm_error.max(ne);
        silu_error = silu_error.max(ge);
        final_error = final_error.max((f64::from(out) - f64::from(expected.unrounded[i])).abs());
        norm_bf16_differences +=
            usize::from(round_bf16(norm) != round_bf16(expected.normalized[i]));
        weight_differences += usize::from(weighted != expected.weighted[i]);
        output_differences += usize::from(actual.output[i] != expected.output[i]);
    }
    Ok(
        json!({"all_passed":true,"elements":count,"normalized_max_abs_error":norm_error,"silu_max_abs_error":silu_error,
        "full_reference_output_max_abs_error":final_error,"normalized_bf16_reference_differences":norm_bf16_differences,
        "weighted_bf16_reference_differences":weight_differences,"output_bf16_reference_differences":output_differences,
        "intermediate_and_final_rounding_exact":true,"normalized_tolerance":"2e-6 + 2e-6 * abs(reference)","silu_tolerance":"3e-6 + 5e-6 * abs(reference)",
        "profile":"FP32 RMSNorm -> BF16 -> direct BF16 gamma multiplication -> BF16 -> FP32 SiLU gate -> BF16"}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for (groups, width) in [(7, 8), (3, 1), (3, 256)] {
        let count = groups * width;
        let mut x: Vec<_> = (0..count)
            .map(|i| round_bf16(((i * 7 % 31) as f32 - 15.0) / 16.0))
            .collect();
        x[..width].fill(0);
        let z: Vec<_> = (0..count)
            .map(|i| {
                round_bf16(match i % 9 {
                    0 => 0.0,
                    1 => -90.0,
                    2 => 90.0,
                    3 => -1e20,
                    4 => 1e20,
                    _ => ((i * 11 % 47) as f32 - 23.0) / 4.0,
                })
            })
            .collect();
        let weight: Vec<_> = (0..width)
            .map(|i| round_bf16(((i * 3 % 11) as f32 - 5.0) / 4.0))
            .collect();
        let xd = upload(context, &word_bytes(&x))?;
        let zd = upload(context, &word_bytes(&z))?;
        let result = normalize(
            context,
            module,
            Input {
                x: &xd,
                x_words: &x,
                z: &zd,
                z_words: &z,
            },
            &weight,
            groups,
            width,
            1e-6,
        )?;
        let mut report = result.report;
        report["groups"] = json!(groups);
        report["width"] = json!(width);
        reports.push(report);
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let b = Buffer::new(context, bytes.len())?;
    b.upload(bytes)?;
    Ok(b)
}
fn word_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn decode(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(decode(&bytes))
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_rejects_wrong_intermediate_rounding() {
        let input = [round_bf16(0.75)];
        let gate = [round_bf16(1.0)];
        let weight = [round_bf16(-0.5)];
        let expected = reference::run(&input, &gate, &weight, 1, 1, 1e-6).unwrap();
        assert!(compare(&expected, &expected, &weight).is_ok());
        let mut broken = reference::run(&input, &gate, &weight, 1, 1, 1e-6).unwrap();
        broken.weighted[0] ^= 1;
        assert!(compare(&broken, &expected, &weight).is_err());
    }
}
