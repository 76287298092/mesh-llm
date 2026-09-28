//! Small exact-integer NVFP4 decode fixtures against the independent logical oracle.
use super::driver::{Buffer, Context, Module};
use crate::nvfp4_linear_reference::{self as reference, Matrix};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

#[derive(Default)]
struct CaseCounts {
    fixtures: usize,
    outputs: usize,
    fp32_differences: usize,
    bf16_differences: usize,
    nonfinite_fp32: usize,
}

impl CaseCounts {
    fn report(&self, name: &str) -> Value {
        json!({
            "fixture": name,
            "fixtures": self.fixtures,
            "outputs": self.outputs,
            "fp32_differences": self.fp32_differences,
            "bf16_differences": self.bf16_differences,
            "nonfinite_fp32": self.nonfinite_fp32,
            "all_passed": self.fp32_differences == 0
                && self.bf16_differences == 0
                && self.nonfinite_fp32 == 0,
        })
    }
}

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    if super::nvfp4_decode_prmt_trial::enabled()? {
        return super::nvfp4_decode_prmt_trial::run(ctx, module);
    }
    let mut code_counts = CaseCounts::default();
    let weights = (0..16_u8)
        .flat_map(|code| [code | (code << 4); 8])
        .collect::<Vec<_>>();
    for code in 0..16_u8 {
        check_fixture(
            ctx,
            module,
            [16, 16],
            [&[code | (code << 4); 8], &weights],
            [&[0x38], &[0x38; 16]],
            &mut code_counts,
        )?;
    }

    let mut scale_counts = CaseCounts::default();
    let scale_n = 127;
    let scale_k = 16;
    let scale_activation = vec![0x11; scale_k / 2];
    let scale_weights = vec![0x33; scale_n * scale_k / 2];
    let weight_scales = (0_u8..=126).collect::<Vec<_>>();
    for activation_scale in 0_u8..=126 {
        check_fixture(
            ctx,
            module,
            [scale_n, scale_k],
            [&scale_activation, &scale_weights],
            [&[activation_scale], &weight_scales],
            &mut scale_counts,
        )?;
    }

    let mut wide_counts = CaseCounts::default();
    let wide_n = 3;
    let wide_k = 32768;
    let groups = wide_k / 16;
    let wide_activation = vec![0x77; wide_k / 2];
    let mut wide_weights = Vec::with_capacity(wide_n * wide_k / 2);
    wide_weights.extend(vec![0x77; wide_k / 2]);
    wide_weights.extend(vec![0xff; wide_k / 2]);
    wide_weights.extend(vec![0x77; wide_k / 4]);
    wide_weights.extend(vec![0xff; wide_k / 4]);
    let wide_activation_scales = vec![126; groups];
    let wide_weight_scales = vec![126; wide_n * groups];
    check_fixture(
        ctx,
        module,
        [wide_n, wide_k],
        [&wide_activation, &wide_weights],
        [&wide_activation_scales, &wide_weight_scales],
        &mut wide_counts,
    )?;

    let cases = vec![
        code_counts.report("all-e2m1-codes"),
        scale_counts.report("all-ue4m3-scale-pairs"),
        wide_counts.report("max-width-max-scale-and-n-tail"),
    ];
    let all_passed = cases.iter().all(|case| case["all_passed"] == true);
    Ok(json!({
        "all_passed": all_passed,
        "cases": cases,
        "resources": module.function("nvfp4_decode_exact")?.resources()?,
    }))
}

fn check_fixture(
    ctx: &Context,
    module: &Module<'_>,
    shape: [usize; 2],
    packed: [&[u8]; 2],
    scales: [&[u8]; 2],
    counts: &mut CaseCounts,
) -> Result<()> {
    let [n, k] = shape;
    let [activation, weights] = packed;
    let [activation_scales, weight_scales] = scales;
    let expected = reference::run(
        Matrix {
            packed: activation,
            scales: activation_scales,
            rows: 1,
            global: 1.0,
        },
        Matrix {
            packed: weights,
            scales: weight_scales,
            rows: n,
            global: 1.0,
        },
        k,
    )?;
    let (actual_bf16, actual_fp32) = execute(ctx, module, shape, packed, scales)?;
    ensure!(
        actual_bf16.len() == expected.normalized.len()
            && actual_fp32.len() == expected.unrounded.len(),
        "NVFP4 exact output extent mismatch"
    );

    counts.fixtures += 1;
    counts.outputs += expected.normalized.len();
    counts.bf16_differences += actual_bf16
        .iter()
        .zip(&expected.normalized)
        .filter(|(actual, wanted)| actual != wanted)
        .count();
    counts.fp32_differences += actual_fp32
        .iter()
        .zip(&expected.unrounded)
        .filter(|(actual, wanted)| actual.to_bits() != wanted.to_bits())
        .count();
    counts.nonfinite_fp32 += actual_fp32
        .iter()
        .filter(|value| !value.is_finite())
        .count();
    Ok(())
}

fn execute(
    ctx: &Context,
    module: &Module<'_>,
    shape: [usize; 2],
    packed: [&[u8]; 2],
    scales: [&[u8]; 2],
) -> Result<(Vec<u16>, Vec<f32>)> {
    let [n, k] = shape;
    let [activation, weights] = packed;
    let [activation_scales, weight_scales] = scales;
    ensure!(
        activation.len() == k / 2
            && weights.len() == n * k / 2
            && activation_scales.len() == k / 16
            && weight_scales.len() == n * k / 16,
        "NVFP4 exact fixture extents"
    );
    ensure!(
        activation_scales.iter().all(|&code| code <= 126)
            && weight_scales.iter().all(|&code| code <= 126),
        "NVFP4 exact fixture scale domain"
    );

    let activation = upload(ctx, activation)?;
    let weights = upload(ctx, weights)?;
    let activation_scales = upload(ctx, activation_scales)?;
    let weight_scales = upload(ctx, weight_scales)?;
    let output = upload(ctx, &vec![0xa5; n * 2])?;
    let unrounded = upload(ctx, &vec![0xff; n * 4])?;
    let mut pointers = [
        activation.pointer(),
        weights.pointer(),
        activation_scales.pointer(),
        weight_scales.pointer(),
        output.pointer(),
        unrounded.pointer(),
    ];
    let mut dimensions = [1_u32, u32::try_from(n)?, u32::try_from(k)?];
    let mut factor = 1.0_f32;
    let mut args = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect::<Vec<*mut c_void>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    args.push((&mut factor as *mut f32).cast());

    // SAFETY: The fixtures have checked packed/scales extents and separate device
    // buffers. The exact kernel gets its required m=1, grid, and 128-thread blocks.
    unsafe {
        module.function("nvfp4_decode_exact")?.launch(
            [u32::try_from(n.div_ceil(4))?, 1, 1],
            [128, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;

    let mut bf16_bytes = vec![0; n * 2];
    output.download(&mut bf16_bytes)?;
    let mut fp32_bytes = vec![0; n * 4];
    unrounded.download(&mut fp32_bytes)?;
    Ok((
        bf16_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect(),
        fp32_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect(),
    ))
}

fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
