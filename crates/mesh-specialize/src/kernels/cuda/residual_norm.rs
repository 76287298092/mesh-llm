//! Resident residual addition and zero-centered post-attention normalization.
use super::driver::{Buffer, Context, Module};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    kernels::ResidualNormWeights,
    residual_norm_reference as reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) residual: &'a Buffer<'ctx>,
    pub(super) residual_words: &'a [u16],
    pub(super) branch: &'a Buffer<'ctx>,
    pub(super) branch_words: &'a [u16],
}

pub(super) struct CheckedNorm<'a> {
    pub(super) normalized: Buffer<'a>,
    pub(super) words: Vec<u16>,
    pub(super) report: Value,
}

pub(super) fn validate(weights: &ResidualNormWeights, width: usize) -> Result<()> {
    ensure!((1..=32768).contains(&width), "residual norm invalid width");
    ensure!(
        weights.weight.len() == width * 2,
        "residual norm weight extent mismatch"
    );
    ensure!(
        weights.epsilon.is_finite() && weights.epsilon > 0.0,
        "residual norm invalid epsilon"
    );
    ensure!(
        decode(&weights.weight)
            .iter()
            .all(|&w| bf16_to_f32(w).is_finite()),
        "residual norm nonfinite weight"
    );
    Ok(())
}

pub(super) fn check<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: Input<'_, '_>,
    weights: &ResidualNormWeights,
    shape: [usize; 2],
) -> Result<CheckedNorm<'a>> {
    let [rows, width] = shape;
    validate(weights, width)?;
    let expected = reference::run(
        input.residual_words,
        input.branch_words,
        &decode(&weights.weight),
        rows,
        width,
        weights.epsilon,
    )?;
    let count = rows * width;
    let weight = upload(context, &weights.weight)?;
    let sum = upload(context, &vec![0xa5; count * 2])?;
    let normalized = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers = [
        input.residual.pointer(),
        input.branch.pointer(),
        weight.pointer(),
        sum.pointer(),
        normalized.pointer(),
        unrounded.pointer(),
    ];
    let mut width_arg = u32::try_from(width)?;
    let mut epsilon = weights.epsilon;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.push((&mut width_arg as *mut u32).cast());
    args.push((&mut epsilon as *mut f32).cast());
    // SAFETY: Six disjoint buffers and scalar width/epsilon match the kernel ABI.
    // The independent reference validates shapes and domains before launch. All
    // 256 threads reach each barrier; buffers remain live through synchronization.
    unsafe {
        module.function("residual_norm_bf16")?.launch(
            [u32::try_from(rows)?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let actual = reference::ResidualNorm {
        residual: words(&sum, count)?,
        normalized: words(&normalized, count)?,
        unrounded: floats(&unrounded, count)?,
    };
    let mut report = compare(&actual, &expected)?;
    report["shape"] = json!(shape);
    report["device_inputs_resident"] = json!(true);
    Ok(CheckedNorm {
        normalized,
        words: actual.normalized,
        report,
    })
}

fn compare(actual: &reference::ResidualNorm, expected: &reference::ResidualNorm) -> Result<Value> {
    let count = expected.normalized.len();
    ensure!(
        count > 0
            && [
                actual.residual.len(),
                actual.normalized.len(),
                actual.unrounded.len(),
                expected.residual.len(),
                expected.unrounded.len()
            ]
            .iter()
            .all(|&n| n == count),
        "residual norm comparison extent mismatch"
    );
    ensure!(
        actual.residual == expected.residual,
        "residual addition BF16 mismatch"
    );
    let mut max_error = 0.0_f64;
    let mut bf16_differences = 0;
    for i in 0..count {
        let value = actual.unrounded[i];
        ensure!(
            value.is_finite() && bf16_to_f32(actual.normalized[i]).is_finite(),
            "residual norm nonfinite output"
        );
        let error = (f64::from(value) - f64::from(expected.unrounded[i])).abs();
        ensure!(
            error <= 2e-6 + 2e-6 * f64::from(expected.unrounded[i]).abs(),
            "residual norm FP32 mismatch at {i}"
        );
        ensure!(
            actual.normalized[i] == round_bf16(value),
            "residual norm BF16 rounding mismatch at {i}"
        );
        max_error = max_error.max(error);
        bf16_differences += usize::from(actual.normalized[i] != expected.normalized[i]);
    }
    Ok(
        json!({"all_passed":true,"elements":count,"residual_bf16_exact":true,"normalized_max_abs_error":max_error,
        "normalized_bf16_reference_differences":bf16_differences,"final_rounding_exact":true,
        "tolerance":"2e-6 + 2e-6 * abs(reference)","profile":"BF16 rounded residual addition -> FP32 zero-centered RMSNorm -> BF16"}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for (rows, width) in [(3, 1), (3, 71), (2, 5120)] {
        let count = rows * width;
        let residual: Vec<_> = (0..count)
            .map(|i| {
                round_bf16(match i % 4 {
                    0 => 1.0,
                    1 => -0.0,
                    _ => ((i * 7 % 31) as f32 - 15.0) / 8.0,
                })
            })
            .collect();
        let branch: Vec<_> = (0..count)
            .map(|i| match i % 4 {
                0 => round_bf16(1.0 / 256.0),
                1 => 0x8000,
                2 => residual[i] ^ 0x8000,
                _ => round_bf16(0.75),
            })
            .collect();
        let weight: Vec<_> = (0..width)
            .map(|i| round_bf16(((i * 3 % 11) as f32 - 5.0) / 4.0))
            .collect();
        let rd = upload(context, &word_bytes(&residual))?;
        let bd = upload(context, &word_bytes(&branch))?;
        let result = check(
            context,
            module,
            Input {
                residual: &rd,
                residual_words: &residual,
                branch: &bd,
                branch_words: &branch,
            },
            &ResidualNormWeights {
                weight: word_bytes(&weight),
                epsilon: 1e-6,
            },
            [rows, width],
        )?;
        reports.push(result.report);
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
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
    fn comparison_rejects_residual_and_rounding_corruption() {
        let x = [round_bf16(1.0)];
        let b = [round_bf16(1.0 / 256.0)];
        let w = [0];
        let expected = reference::run(&x, &b, &w, 1, 1, 1e-6).unwrap();
        let mut actual = reference::run(&x, &b, &w, 1, 1, 1e-6).unwrap();
        assert!(compare(&actual, &expected).is_ok());
        actual.residual[0] ^= 1;
        assert!(compare(&actual, &expected).is_err());
        actual.residual[0] ^= 1;
        actual.normalized[0] ^= 1;
        assert!(compare(&actual, &expected).is_err());
    }
}
