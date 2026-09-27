//! BF16 MLP activation boundary checked against independent scalar SiLU.
use super::driver::{Buffer, Context, Module};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    mlp_activation_reference as reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Activated<'a> {
    pub(super) output: Buffer<'a>,
    pub(super) words: Vec<u16>,
    pub(super) report: Value,
}
pub(super) struct Input<'a, 'ctx> {
    pub(super) gate: &'a Buffer<'ctx>,
    pub(super) gate_words: &'a [u16],
    pub(super) up: &'a Buffer<'ctx>,
    pub(super) up_words: &'a [u16],
}
pub(super) fn check<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: Input<'_, '_>,
) -> Result<Activated<'a>> {
    let expected = reference::run(input.gate_words, input.up_words)?;
    let count = input.gate_words.len();
    let output = upload(context, &vec![0xa5; count * 2])?;
    let silu = upload(context, &vec![0xff; count * 4])?;
    let activated = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let mut pointers = [
        input.gate.pointer(),
        input.up.pointer(),
        output.pointer(),
        silu.pointer(),
        activated.pointer(),
        unrounded.pointer(),
    ];
    let mut count_arg = u32::try_from(count)?;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.push((&mut count_arg as *mut u32).cast());
    // SAFETY: All six exact-sized allocations are disjoint and live through sync.
    // Scalar reference validates finite BF16 inputs/results and the element bound.
    // Kernel guards padded elements; no lane collective crosses its tail branch.
    unsafe {
        module.function("mlp_silu_product")?.launch(
            [count_arg.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let actual = reference::Activation {
        output: words(&output, count)?,
        silu: floats(&silu, count)?,
        activated: words(&activated, count)?,
        unrounded: floats(&unrounded, count)?,
    };
    let report = compare(&actual, &expected, input.up_words)?;
    Ok(Activated {
        output,
        words: actual.output,
        report,
    })
}
fn compare(
    actual: &reference::Activation,
    expected: &reference::Activation,
    up: &[u16],
) -> Result<Value> {
    let count = up.len();
    ensure!(
        [
            actual.output.len(),
            actual.silu.len(),
            actual.activated.len(),
            actual.unrounded.len(),
            expected.output.len(),
            expected.silu.len()
        ]
        .iter()
        .all(|&n| n == count),
        "MLP activation comparison extent mismatch"
    );
    let mut silu_error = 0.0_f64;
    let mut output_error = 0.0_f64;
    let mut bf16_differences = 0;
    let mut activated_differences = 0;
    for (i, &up_bits) in up.iter().enumerate() {
        let s = actual.silu[i];
        let p = actual.unrounded[i];
        ensure!(
            [
                s,
                p,
                bf16_to_f32(actual.output[i]),
                bf16_to_f32(actual.activated[i])
            ]
            .iter()
            .all(|v| v.is_finite()),
            "MLP activation nonfinite output"
        );
        let error = (f64::from(s) - f64::from(expected.silu[i])).abs();
        ensure!(
            error <= 3e-6 + 5e-6 * f64::from(expected.silu[i]).abs(),
            "MLP SiLU mismatch at {i}"
        );
        ensure!(
            actual.activated[i] == round_bf16(s),
            "MLP SiLU BF16 rounding mismatch at {i}"
        );
        let product = bf16_to_f32(actual.activated[i]) * bf16_to_f32(up_bits);
        ensure!(
            p.to_bits() == product.to_bits(),
            "MLP multiply mismatch at {i}"
        );
        ensure!(
            actual.output[i] == round_bf16(p),
            "MLP product BF16 rounding mismatch at {i}"
        );
        silu_error = silu_error.max(error);
        output_error = output_error.max((f64::from(p) - f64::from(expected.unrounded[i])).abs());
        activated_differences += usize::from(actual.activated[i] != expected.activated[i]);
        bf16_differences += usize::from(actual.output[i] != expected.output[i]);
    }
    Ok(
        json!({"all_passed":true,"elements":count,"silu_max_abs_error":silu_error,"full_reference_max_abs_error":output_error,
        "activated_bf16_reference_differences":activated_differences,"output_bf16_reference_differences":bf16_differences,
        "intermediate_and_final_rounding_exact":true,"tolerance":"3e-6 + 5e-6 * abs(reference SiLU)",
        "profile":"BF16 gate -> FP32 SiLU -> BF16 -> multiply BF16 up in FP32 -> BF16"}),
    )
}
pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for count in [1, 257, 513] {
        let gate: Vec<_> = (0..count)
            .map(|i| round_bf16([0.0, -0.0, -90.0, 90.0, -1e20, 1e20, -1.25, 0.5][i % 8]))
            .collect();
        let up: Vec<_> = (0..count)
            .map(|i| round_bf16(((i * 7 % 13) as f32 - 6.0) / 4.0))
            .collect();
        let gd = upload(context, &word_bytes(&gate))?;
        let ud = upload(context, &word_bytes(&up))?;
        reports.push(
            check(
                context,
                module,
                Input {
                    gate: &gd,
                    gate_words: &gate,
                    up: &ud,
                    up_words: &up,
                },
            )?
            .report,
        );
    }
    Ok(reports)
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let b = Buffer::new(context, bytes.len())?;
    b.upload(bytes)?;
    Ok(b)
}
fn word_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn words(b: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut v = vec![0; count * 2];
    b.download(&mut v)?;
    Ok(v.as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect())
}
fn floats(b: &Buffer<'_>, count: usize) -> Result<Vec<f32>> {
    let mut v = vec![0; count * 4];
    b.download(&mut v)?;
    Ok(v.as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_rejects_wrong_activation_boundary() {
        let gate = [round_bf16(1.0)];
        let up = [round_bf16(1.75)];
        let expected = reference::run(&gate, &up).unwrap();
        let mut actual = reference::run(&gate, &up).unwrap();
        assert!(compare(&actual, &expected, &up).is_ok());
        actual.activated[0] ^= 1;
        assert!(compare(&actual, &expected, &up).is_err());
    }
}
