//! Synthetic qualification for the separately named A16 sliced-K head.
use super::driver::{Buffer, Context, Module};
use crate::{entry_reference::round_bf16, fp8_a16_head_reference};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

// Frozen before GPU execution, matching the existing F01 limits.
const BF16_L2_LIMIT: f64 = 0.01;
const FP32_SCALED_LIMIT: f64 = 1e-4;

struct Fixture {
    name: &'static str,
    shape: [usize; 3],
    input: Vec<u16>,
    weights: Vec<u8>,
    scales: Vec<u16>,
    exact: bool,
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "A16 head trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let resources = module.function("fp8_a16_head")?.resources()?;
    let mut cases = vec![execute(&ctx, &module, exhaustive())?];
    for m in [1, 5, 8] {
        for n in [8, 24] {
            for k in [16, 240, 256, 272] {
                cases.push(execute(&ctx, &module, fixture([m, n, k], true))?);
            }
            for k in [272, 5120] {
                cases.push(execute(&ctx, &module, fixture([m, n, k], false))?);
            }
        }
    }
    Ok(json!({
        "kind":"a16-head-synthetic-trial", "device":info, "resources":resources,
        "jit_log":module.jit_log(),
        "all_passed":cases.iter().all(|case| case["all_passed"] == true),
        "cases":cases, "bf16_l2_limit":BF16_L2_LIMIT,
        "fp32_scaled_limit":FP32_SCALED_LIMIT,
        "scope":"synthetic arithmetic and completion only; no timing or model claims"
    }))
}

fn finite_code(index: usize) -> u8 {
    let code = (index % 254) as u8;
    if code >= 127 { code + 1 } else { code }
}

fn exhaustive() -> Fixture {
    let [m, n, k] = [8, 256, 16];
    let mut input = vec![0; m * k];
    for row in 0..m {
        input[row * k + row * 2] = 0x3f80;
    }
    // Every token sees every finite code across columns. Different K positions
    // and cyclic column shifts distinguish token/column/fragment mappings.
    let weights = (0..n * k)
        .map(|index| finite_code(index / k + (index % k) * 31))
        .collect();
    Fixture {
        name: "all-254-finite-codes-one-hot",
        shape: [m, n, k],
        input,
        weights,
        scales: vec![0x3f80; n],
        exact: true,
    }
}

fn fixture(shape: [usize; 3], exact: bool) -> Fixture {
    let [m, n, k] = shape;
    let input = (0..m * k)
        .map(|index| {
            let row = index / k;
            let kk = index % k;
            let value = if exact {
                ((kk * 3 + row * 7 + kk / 16) % 17) as i32 - 8
            } else {
                ((kk * 13 + row * 11 + kk / 7) % 61) as i32 - 30
            };
            round_bf16(value as f32 / if exact { 8.0 } else { 13.0 })
        })
        .collect();
    let dyadic = [0x00, 0x28, 0x30, 0x38, 0x40, 0xa8, 0xb0, 0xb8, 0xc0];
    let weights = (0..n * k)
        .map(|index| {
            let column = index / k;
            let kk = index % k;
            if exact {
                dyadic[(kk * 5 + column * 7 + kk / 16) % dyadic.len()]
            } else {
                finite_code(kk + column * 37)
            }
        })
        .collect();
    let scales = (0..n)
        .map(|column| {
            let value = (column % 5 + 1) as f32 / 8.0;
            round_bf16(if column % 2 == 0 { value } else { -value })
        })
        .collect();
    Fixture {
        name: if exact {
            "asymmetric-exact-dyadic"
        } else {
            "all-code-signed-cancellation"
        },
        shape,
        input,
        weights,
        scales,
        exact,
    }
}

fn words(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

fn execute(ctx: &Context, module: &Module<'_>, fixture: Fixture) -> Result<Value> {
    let [m, n, k] = fixture.shape;
    let reference =
        fp8_a16_head_reference::run(&fixture.input, &fixture.weights, &fixture.scales, m, n, k)?;
    let input = upload(ctx, &words(&fixture.input))?;
    let weight = upload(ctx, &fixture.weights)?;
    let scales = upload(ctx, &words(&fixture.scales))?;
    // Quiet NaN poisons make unwritten outputs fail the finite check.
    let out = upload(ctx, &words(&vec![0x7fc1; m * n]))?;
    let raw_poison: Vec<u8> = (0..m * n)
        .flat_map(|_| 0x7fc1_2345_u32.to_le_bytes())
        .collect();
    let raw = upload(ctx, &raw_poison)?;
    let mut pointers = [
        input.pointer(),
        weight.pointer(),
        scales.pointer(),
        out.pointer(),
        raw.pointer(),
    ];
    let mut dimensions = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|v| (v as *mut u32).cast::<c_void>()),
    );
    // SAFETY: Reference validates bounded dimensions and full input extents.
    // Five disjoint allocations have the required alignment and remain live
    // through synchronization and downloads. Every launch has exactly 512 threads.
    let launch = unsafe {
        module.function("fp8_a16_head")?.launch(
            [u32::try_from(n / 8)?, 1, 1],
            [512, 1, 1],
            0,
            &mut args,
        )
    };
    if let Err(error) = launch {
        let sync = ctx.synchronize();
        return Err(error.context(format!("A16 head launch failed; drain: {sync:?}")));
    }
    ctx.synchronize()?;
    let mut out_bytes = vec![0; out.len()];
    let mut raw_bytes = vec![0; raw.len()];
    out.download(&mut out_bytes)?;
    raw.download(&mut raw_bytes)?;
    let actual = out_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let actual_raw = raw_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    Ok(compare(&fixture, &reference, &actual, &actual_raw))
}

fn compare(
    fixture: &Fixture,
    reference: &fp8_a16_head_reference::Fp8A16HeadResult,
    actual: &[u16],
    raw: &[f32],
) -> Value {
    let bf = |bits: u16| f64::from(f32::from_bits(u32::from(bits) << 16));
    let expected = &reference.output_bf16;
    let oracle = &reference.unrounded_fp32;
    let finite = actual.iter().all(|&v| bf(v).is_finite()) && raw.iter().all(|v| v.is_finite());
    let bf16_differences = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    let raw_differences = raw
        .iter()
        .zip(oracle)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let rne_differences = actual
        .iter()
        .zip(raw)
        .filter(|(a, b)| **a != round_bf16(**b))
        .count();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (bf(a) - bf(b)).powi(2))
        .sum::<f64>();
    let norm = expected.iter().map(|&b| bf(b).powi(2)).sum::<f64>();
    let l2 = (error / norm.max(1e-30)).sqrt();
    let raw_error = raw
        .iter()
        .zip(oracle)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let raw_norm = oracle.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>();
    let max_error = raw
        .iter()
        .zip(oracle)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max);
    let max_oracle = oracle
        .iter()
        .map(|&v| f64::from(v).abs())
        .fold(1.0, f64::max);
    let numeric_passed = if fixture.exact {
        bf16_differences == 0 && raw_differences == 0
    } else {
        l2 <= BF16_L2_LIMIT && max_error / max_oracle <= FP32_SCALED_LIMIT
    };
    json!({"name":fixture.name,"shape":fixture.shape,"requires_exact":fixture.exact,
        "all_passed":finite && rne_differences == 0 && numeric_passed,
        "completed":true,"outputs_poisoned_before_launch":true,"finite":finite,
        "bf16_differences":bf16_differences,"fp32_bit_differences":raw_differences,
        "stored_bf16_rne_differences":rne_differences,"bf16_normalized_l2":l2,
        "fp32_normalized_l2":(raw_error/raw_norm.max(1e-30)).sqrt(),
        "fp32_max_abs_error":max_error,"fp32_max_scaled_error":max_error/max_oracle,
        "bf16_l2_limit":BF16_L2_LIMIT,"fp32_scaled_limit":FP32_SCALED_LIMIT})
}
