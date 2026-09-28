//! Bounded same-input qualification of the experimental NVFP4 pipeline.
use super::driver::{Buffer, Context, Module};
use crate::{
    entry_reference::round_bf16, nvfp4_linear_reference::Matrix, nvfp4_prefill_tiled_reference,
    projection_reference::LinearReference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

// Frozen before GPU execution. Native baseline equivalence is always bitwise.
const BF16_L2_LIMIT: f64 = 0.01;
const RAW_SCALED_LIMIT: f64 = 1e-4;

struct Fixture {
    name: &'static str,
    shape: [usize; 3],
    a: Vec<u8>,
    w: Vec<u8>,
    sa: Vec<u8>,
    sw: Vec<u8>,
    exact: bool,
}

struct Output {
    bf16: Vec<u16>,
    raw: Vec<f32>,
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "NVFP4 pipeline trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let candidate_resources = module.function("nvfp4_prefill_tiled")?.resources()?;
    let wide_resources = module.function("nvfp4_prefill_wide")?.resources()?;
    let baseline_resources = module.function("nvfp4_linear")?.resources()?;
    let fixtures = fixtures();
    let products: usize = fixtures
        .iter()
        .map(|f| f.shape.iter().product::<usize>())
        .sum();
    ensure!(
        products <= 25_000_000,
        "NVFP4 trial exceeds CPU product budget"
    );
    let mut cases = Vec::with_capacity(fixtures.len());
    for fixture in fixtures {
        cases.push(execute(&ctx, &module, fixture)?);
    }
    Ok(json!({
        "kind":"nvfp4-pipeline-synthetic-trial", "device":info,
        "candidate_resources":candidate_resources,"wide_resources":wide_resources,
        "baseline_resources":baseline_resources,
        "jit_log":module.jit_log(),"cpu_oracle_products":products,
        "all_passed":cases.iter().all(|case| case["all_passed"] == true),
        "cases":cases,"bf16_l2_limit":BF16_L2_LIMIT,"raw_scaled_limit":RAW_SCALED_LIMIT,
        "global_factor":1.0,"baseline_requires_bit_equality":true,
        "scope":"synthetic arithmetic and completion only; no timing or model claims"
    }))
}

fn fixtures() -> Vec<Fixture> {
    let mut result = vec![fixture([16, 16, 192], true, true)];
    for shape in [
        [1, 8, 64],
        [17, 24, 128],
        [31, 32, 192],
        [32, 40, 64],
        [33, 8, 192],
        [128, 24, 128],
        [512, 40, 192],
    ] {
        result.push(fixture(shape, true, false));
        result.push(fixture(shape, false, false));
    }
    result.push(fixture([16, 136, 192], true, true));
    for shape in [
        [1, 120, 64],
        [17, 128, 128],
        [31, 136, 192],
        [32, 120, 192],
        [33, 136, 128],
    ] {
        result.push(fixture(shape, true, false));
        result.push(fixture(shape, false, false));
    }
    for shape in [
        [1, 8, 5120],
        [17, 8, 5120],
        [1, 24, 17408],
        [5, 8, 17408],
        [512, 128, 64],
        [1, 136, 5120],
        [1, 120, 17408],
    ] {
        result.push(fixture(shape, false, false));
    }
    result
}

fn pack(rows: usize, k: usize, code: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    (0..rows * k / 2)
        .map(|index| {
            let row = index / (k / 2);
            let kk = (index % (k / 2)) * 2;
            code(row, kk) | (code(row, kk + 1) << 4)
        })
        .collect()
}

fn fixture(shape: [usize; 3], exact: bool, one_hot: bool) -> Fixture {
    let [m, n, k] = shape;
    let a = pack(m, k, |row, kk| {
        if one_hot {
            if kk == (row * 11 + 7) % k { 2 } else { 0 }
        } else {
            ((kk * 3 + row * 7 + kk / 16) % 16) as u8
        }
    });
    // For each one-hot token, the 16 columns expose all 16 signed E2M1 codes.
    let w = pack(n, k, |column, kk| {
        ((kk * 5 + column * 7 + kk / 16) % 16) as u8
    });
    let scales = |rows: usize, salt: usize| {
        (0..rows * (k / 16))
            .map(|index| {
                let row = index / (k / 16);
                let group = index % (k / 16);
                if exact {
                    [0x30, 0x38, 0x40][(row * salt + group) % 3]
                } else {
                    ((row * salt + group * 19 + group / 3) % 127) as u8
                }
            })
            .collect()
    };
    Fixture {
        name: if one_hot {
            "all-e2m1-one-hot"
        } else if exact {
            "asymmetric-exact-dyadic"
        } else {
            "nonuniform-scales-signed-cancellation"
        },
        shape,
        a,
        w,
        sa: scales(m, 7),
        sw: scales(n, 11),
        exact,
    }
}

fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

fn execute(ctx: &Context, module: &Module<'_>, fixture: Fixture) -> Result<Value> {
    let [m, n, k] = fixture.shape;
    let reference = nvfp4_prefill_tiled_reference::run(
        Matrix {
            packed: &fixture.a,
            scales: &fixture.sa,
            rows: m,
            global: 1.0,
        },
        Matrix {
            packed: &fixture.w,
            scales: &fixture.sw,
            rows: n,
            global: 1.0,
        },
        k,
    )?;
    let inputs = [
        upload(ctx, &fixture.a)?,
        upload(ctx, &fixture.w)?,
        upload(ctx, &fixture.sa)?,
        upload(ctx, &fixture.sw)?,
    ];
    let candidate = launch(ctx, module, &inputs, fixture.shape, "nvfp4_prefill_tiled")?;
    let wide = launch(ctx, module, &inputs, fixture.shape, "nvfp4_prefill_wide")?;
    let baseline = launch(ctx, module, &inputs, fixture.shape, "nvfp4_linear")?;
    let wide_metrics = compare(&reference, &wide, fixture.exact);
    let wide_raw_differences = wide
        .raw
        .iter()
        .zip(&baseline.raw)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let wide_bf16_differences = wide
        .bf16
        .iter()
        .zip(&baseline.bf16)
        .filter(|(a, b)| a != b)
        .count();
    let candidate_metrics = compare(&reference, &candidate, fixture.exact);
    let baseline_metrics = compare(&reference, &baseline, fixture.exact);
    let raw_differences = candidate
        .raw
        .iter()
        .zip(&baseline.raw)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let bf16_differences = candidate
        .bf16
        .iter()
        .zip(&baseline.bf16)
        .filter(|(a, b)| a != b)
        .count();
    let maximum = candidate
        .raw
        .iter()
        .zip(&baseline.raw)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max);
    Ok(
        json!({"name":fixture.name,"shape":fixture.shape,"requires_exact":fixture.exact,
        "all_passed":candidate_metrics["all_passed"] == true && baseline_metrics["all_passed"] == true
            && raw_differences == 0 && bf16_differences == 0
            && wide_metrics["all_passed"] == true
            && wide_raw_differences == 0 && wide_bf16_differences == 0,
        "completed":true,"outputs_poisoned_before_launch":true,"comparison_count":m*n,
        "candidate":candidate_metrics,"baseline":baseline_metrics,"wide":wide_metrics,
        "wide_native_raw_bit_differences":wide_raw_differences,
        "wide_native_bf16_bit_differences":wide_bf16_differences,
        "native_baseline_raw_bit_differences":raw_differences,
        "native_baseline_bf16_bit_differences":bf16_differences,
        "native_baseline_raw_max_abs_error":(candidate_metrics["finite"] == true && baseline_metrics["finite"] == true).then_some(maximum)}),
    )
}

fn launch(
    ctx: &Context,
    module: &Module<'_>,
    inputs: &[Buffer<'_>; 4],
    shape: [usize; 3],
    symbol: &str,
) -> Result<Output> {
    let [m, n, k] = shape;
    let poison: Vec<u8> = (0..m * n).flat_map(|_| 0x7fc1_u16.to_le_bytes()).collect();
    let raw_poison: Vec<u8> = (0..m * n)
        .flat_map(|_| 0x7fc1_2345_u32.to_le_bytes())
        .collect();
    let out = upload(ctx, &poison)?;
    let raw = upload(ctx, &raw_poison)?;
    let mut pointers = [
        inputs[0].pointer(),
        inputs[1].pointer(),
        inputs[2].pointer(),
        inputs[3].pointer(),
        out.pointer(),
        raw.pointer(),
    ];
    let mut dimensions = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut factor = 1.0_f32;
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(
        dimensions
            .iter_mut()
            .map(|v| (v as *mut u32).cast::<c_void>()),
    );
    args.push((&raw mut factor).cast::<c_void>());
    let (tile_n, tile_m, threads) = match symbol {
        "nvfp4_prefill_tiled" => (32, 32, 256),
        "nvfp4_prefill_wide" => (128, 32, 256),
        "nvfp4_linear" => (8, 16, 32),
        _ => anyhow::bail!("unsupported NVFP4 trial symbol"),
    };
    let function = module.function(symbol)?;
    let grid = [
        u32::try_from(n.div_ceil(tile_n))?,
        u32::try_from(m.div_ceil(tile_m))?,
        1,
    ];
    // SAFETY: Independent reference validates shape, extents and scale codes.
    // CUDA allocations are aligned and disjoint. Inputs and poisoned outputs
    // stay live until synchronization; launch dimensions match each symbol's ABI.
    let launched = unsafe { function.launch(grid, [threads, 1, 1], 0, &mut args) };
    if let Err(error) = launched {
        let drain = ctx.synchronize();
        return Err(error.context(format!("{symbol} launch failed; drain: {drain:?}")));
    }
    ctx.synchronize()?;
    let mut output_bytes = vec![0; out.len()];
    let mut raw_bytes = vec![0; raw.len()];
    out.download(&mut output_bytes)?;
    raw.download(&mut raw_bytes)?;
    Ok(Output {
        bf16: output_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect(),
        raw: raw_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
    })
}

fn compare(reference: &LinearReference, output: &Output, exact: bool) -> Value {
    let bf = |bits: u16| f64::from(f32::from_bits(u32::from(bits) << 16));
    let finite =
        output.bf16.iter().all(|&v| bf(v).is_finite()) && output.raw.iter().all(|v| v.is_finite());
    let bf16_differences = output
        .bf16
        .iter()
        .zip(&reference.normalized)
        .filter(|(a, b)| a != b)
        .count();
    let raw_differences = output
        .raw
        .iter()
        .zip(&reference.unrounded)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let rne_differences = output
        .bf16
        .iter()
        .zip(&output.raw)
        .filter(|(a, b)| **a != round_bf16(**b))
        .count();
    let error = output
        .bf16
        .iter()
        .zip(&reference.normalized)
        .map(|(&a, &b)| (bf(a) - bf(b)).powi(2))
        .sum::<f64>();
    let norm = reference
        .normalized
        .iter()
        .map(|&v| bf(v).powi(2))
        .sum::<f64>();
    let l2 = (error / norm.max(1e-30)).sqrt();
    let maximum = output
        .raw
        .iter()
        .zip(&reference.unrounded)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max);
    let oracle_maximum = reference
        .unrounded
        .iter()
        .map(|&v| f64::from(v).abs())
        .fold(1.0, f64::max);
    let numeric = if exact {
        bf16_differences == 0 && raw_differences == 0
    } else {
        l2 <= BF16_L2_LIMIT && maximum / oracle_maximum <= RAW_SCALED_LIMIT
    };
    json!({"all_passed":finite && rne_differences == 0 && numeric,"finite":finite,
        "comparison_count":output.raw.len(),"bf16_bit_differences":bf16_differences,
        "raw_bit_differences":raw_differences,"stored_bf16_rne_differences":rne_differences,
        "bf16_normalized_l2":finite.then_some(l2),"raw_max_abs_error":finite.then_some(maximum),
        "raw_max_scaled_error":finite.then_some(maximum/oracle_maximum)})
}
