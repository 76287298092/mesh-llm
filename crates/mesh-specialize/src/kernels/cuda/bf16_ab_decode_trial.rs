//! Bounded synthetic operator qualification, not resident/model admission.
use super::driver::{Buffer, Context, Event, Function, Module};
use crate::{
    bf16_ab_decode_reference as reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

struct Fixture {
    name: &'static str,
    n: usize,
    k: usize,
    x: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    exact: bool,
}

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "A/B trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let candidate = module.function("bf16_ab_decode_fp32")?;
    let baseline = module.function("bf16_linear_decode")?;
    let mut cases = Vec::new();
    for (n, k, exact) in [
        (1, 8, true),
        (3, 24, true),
        (48, 1024, true),
        (48, 1032, true),
        (48, 5120, true),
        (48, 5120, false),
        (48, 5120, false),
        (48, 32768, false),
    ] {
        let seed = cases.len() as u32 + 1;
        cases.push(execute(
            &ctx,
            &candidate,
            &baseline,
            fixture(n, k, exact, seed),
        )?);
    }
    Ok(
        json!({"kind":"bf16-ab-decode-fp32-synthetic-trial","schema_version":1,
        "device":info,"jit_log":module.jit_log(),"candidate_resources":candidate.resources()?,
        "baseline_resources":baseline.resources()?,"cases":cases,
        "all_passed":cases.iter().all(|v| v["all_passed"] == true),
        "scope":"synthetic M1 A/B arithmetic, completion, determinism and warm-cache timing only; no model or bandwidth claim"}),
    )
}

fn fixture(n: usize, k: usize, exact: bool, mut seed: u32) -> Fixture {
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let x = (0..k)
        .map(|i| {
            if exact {
                round_bf16((i as i32 % 7 - 3) as f32 / 8.0)
            } else {
                round_bf16((next() as i32 % 1009) as f32 / 1024.0)
            }
        })
        .collect();
    let mut matrix = |salt: usize| {
        (0..n * k)
            .map(|i| {
                if exact {
                    round_bf16(((i * salt + i / k) as i32 % 11 - 5) as f32 / 16.0)
                } else {
                    round_bf16((next() as i32 % 1999) as f32 / 2048.0)
                }
            })
            .collect()
    };
    Fixture {
        name: if exact {
            "exact-small-dyadics"
        } else {
            "seeded-signed-cancellation"
        },
        n,
        k,
        x,
        a: matrix(3),
        b: matrix(7),
        exact,
    }
}

fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(ctx, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn words_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}

struct Launch {
    pointers: Vec<u64>,
    dimensions: Vec<u32>,
    grid: [u32; 3],
}
impl Launch {
    fn run(&mut self, function: &Function<'_, '_>) -> Result<()> {
        let mut args: Vec<*mut c_void> = self
            .pointers
            .iter_mut()
            .map(|v| (v as *mut u64).cast())
            .collect();
        args.extend(self.dimensions.iter_mut().map(|v| (v as *mut u32).cast()));
        // SAFETY: execute constructs the exact pointer/scalar order for each symbol,
        // validates inputs through the independent oracle, and retains every buffer
        // until synchronization. Both symbols require 128 threads, no dynamic shared.
        unsafe { function.launch(self.grid, [128, 1, 1], 0, &mut args) }
    }
}

fn execute(
    ctx: &Context,
    candidate: &Function<'_, '_>,
    baseline: &Function<'_, '_>,
    f: Fixture,
) -> Result<Value> {
    let expected = [
        reference::run(&f.x, &f.a, f.n, f.k)?,
        reference::run(&f.x, &f.b, f.n, f.k)?,
    ];
    let inputs = [
        upload(ctx, &words_bytes(&f.x))?,
        upload(ctx, &words_bytes(&f.a))?,
        upload(ctx, &words_bytes(&f.b))?,
    ];
    let poison = words_bytes(&vec![0x7fc1; f.n]);
    let raw_poison: Vec<_> = (0..f.n)
        .flat_map(|_| 0x7fc12345_u32.to_le_bytes())
        .collect();
    let outputs = [
        upload(ctx, &poison)?,
        upload(ctx, &poison)?,
        upload(ctx, &raw_poison)?,
        upload(ctx, &raw_poison)?,
    ];
    let controls = [
        upload(ctx, &poison)?,
        upload(ctx, &poison)?,
        upload(ctx, &raw_poison)?,
        upload(ctx, &raw_poison)?,
    ];
    let mut fast = Launch {
        pointers: inputs
            .iter()
            .chain(outputs.iter())
            .map(Buffer::pointer)
            .collect(),
        dimensions: vec![f.n as u32, f.k as u32],
        grid: [f.n as u32, 2, 1],
    };
    let mut control: Vec<_> = (0..2)
        .map(|p| Launch {
            pointers: vec![
                inputs[0].pointer(),
                inputs[p + 1].pointer(),
                controls[p].pointer(),
                controls[p + 2].pointer(),
            ],
            dimensions: vec![1, f.n as u32, f.k as u32],
            grid: [f.n.div_ceil(4) as u32, 1, 1],
        })
        .collect();
    let launched = (|| -> Result<()> {
        fast.run(candidate)?;
        for launch in &mut control {
            launch.run(baseline)?;
        }
        Ok(())
    })();
    let drain = ctx.synchronize();
    launched?;
    drain?;
    let first = outputs.iter().map(download).collect::<Result<Vec<_>>>()?;
    let mut metrics = Vec::new();
    let mut control_metrics = Vec::new();
    for p in 0..2 {
        metrics.push(compare(
            &first[p],
            &first[p + 2],
            &expected[p],
            f.k,
            f.exact,
        ));
        control_metrics.push(compare(
            &download(&controls[p])?,
            &download(&controls[p + 2])?,
            &expected[p],
            f.k,
            true,
        ));
    }
    // All repeated writes use the same immutable inputs and disjoint outputs.
    let timing = (|| -> Result<(Value, Value)> {
        let fast_time = time(ctx, || fast.run(candidate), 4 * f.n * f.k)?;
        let control_time = time(
            ctx,
            || {
                for launch in &mut control {
                    launch.run(baseline)?;
                }
                Ok(())
            },
            4 * f.n * f.k,
        )?;
        Ok((fast_time, control_time))
    })();
    let drain = ctx.synchronize();
    let (fast_time, control_time) = timing?;
    drain?;
    let repeated = outputs.iter().map(download).collect::<Result<Vec<_>>>()?;
    let deterministic = first == repeated;
    Ok(
        json!({"name":f.name,"shape":[1,f.n,f.k],"two_distinct_matrices":true,
        "exact_fixture":f.exact,"candidate":metrics,"baseline":control_metrics,
        "all_passed":deterministic && metrics.iter().chain(&control_metrics).all(|m| m["all_passed"] == true),
        "deterministic":deterministic,"completed":true,"poisoned_before_launch":true,
        "candidate_timing":fast_time,"baseline_two_launch_timing":control_time}),
    )
}

fn compare(
    words: &[u8],
    raw: &[u8],
    expected: &reference::Projection,
    k: usize,
    exact: bool,
) -> Value {
    let mut finite = true;
    let mut mismatches = 0;
    let mut rne_mismatches = 0;
    let mut raw_max = 0.0_f64;
    let mut bf_max = 0.0_f64;
    let mut max_budget = 0.0_f64;
    for (i, (word, raw)) in words
        .as_chunks::<2>()
        .0
        .iter()
        .zip(raw.as_chunks::<4>().0)
        .enumerate()
    {
        let word = u16::from_le_bytes([word[0], word[1]]);
        let value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let bf = bf16_to_f32(word);
        finite &= value.is_finite() && bf.is_finite();
        let raw_error = (f64::from(value) - f64::from(expected.raw[i])).abs();
        let bf_error = (f64::from(bf) - f64::from(bf16_to_f32(expected.bf16[i]))).abs();
        let budget = reference::raw_budget(k, expected.absolute_products[i]);
        // Sum of the two endpoint half-ULPs safely handles binade boundaries.
        let rounding =
            0.5 * (reference::bf16_spacing(value) + reference::bf16_spacing(expected.raw[i]));
        let passed = if exact {
            value.to_bits() == expected.raw[i].to_bits() && word == expected.bf16[i]
        } else {
            raw_error <= budget && bf_error <= budget + rounding
        };
        mismatches += usize::from(!passed);
        rne_mismatches += usize::from(word != round_bf16(value));
        raw_max = raw_max.max(raw_error);
        bf_max = bf_max.max(bf_error);
        max_budget = max_budget.max(budget);
    }
    json!({"all_passed":finite && mismatches == 0 && rne_mismatches == 0,
        "finite":finite,"mismatches":mismatches,"stored_rne_mismatches":rne_mismatches,
        "raw_max_abs":raw_max,"bf16_max_abs":bf_max,"largest_raw_budget":max_budget,
        "budget":"gamma(2*ceil(K/1024)+14)*sum_abs_products + 1e-37; BF16 adds endpoint half-ULPs",
        "elements":expected.raw.len(),"requires_exact":exact})
}

fn time(
    ctx: &Context,
    mut launch: impl FnMut() -> Result<()>,
    weight_bytes: usize,
) -> Result<Value> {
    for _ in 0..3 {
        launch()?;
    }
    ctx.synchronize()?;
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut batches = Vec::new();
    for _ in 0..3 {
        start.record()?;
        for _ in 0..10 {
            launch()?;
        }
        end.record()?;
        end.synchronize()?;
        batches.push(end.elapsed_since(&start)?);
    }
    let micros: Vec<_> = batches.iter().map(|&ms| f64::from(ms) * 100.0).collect();
    let gbps: Vec<_> = micros
        .iter()
        .map(|&us| weight_bytes as f64 / (us * 1000.0))
        .collect();
    Ok(
        json!({"warmup_pairs":3,"batches":3,"pairs_per_batch":10,"batch_event_ms":batches,
        "pair_event_us":micros,"logical_weight_bytes_per_pair":weight_bytes,"logical_weight_gb_s":gbps,
        "note":"warm-cache event batches include host submission gaps; not DRAM bandwidth; fixed candidate-first order"}),
    )
}
