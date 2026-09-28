//! Bounded synthetic operator qualification, not resident/model admission.
use super::driver::{Buffer, Context, Event, Function, Module};
use crate::{
    bf16_ab_decode_reference as reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD_BYTES: usize = 16;
const GUARD_VALUE: u8 = 0xa5;

#[derive(Clone)]
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
    let fp32 = module.function("bf16_ab_decode_fp32")?;
    let fp64 = module.function("bf16_ab_decode_fp64")?;
    let baseline = module.function("bf16_linear_decode")?;
    let mut fixtures = Vec::new();
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
        let seed = fixtures.len() as u32 + 1;
        fixtures.push(fixture(n, k, exact, seed));
    }
    fixtures.push(cancellation_fixture());
    fixtures.push(magnitude_fixture());
    let mut cases = Vec::new();
    for f in fixtures {
        let candidate_fp32 = execute(&ctx, &fp32, &baseline, f.clone(), false)?;
        let candidate_fp64 = execute(&ctx, &fp64, &baseline, f, true)?;
        cases.push(json!({"all_passed":candidate_fp32["all_passed"] == true
                && candidate_fp64["all_passed"] == true,
            "fp32":candidate_fp32,"fp64":candidate_fp64}));
    }
    Ok(
        json!({"kind":"bf16-ab-decode-synthetic-trial","schema_version":2,
        "device":info,"jit_log":module.jit_log(),
        "fp32_resources":fp32.resources()?,"fp64_resources":fp64.resources()?,
        "baseline_resources":baseline.resources()?,"cases":cases,
        "fp32_all_passed":cases.iter().all(|v| v["fp32"]["all_passed"] == true),
        "fp64_all_passed":cases.iter().all(|v| v["fp64"]["all_passed"] == true),
        "all_passed":cases.iter().all(|v| v["all_passed"] == true),
        "scope":"synthetic M1 A/B arithmetic, completion, output guards, determinism and warm-cache timing only; no model or bandwidth claim"}),
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

fn cancellation_fixture() -> Fixture {
    let (n, k) = (48, 5120);
    let matrix = |sign: f32| {
        (0..n * k)
            .map(|i| {
                let residue = (i / k + 1) as f32 / 65536.0;
                round_bf16(sign * [1048576.0, residue, -1048576.0, residue][i % 4])
            })
            .collect()
    };
    Fixture {
        name: "large-cancellation-small-residue",
        n,
        k,
        x: vec![round_bf16(1.0); k],
        a: matrix(1.0),
        b: matrix(-1.0),
        exact: false,
    }
}

fn magnitude_fixture() -> Fixture {
    let (n, k) = (48, 5120);
    // Values span exponents -5..5 with all eight BF16 significand bits.
    // Products lie on a 2^-24 lattice; sum_abs_products / 2^-24 < 2^53.
    // This deliberately bounded range admits exact FP64 sums in every order.
    let value = |i: usize, salt: usize| {
        let exponent = ((i * salt) % 11) as i32 - 5;
        let significand = 1.0 + ((i * (salt + 2)) % 128) as f32 / 128.0;
        let sign = if (i + i / k).is_multiple_of(3) {
            -1.0
        } else {
            1.0
        };
        round_bf16(sign * significand * 2.0_f32.powi(exponent))
    };
    Fixture {
        name: "bounded-magnitude-range",
        n,
        k,
        x: (0..k).map(|i| value(i, 3)).collect(),
        a: (0..n * k).map(|i| value(i, 5)).collect(),
        b: (0..n * k).map(|i| value(i, 7)).collect(),
        exact: false,
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

fn guarded(bytes: &[u8]) -> Vec<u8> {
    let mut storage = vec![GUARD_VALUE; bytes.len() + 2 * GUARD_BYTES];
    storage[GUARD_BYTES..GUARD_BYTES + bytes.len()].copy_from_slice(bytes);
    storage
}
fn payload(bytes: &[u8]) -> &[u8] {
    &bytes[GUARD_BYTES..bytes.len() - GUARD_BYTES]
}
fn guards_intact(buffers: &[Vec<u8>]) -> bool {
    buffers.iter().all(|b| {
        b[..GUARD_BYTES]
            .iter()
            .chain(&b[b.len() - GUARD_BYTES..])
            .all(|&v| v == GUARD_VALUE)
    })
}
fn output_pointer(buffer: &Buffer<'_>) -> u64 {
    buffer.pointer() + GUARD_BYTES as u64
}
fn read_outputs(buffers: &[Buffer<'_>]) -> Result<Vec<Vec<u8>>> {
    buffers.iter().map(download).collect()
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
    fp64: bool,
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
    let poison = guarded(&words_bytes(&vec![0x7fc1; f.n]));
    let raw_poison = guarded(
        &(0..f.n)
            .flat_map(|_| 0x7fc12345_u32.to_le_bytes())
            .collect::<Vec<_>>(),
    );
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
            .map(Buffer::pointer)
            .chain(outputs.iter().map(output_pointer))
            .collect(),
        dimensions: vec![f.n as u32, f.k as u32],
        grid: [f.n as u32, 2, 1],
    };
    let mut control: Vec<_> = (0..2)
        .map(|p| Launch {
            pointers: vec![
                inputs[0].pointer(),
                inputs[p + 1].pointer(),
                output_pointer(&controls[p]),
                output_pointer(&controls[p + 2]),
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
    let first = read_outputs(&outputs)?;
    let first_control = read_outputs(&controls)?;
    let mut metrics = Vec::new();
    let mut control_metrics = Vec::new();
    for p in 0..2 {
        metrics.push(compare(
            payload(&first[p]),
            payload(&first[p + 2]),
            &expected[p],
            f.k,
            fp64 || f.exact,
        ));
        control_metrics.push(compare(
            payload(&first_control[p]),
            payload(&first_control[p + 2]),
            &expected[p],
            f.k,
            true,
        ));
    }
    // Re-poison payloads before timing so stale first-run output cannot satisfy
    // the repeated-run check. First-run guard failures remain in the report.
    for buffers in [&outputs, &controls] {
        for (i, buffer) in buffers.iter().enumerate() {
            buffer.upload(if i < 2 { &poison } else { &raw_poison })?;
        }
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
    let repeated = read_outputs(&outputs)?;
    let repeated_control = read_outputs(&controls)?;
    let deterministic = first == repeated;
    let control_deterministic = first_control == repeated_control;
    let guards = [&first, &first_control, &repeated, &repeated_control]
        .iter()
        .all(|buffers| guards_intact(buffers));
    let against_control = control_bits(&first, &first_control);
    let control_exact = against_control.iter().all(|m| m["bit_identical"] == true);
    Ok(
        json!({"name":f.name,"shape":[1,f.n,f.k],"two_distinct_matrices":true,
        "candidate_symbol":if fp64 { "bf16_ab_decode_fp64" } else { "bf16_ab_decode_fp32" },
        "exact_fixture":f.exact,"candidate_requires_exact":fp64 || f.exact,
        "candidate":metrics,"baseline":control_metrics,"candidate_vs_control":against_control,
        "all_passed":deterministic && control_deterministic && guards && (!fp64 || control_exact)
            && metrics.iter().chain(&control_metrics).all(|m| m["all_passed"] == true),
        "deterministic":deterministic,"baseline_deterministic":control_deterministic,
        "guards_intact":guards,"guard_bytes_each_side":GUARD_BYTES,
        "completed":true,"poisoned_before_launch":true,"repoisoned_before_repeats":true,
        "candidate_timing":fast_time,"baseline_two_launch_timing":control_time}),
    )
}

fn control_bits(candidate: &[Vec<u8>], baseline: &[Vec<u8>]) -> Vec<Value> {
    (0..2)
        .map(|p| {
            let bf16 = payload(&candidate[p])
                .as_chunks::<2>()
                .0
                .iter()
                .zip(payload(&baseline[p]).as_chunks::<2>().0)
                .filter(|(a, b)| a != b)
                .count();
            let raw = payload(&candidate[p + 2])
                .as_chunks::<4>()
                .0
                .iter()
                .zip(payload(&baseline[p + 2]).as_chunks::<4>().0)
                .filter(|(a, b)| a != b)
                .count();
            json!({"projection":if p == 0 { "A" } else { "B" },
                "raw_bit_mismatches":raw,"bf16_bit_mismatches":bf16,
                "bit_identical":raw == 0 && bf16 == 0})
        })
        .collect()
}

fn compare(
    words: &[u8],
    raw: &[u8],
    expected: &reference::Projection,
    k: usize,
    exact: bool,
) -> Value {
    if words.len() != expected.bf16.len() * 2 || raw.len() != expected.raw.len() * 4 {
        return json!({"all_passed":false,"error":"output extent mismatch"});
    }
    let mut finite = true;
    let mut raw_bit_mismatches = 0;
    let mut bf16_bit_mismatches = 0;
    let mut failure_samples = Vec::new();
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
        let budget = if exact {
            0.0
        } else {
            reference::raw_budget(k, expected.absolute_products[i])
        };
        raw_bit_mismatches += usize::from(value.to_bits() != expected.raw[i].to_bits());
        bf16_bit_mismatches += usize::from(word != expected.bf16[i]);
        // Sum of the two endpoint half-ULPs safely handles binade boundaries.
        let rounding =
            0.5 * (reference::bf16_spacing(value) + reference::bf16_spacing(expected.raw[i]));
        let passed = if exact {
            value.to_bits() == expected.raw[i].to_bits() && word == expected.bf16[i]
        } else {
            raw_error <= budget && bf_error <= budget + rounding
        };
        if (!passed || !value.is_finite() || !bf.is_finite() || word != round_bf16(value))
            && failure_samples.len() < 8
        {
            failure_samples.push(json!({"index":i,
                "raw_bits":format!("{:08x}", value.to_bits()),
                "expected_raw_bits":format!("{:08x}", expected.raw[i].to_bits()),
                "bf16_bits":format!("{word:04x}"),
                "expected_bf16_bits":format!("{:04x}", expected.bf16[i])}));
        }
        mismatches += usize::from(!passed);
        rne_mismatches += usize::from(word != round_bf16(value));
        raw_max = raw_max.max(raw_error);
        bf_max = bf_max.max(bf_error);
        max_budget = max_budget.max(budget);
    }
    json!({"all_passed":finite && mismatches == 0 && rne_mismatches == 0,
        "finite":finite,"mismatches":mismatches,"stored_rne_mismatches":rne_mismatches,
        "raw_bit_mismatches":raw_bit_mismatches,"bf16_bit_mismatches":bf16_bit_mismatches,
        "failure_samples":failure_samples,
        "raw_max_abs":raw_max,"bf16_max_abs":bf_max,"largest_raw_budget":max_budget,
        "budget":if exact { "raw FP32 and BF16 bit identity; no tolerance" } else {
            "gamma(2*ceil(K/1024)+14)*sum_abs_products + 1e-37; BF16 adds endpoint half-ULPs" },
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
    Ok(
        json!({"warmup_pairs":3,"batches":3,"pairs_per_batch":10,"batch_event_ms":batches,
        "pair_event_us":micros,"logical_weight_bytes_per_pair":weight_bytes,
        "note":"warm-cache event batches include host submission gaps; not DRAM bandwidth; fixed FP32/control then FP64/control order"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_guards_exclude_payload_and_catch_both_sides() {
        let mut bytes = guarded(&[1, 2, 3, 4]);
        assert_eq!(payload(&bytes), [1, 2, 3, 4]);
        bytes[GUARD_BYTES] = 99;
        assert!(guards_intact(&[bytes.clone()]));
        bytes[0] ^= 1;
        assert!(!guards_intact(&[bytes.clone()]));
        bytes[0] ^= 1;
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(!guards_intact(&[bytes]));
    }

    #[test]
    fn exact_gate_rejects_raw_bit_drift_even_if_bf16_matches() {
        let f = fixture(1, 8, true, 1);
        let expected = reference::run(&f.x, &f.a, f.n, f.k).unwrap();
        let words = words_bytes(&expected.bf16);
        let raw = expected.raw[0].to_le_bytes();
        assert_eq!(
            compare(&words, &raw, &expected, f.k, true)["all_passed"],
            true
        );
        let drift = (expected.raw[0].to_bits() ^ 1).to_le_bytes();
        let result = compare(&words, &drift, &expected, f.k, true);
        assert_eq!(result["all_passed"], false);
        assert_eq!(result["raw_bit_mismatches"], 1);
        assert_eq!(result["bf16_bit_mismatches"], 0);
        assert_eq!(result["failure_samples"].as_array().unwrap().len(), 1);
        let poison = 0x7fc12345_u32.to_le_bytes();
        assert_eq!(
            compare(&words, &poison, &expected, f.k, true)["finite"],
            false
        );
        assert_eq!(
            compare(&[], &raw, &expected, f.k, true)["all_passed"],
            false
        );
    }

    #[test]
    fn cancellation_fixture_has_hand_computed_residue() {
        let f = cancellation_fixture();
        for (matrix, sign) in [(&f.a, 1.0_f32), (&f.b, -1.0_f32)] {
            let expected = reference::run(&f.x, matrix, f.n, f.k).unwrap();
            for (row, &raw) in expected.raw.iter().enumerate() {
                assert_eq!(raw, sign * 2560.0 * (row + 1) as f32 / 65536.0);
            }
        }
    }

    #[test]
    fn magnitude_fixture_has_exact_fp64_lattice_bound() {
        let f = magnitude_fixture();
        for matrix in [&f.a, &f.b] {
            let expected = reference::run(&f.x, matrix, f.n, f.k).unwrap();
            assert!(
                expected
                    .absolute_products
                    .iter()
                    .all(|&sum| { sum * 2.0_f64.powi(24) < 2.0_f64.powi(53) })
            );
        }
    }
}
