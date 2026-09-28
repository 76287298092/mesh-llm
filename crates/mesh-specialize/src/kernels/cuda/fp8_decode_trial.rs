//! Bounded schedule trial: independent decoded FP64 oracle plus unchanged GPU control.
//! Synthetic values at real model shapes are operator timings, not model timings.
use super::driver::{Buffer, Context, Event, Function, Module};
use crate::{
    entry_reference::round_bf16,
    kernels::fp8_decode_schedule::{BASELINE_KERNEL, VECTOR16_KERNEL},
    projection_reference::{self, LinearReference, QuantizedRows},
};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD: usize = 32;
const CANARY: u8 = 0xd3;
const REPEATS: usize = 3;
const WARMUP: usize = 3;
const TIMING_REPS: usize = 20;
const TIMING_SAMPLES: usize = 3;
const ORACLE_CHANNELS: usize = 257;

pub(super) fn enabled() -> Result<bool> {
    flag("MESH_SPECIALIZE_FP8_DECODE_TRIAL", false)
}

fn flag(name: &str, default: bool) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        _ => bail!("{name} must be 0 or 1"),
    }
}

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    let timing = flag("MESH_SPECIALIZE_FP8_DECODE_TRIAL_TIMING", true)?;
    let head = flag("MESH_SPECIALIZE_FP8_DECODE_TRIAL_HEAD", false)?;
    let baseline = module.function(BASELINE_KERNEL)?;
    let vector16 = module.function(VECTOR16_KERNEL)?;
    let mut cases = Vec::new();
    // Isolate all 254*254 finite code pairs as separate outputs, distributing
    // each A-code batch across the 16 byte positions of one aligned vector.
    for first in (0..254).step_by(16) {
        cases.push(run_case(
            ctx,
            [&baseline, &vector16],
            Fixture::finite_pairs(first),
            false,
        )?);
    }
    for k in [16, 32, 128, 5120, 6144, 17408] {
        for n in [1, 3, 4, 5, 17] {
            cases.push(run_case(
                ctx,
                [&baseline, &vector16],
                Fixture::new(n, k, Pattern::Irregular),
                false,
            )?);
        }
    }
    for pattern in [
        Pattern::Bf16Ties,
        Pattern::Cancellation,
        Pattern::MaxMagnitude,
    ] {
        for k in [16, 128, 32768] {
            cases.push(run_case(
                ctx,
                [&baseline, &vector16],
                Fixture::new(9, k, pattern),
                false,
            )?);
        }
    }
    // GDN QKV/Z/output, attention Q+gate/KV, FP8 MLP gate/down. Same shapes,
    // not checkpoint weights. Each case drops all allocations before the next.
    for [n, k] in [
        [10240, 5120],
        [6144, 5120],
        [5120, 6144],
        [12288, 5120],
        [1024, 5120],
        [17408, 5120],
        [5120, 17408],
    ] {
        cases.push(run_case(
            ctx,
            [&baseline, &vector16],
            Fixture::new(n, k, Pattern::Irregular),
            timing,
        )?);
    }
    if head {
        // No full-sized host or device clone of W is made. Explicit opt-in caps
        // the largest weight allocation at 1,271,398,400 bytes on each side.
        let required = 248320_usize * 5120 + 16 * 1024 * 1024;
        ensure!(
            ctx.memory()?.0 >= required + 128 * 1024 * 1024,
            "insufficient free GPU memory for bounded vocabulary trial"
        );
        cases.push(run_case(
            ctx,
            [&baseline, &vector16],
            Fixture::new(248320, 5120, Pattern::Irregular),
            timing,
        )?);
    }
    Ok(json!({
        "kind":"fp8-exact-vector16-schedule-trial", "device":ctx.info(),
        "baseline_kernel":BASELINE_KERNEL, "candidate_kernel":VECTOR16_KERNEL,
        "baseline_resources":baseline.resources()?, "candidate_resources":vector16.resources()?,
        "dynamic_shared_bytes":0, "jit_log":module.jit_log(),
        "all_passed":cases.iter().all(|c| c["all_passed"] == true), "cases":cases,
        "correctness_repeats_per_kernel":REPEATS, "output_repoisoned_each_repeat":true,
        "isolated_finite_code_pairs_with_full_cpu_oracle":254 * 254,
        "guard_bytes_each_side_each_allocation":GUARD,
        "timing_enabled":timing, "timing_warmup_per_kernel":WARMUP,
        "timing_repetitions_per_sample":TIMING_REPS, "timing_samples":TIMING_SAMPLES,
        "timing_order":"alternating baseline/vector16 order by sample",
        "vocabulary_head_included":head,
        "vocabulary_head_skip_reason":if head { Value::Null } else { json!("set MESH_SPECIALIZE_FP8_DECODE_TRIAL_HEAD=1; largest host/device weight allocation is 1,271,398,400 bytes each") },
        "scope":"synthetic finite FP8 operator checks and CUDA-event timings; no model-throughput, bandwidth, or model-qualification claim",
        "oracle":"unchanged reference/projections.rs::linear, independently decoded FP64 accumulation; full coverage for small N, explicitly sampled columns for real shapes"
    }))
}

#[derive(Clone, Copy)]
enum Pattern {
    Irregular,
    Bf16Ties,
    Cancellation,
    MaxMagnitude,
    FinitePairs,
}
impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Self::Irregular => "irregular-all-finite-code-domain",
            Self::Bf16Ties => "positive-negative-even-odd-BF16-ties",
            Self::Cancellation => "large-cancellation-subnormal-residual",
            Self::MaxMagnitude => "maximum-magnitude-signed-sums",
            Self::FinitePairs => "isolated-finite-code-pairs",
        }
    }
}

struct Fixture {
    n: usize,
    k: usize,
    pattern: Pattern,
    a: Vec<u8>,
    w: Vec<u8>,
    sa: f32,
    sw: Vec<u16>,
}
impl Fixture {
    fn new(n: usize, k: usize, pattern: Pattern) -> Self {
        let finite: Vec<u8> = (0..=255_u16)
            .map(|v| v as u8)
            .filter(|v| v & 127 != 127)
            .collect();
        let mut a: Vec<_> = (0..k).map(|i| finite[(i * 37 + i / 7) % 254]).collect();
        let mut w: Vec<_> = (0..n * k)
            .map(|i| finite[(i * 53 + i / 11 + i / k * 19) % 254])
            .collect();
        match pattern {
            Pattern::Irregular | Pattern::FinitePairs => {}
            Pattern::Bf16Ties => {
                a.fill(56); // 1.0: exact tie values 1+1/256 and 1+3/256.
                for (c, row) in w.chunks_exact_mut(k).enumerate() {
                    row.fill(0);
                    let sign = if c % 4 >= 2 { 128 } else { 0 };
                    row[0] = 56 | sign;
                    row[1] = (if c % 2 == 0 { 2 } else { 6 }) | sign;
                }
            }
            Pattern::Cancellation => {
                a.fill(56);
                for (c, row) in w.chunks_exact_mut(k).enumerate() {
                    row.fill(0);
                    row[0] = 126;
                    row[1] = 1;
                    row[2] = 254;
                    row[k - 1] = if c % 2 == 0 { 1 } else { 129 };
                }
            }
            Pattern::MaxMagnitude => {
                a.fill(126);
                for (c, row) in w.chunks_exact_mut(k).enumerate() {
                    row.fill(if c % 2 == 0 { 126 } else { 254 });
                }
            }
        }
        let irregular = matches!(pattern, Pattern::Irregular);
        Self {
            n,
            k,
            pattern,
            a,
            w,
            sa: if irregular { 0.073123 } else { 1.0 },
            sw: (0..n)
                .map(|i| {
                    round_bf16(if irregular {
                        (i % 11 + 1) as f32 / 17.0
                    } else {
                        1.0
                    })
                })
                .collect(),
        }
    }

    fn finite_pairs(first: usize) -> Self {
        let finite: Vec<u8> = (0..=255_u16)
            .map(|v| v as u8)
            .filter(|v| v & 127 != 127)
            .collect();
        let lanes = (254 - first).min(16);
        let n = lanes * 254;
        let mut a = vec![0; 16];
        a[..lanes].copy_from_slice(&finite[first..first + lanes]);
        let mut w = vec![0; n * 16];
        for lane in 0..lanes {
            for (index, &code) in finite.iter().enumerate() {
                w[(lane * 254 + index) * 16 + lane] = code;
            }
        }
        Self {
            n,
            k: 16,
            pattern: Pattern::FinitePairs,
            a,
            w,
            sa: 1.0,
            sw: vec![round_bf16(1.0); n],
        }
    }

    fn oracle(&self) -> Result<(Vec<usize>, LinearReference)> {
        let count = if matches!(self.pattern, Pattern::FinitePairs) {
            self.n
        } else {
            self.n.min(ORACLE_CHANNELS)
        };
        let columns: Vec<_> = if count == self.n {
            (0..self.n).collect()
        } else {
            (0..count).map(|i| i * (self.n - 1) / (count - 1)).collect()
        };
        let mut weights = Vec::with_capacity(columns.len() * self.k);
        let mut scales = Vec::with_capacity(columns.len());
        for &c in &columns {
            weights.extend_from_slice(&self.w[c * self.k..(c + 1) * self.k]);
            scales.push(self.sw[c]);
        }
        let expected = projection_reference::linear(
            &QuantizedRows {
                codes: self.a.clone(),
                scales: vec![self.sa],
            },
            &weights,
            &scales,
            self.k,
        )?;
        Ok((columns, expected))
    }
}

struct Guarded<'ctx> {
    buffer: Buffer<'ctx>,
    bytes: usize,
}
impl<'ctx> Guarded<'ctx> {
    fn new(ctx: &'ctx Context, bytes: usize) -> Result<Self> {
        let buffer = Buffer::new(ctx, bytes + GUARD * 2)?;
        buffer.upload_at(0, &[CANARY; GUARD])?;
        buffer.upload_at(GUARD + bytes, &[CANARY; GUARD])?;
        Ok(Self { buffer, bytes })
    }
    fn upload(ctx: &'ctx Context, bytes: &[u8]) -> Result<Self> {
        let result = Self::new(ctx, bytes.len())?;
        result.buffer.upload_at(GUARD, bytes)?;
        Ok(result)
    }
    fn pointer(&self) -> u64 {
        self.buffer.pointer() + GUARD as u64
    }
    fn poison(&self, byte: u8) -> Result<()> {
        self.buffer.upload_at(GUARD, &vec![byte; self.bytes])
    }
    fn download(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.bytes];
        self.buffer.download_at(GUARD, &mut bytes)?;
        Ok(bytes)
    }
    fn guards_ok(&self) -> Result<bool> {
        let mut before = [0; GUARD];
        let mut after = [0; GUARD];
        self.buffer.download_at(0, &mut before)?;
        self.buffer.download_at(GUARD + self.bytes, &mut after)?;
        Ok(before == [CANARY; GUARD] && after == [CANARY; GUARD])
    }
}

struct DeviceCase<'ctx> {
    // A/W/scales/output/raw all have distinct guarded storage. Payload starts at
    // a 32-byte offset, preserving the CUDA allocation's vector alignment.
    allocations: [Guarded<'ctx>; 6],
    n: usize,
    k: usize,
}
impl<'ctx> DeviceCase<'ctx> {
    fn new(ctx: &'ctx Context, fixture: &Fixture) -> Result<Self> {
        let sw: Vec<_> = fixture.sw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let allocations = [
            Guarded::upload(ctx, &fixture.a)?,
            Guarded::upload(ctx, &fixture.w)?,
            Guarded::upload(ctx, &fixture.sa.to_le_bytes())?,
            Guarded::upload(ctx, &sw)?,
            Guarded::new(ctx, fixture.n * 2)?,
            Guarded::new(ctx, fixture.n * 4)?,
        ];
        ensure!(
            allocations[0].pointer().is_multiple_of(16)
                && allocations[1].pointer().is_multiple_of(16),
            "unaligned trial allocation"
        );
        Ok(Self {
            allocations,
            n: fixture.n,
            k: fixture.k,
        })
    }
    fn launch(&self, function: &Function<'_, '_>) -> Result<()> {
        let mut pointers = self.allocations.each_ref().map(Guarded::pointer);
        let mut dimensions = [1_u32, u32::try_from(self.n)?, u32::try_from(self.k)?];
        let mut args: [*mut c_void; 9] = [std::ptr::null_mut(); 9];
        for (arg, value) in args[..6].iter_mut().zip(pointers.iter_mut()) {
            *arg = (value as *mut u64).cast();
        }
        for (arg, value) in args[6..].iter_mut().zip(dimensions.iter_mut()) {
            *arg = (value as *mut u32).cast();
        }
        // SAFETY: Both functions use exactly this nine-argument ABI, grid and block.
        // Fixed fixtures use finite codes, M=1, bounded N/K, positive scales and
        // aligned complete K vectors. All disjoint guarded buffers outlive the launch.
        unsafe {
            function.launch(
                [u32::try_from(self.n.div_ceil(4))?, 1, 1],
                [128, 1, 1],
                0,
                &mut args,
            )
        }
    }
    fn check_run(&self, ctx: &Context, function: &Function<'_, '_>, poison: u8) -> Result<Output> {
        self.allocations[4].poison(poison)?;
        self.allocations[5].poison(poison ^ 0xff)?;
        let launched = self.launch(function);
        let synchronized = ctx.synchronize();
        launched?;
        synchronized?;
        let bf16 = self.allocations[4]
            .download()?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect();
        let raw = self.allocations[5]
            .download()?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        let mut guards_ok = true;
        for allocation in &self.allocations {
            guards_ok &= allocation.guards_ok()?;
        }
        Ok(Output {
            bf16,
            raw,
            guards_ok,
        })
    }
}

#[derive(PartialEq, Eq)]
struct Output {
    bf16: Vec<u16>,
    raw: Vec<u32>,
    guards_ok: bool,
}

fn differences<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a != b).count() + left.len().abs_diff(right.len())
}

fn oracle_check(output: &Output, columns: &[usize], expected: &LinearReference) -> Value {
    let bf16_differences = columns
        .iter()
        .zip(&expected.normalized)
        .filter(|(c, v)| output.bf16[**c] != **v)
        .count();
    let raw_differences = columns
        .iter()
        .zip(&expected.unrounded)
        .filter(|(c, v)| output.raw[**c] != v.to_bits())
        .count();
    let finite = output.raw.iter().all(|v| f32::from_bits(*v).is_finite())
        && output.bf16.iter().all(|v| v & 0x7f80 != 0x7f80);
    json!({"bf16_bit_differences":bf16_differences, "fp32_bit_differences":raw_differences,
        "all_outputs_finite":finite, "guards_ok":output.guards_ok,
        "all_passed":bf16_differences == 0 && raw_differences == 0 && finite && output.guards_ok})
}

fn run_case(
    ctx: &Context,
    functions: [&Function<'_, '_>; 2],
    fixture: Fixture,
    timing: bool,
) -> Result<Value> {
    ensure!(
        (1..=262144).contains(&fixture.n)
            && (16..=32768).contains(&fixture.k)
            && fixture.k.is_multiple_of(16),
        "invalid trial shape"
    );
    let (columns, expected) = fixture.oracle()?;
    let device = DeviceCase::new(ctx, &fixture)?;
    let mut checks = Vec::new();
    let mut first: Option<[Output; 2]> = None;
    for repeat in 0..REPEATS {
        let poison = [0xa5, 0xff, 0x3c][repeat];
        let baseline = device.check_run(ctx, functions[0], poison)?;
        let candidate = device.check_run(ctx, functions[1], poison)?;
        let control_bf16 = differences(&baseline.bf16, &candidate.bf16);
        let control_raw = differences(&baseline.raw, &candidate.raw);
        let baseline_oracle = oracle_check(&baseline, &columns, &expected);
        let candidate_oracle = oracle_check(&candidate, &columns, &expected);
        let repeat_equal = first
            .as_ref()
            .is_none_or(|f| f[0] == baseline && f[1] == candidate);
        let passed = baseline_oracle["all_passed"] == true
            && candidate_oracle["all_passed"] == true
            && control_bf16 == 0
            && control_raw == 0
            && repeat_equal;
        checks.push(json!({"repeat":repeat, "baseline_oracle":baseline_oracle,
            "candidate_oracle":candidate_oracle, "baseline_candidate_bf16_bit_differences":control_bf16,
            "baseline_candidate_fp32_bit_differences":control_raw, "repeat_bit_identity":repeat_equal,
            "all_passed":passed}));
        if first.is_none() {
            first = Some([baseline, candidate]);
        }
    }
    let correctness = checks.iter().all(|c| c["all_passed"] == true);
    let timings = if timing && correctness {
        Some(time_pair(ctx, &device, functions)?)
    } else {
        None
    };
    Ok(json!({
        "shape_m_n_k":[1, fixture.n, fixture.k], "fixture":fixture.pattern.name(),
        "weight_bytes":fixture.w.len(), "grid":[fixture.n.div_ceil(4), 1, 1], "block":[128, 1, 1],
        "cpu_oracle_columns":columns, "cpu_oracle_output_count":columns.len(),
        "isolated_finite_code_pairs":if matches!(fixture.pattern, Pattern::FinitePairs) { fixture.n } else { 0 },
        "cpu_oracle_product_count":columns.len() * fixture.k,
        "cpu_oracle_full_output_coverage":columns.len() == fixture.n,
        "control_candidate_outputs_per_repeat":fixture.n, "checks":checks,
        "timing":timings, "all_passed":correctness,
    }))
}

fn time_pair(
    ctx: &Context,
    device: &DeviceCase<'_>,
    functions: [&Function<'_, '_>; 2],
) -> Result<Value> {
    // Allocate events and resolve functions before timing. No host upload/download,
    // output poisoning, allocation or CPU oracle work is inside an event interval.
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let measured = (|| -> Result<Value> {
        for _ in 0..WARMUP {
            for f in functions {
                device.launch(f)?;
            }
        }
        ctx.synchronize()?;
        let mut samples: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for sample in 0..TIMING_SAMPLES {
            for index in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                start.record()?;
                for _ in 0..TIMING_REPS {
                    device.launch(functions[index])?;
                }
                end.record()?;
                end.synchronize()?;
                samples[index].push(end.elapsed_since(&start)? / TIMING_REPS as f32);
            }
        }
        let mut guards_ok = true;
        for allocation in &device.allocations {
            guards_ok &= allocation.guards_ok()?;
        }
        ensure!(guards_ok, "guard corruption after timing");
        ensure!(
            samples.iter().flatten().all(|v| v.is_finite() && *v > 0.0),
            "invalid CUDA timing"
        );
        Ok(
            json!({"baseline_ms_per_launch":samples[0], "vector16_ms_per_launch":samples[1],
            "repetitions":TIMING_REPS, "samples":TIMING_SAMPLES, "warmup":WARMUP,
            "guards_ok":guards_ok, "scope":"CUDA event batches, includes possible host submission gaps; not end-to-end model timing"}),
        )
    })();
    // Even a failed launch or event operation drains submitted work before buffers drop.
    let synchronized = ctx.synchronize();
    let result = measured?;
    synchronized?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_are_real_signed_even_and_odd_bf16_midpoints() {
        let f = Fixture::new(4, 16, Pattern::Bf16Ties);
        let (columns, oracle) = f.oracle().unwrap();
        assert_eq!(columns, [0, 1, 2, 3]);
        assert_eq!(oracle.normalized, [0x3f80, 0x3f82, 0xbf80, 0xbf82]);
        assert_eq!(
            oracle.unrounded,
            [
                1.0 + 1.0 / 256.0,
                1.0 + 3.0 / 256.0,
                -1.0 - 1.0 / 256.0,
                -1.0 - 3.0 / 256.0
            ]
        );
    }

    #[test]
    fn isolated_pairs_cover_every_finite_pair_once_with_full_oracle_coverage() {
        let mut seen = vec![false; 65536];
        let mut count = 0;
        for first in (0..254).step_by(16) {
            let f = Fixture::finite_pairs(first);
            let (columns, oracle) = f.oracle().unwrap();
            assert_eq!(columns.len(), f.n);
            assert_eq!(oracle.normalized.len(), f.n);
            for (row, w) in f.w.as_chunks::<16>().0.iter().enumerate() {
                let lane = row / 254;
                let pair = usize::from(f.a[lane]) * 256 + usize::from(w[lane]);
                assert!(!seen[pair]);
                seen[pair] = true;
                count += 1;
            }
        }
        assert_eq!(count, 254 * 254);
        for a in 0..256 {
            for w in 0..256 {
                assert_eq!(seen[a * 256 + w], a & 127 != 127 && w & 127 != 127);
            }
        }
    }

    #[test]
    fn sampled_oracle_reports_unique_spread_columns_including_tails() {
        let f = Fixture::new(1001, 16, Pattern::Irregular);
        let (columns, oracle) = f.oracle().unwrap();
        assert_eq!(columns.len(), ORACLE_CHANNELS);
        assert_eq!(columns.first(), Some(&0));
        assert_eq!(columns.last(), Some(&1000));
        assert!(columns.windows(2).all(|p| p[0] < p[1]));
        assert_eq!(oracle.unrounded.len(), ORACLE_CHANNELS);
        assert!(f.a.iter().chain(&f.w).all(|v| v & 127 != 127));
    }
}
