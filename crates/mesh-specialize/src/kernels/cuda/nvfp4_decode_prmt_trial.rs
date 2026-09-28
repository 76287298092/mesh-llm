//! Bounded operator trial. Full control-bit comparisons; independent FP64 oracle.
//! Real trials read verified resident tensors, never parse or substitute an artifact.
use super::{
    driver::{Buffer, Context, Event, Function, Module},
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::schema::DType,
    entry_reference::round_bf16,
    nvfp4_decode_prmt_reference::{self as fixtures, Fixture},
    nvfp4_linear_reference::{self as oracle, Matrix},
};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const CONTROL: &str = "nvfp4_decode_exact";
const CANDIDATE: &str = "nvfp4_decode_exact_prmt";
const GUARD: usize = 64;
const CANARY: u8 = 0xd3;
const REPEATS: usize = 3;
const WARMUPS: usize = 3;
const SAMPLES: usize = 5;
const ORACLE_ROWS: usize = 257;

pub(super) fn enabled() -> Result<bool> {
    flag("MESH_SPECIALIZE_NVFP4_DECODE_TRIAL", false)
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
    ensure!(
        fixtures::exhaustive_proof(),
        "PRMT host semantic proof failed"
    );
    let mut cases = Vec::new();
    for upper in [false, true] {
        for position in 0..16 {
            cases.push(case(
                ctx,
                module,
                &fixtures::controls(upper, position),
                [1.0; 2],
                false,
                false,
            )?);
        }
    }
    for code in 0..16 {
        cases.push(case(
            ctx,
            module,
            &fixtures::code_pairs(code),
            [1.0; 2],
            false,
            false,
        )?);
    }
    for scale in 0..=126 {
        cases.push(case(
            ctx,
            module,
            &fixtures::scale_pairs(scale),
            [1.0; 2],
            false,
            false,
        )?);
    }
    for k in [16, 32, 48, 80, 496, 512, 528, 1024, 5120, 17408, 32768] {
        for n in [1, 3, 4, 5, 13, 17, 33] {
            cases.push(case(
                ctx,
                module,
                &fixtures::patterned(n, k),
                [1.25, 3.0],
                false,
                false,
            )?);
        }
    }
    cases.push(case(
        ctx,
        module,
        &fixtures::extrema(),
        [1.0; 2],
        false,
        false,
    )?);
    Ok(json!({
        "all_passed": cases.iter().all(|c| c["all_passed"] == true), "cases": cases,
        "host_controls_proven":65536,"host_halfword_neighbour_combinations":524288,
        "gpu_control_coverage":"all 65536 weight halfwords, 16 basis positions; activation side not exhaustive",
        "resources":{"control":module.function(CONTROL)?.resources()?,
            "candidate":module.function(CANDIDATE)?.resources()?},
        "scope":"synthetic operator correctness only; real-weight trial runs after verified residency"
    }))
}

/// Two real tensor shapes from the current verified model. Synthetic BF16 input,
/// quantized with the existing independent quantizer and the actual input global.
/// Full weight tensor and scale digests are included so source provenance is explicit.
pub(super) fn run_real(
    ctx: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
) -> Result<Value> {
    let timing = flag("MESH_SPECIALIZE_NVFP4_DECODE_TRIAL_TIMING", false)?;
    let mut cases = Vec::new();
    for (suffix, n, k) in [("gate_proj", 17408, 5120), ("down_proj", 5120, 17408)] {
        let prefix = format!("tensors/model.language_model.layers.0.mlp.{suffix}");
        let packed_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        weights.tensor(
            &packed_name,
            DType::U8,
            &[n as u64, (k / 2) as u64],
            (n * k / 2) as u64,
        )?;
        weights.tensor(
            &scale_name,
            DType::Fp8E4m3,
            &[n as u64, (k / 16) as u64],
            (n * k / 16) as u64,
        )?;
        let globals = [
            weights.positive_scalar(&format!("{prefix}.input_global_scale"))?,
            weights.positive_scalar(&format!("{prefix}.weight_global_scale"))?,
        ];
        let input = (0..k)
            .map(|i| round_bf16(((i * 37 % 257) as f32 - 128.0) / 64.0))
            .collect::<Vec<_>>();
        let quantized = crate::nvfp4_quantize_reference::run(&input, 1, k, globals[0])?;
        let mut f = Fixture {
            name: prefix,
            n,
            k,
            activation: quantized.packed,
            activation_scales: quantized.scales,
            weights: vec![0; n * k / 2],
            weight_scales: vec![0; n * k / 16],
        };
        weights.read_range(&packed_name, 0, &mut f.weights)?;
        weights.read_range(&scale_name, 0, &mut f.weight_scales)?;
        let mut report = case(ctx, module, &f, globals, true, timing)?;
        report["weight_object"] = json!(weights.object(&packed_name)?);
        report["scale_object"] = json!(weights.object(&scale_name)?);
        report["activation_source"] =
            json!("deterministic synthetic BF16; existing quantizer, actual global scale");
        cases.push(report);
    }
    Ok(
        json!({"all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,
        "scope":"current verified artifact weights; operator timing is not whole-model throughput"}),
    )
}

struct DeviceCase<'ctx> {
    inputs: Vec<Buffer<'ctx>>,
    bf16: Buffer<'ctx>,
    raw: Buffer<'ctx>,
    n: usize,
    k: usize,
    factor: f32,
}
#[derive(PartialEq, Eq)]
struct Snapshot {
    bf16: Vec<u16>,
    raw: Vec<u32>,
    guards: bool,
}

fn guarded<'ctx>(ctx: &'ctx Context, payload: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(ctx, payload.len() + 2 * GUARD)?;
    buffer.upload(&vec![CANARY; buffer.len()])?;
    buffer.upload_at(GUARD, payload)?;
    Ok(buffer)
}
fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}
fn guards(bytes: &[u8]) -> bool {
    bytes[..GUARD]
        .iter()
        .chain(&bytes[bytes.len() - GUARD..])
        .all(|&b| b == CANARY)
}
fn payload(bytes: &[u8]) -> &[u8] {
    &bytes[GUARD..bytes.len() - GUARD]
}

impl<'ctx> DeviceCase<'ctx> {
    fn new(ctx: &'ctx Context, f: &Fixture, globals: [f32; 2]) -> Result<Self> {
        let product = globals[0] * globals[1];
        let factor = 1.0 / product;
        ensure!(
            globals
                .into_iter()
                .chain([product, factor])
                .all(|v| v.is_finite() && v > 0.0),
            "invalid globals"
        );
        ensure!(
            (1..=32768).contains(&f.n) && (16..=32768).contains(&f.k) && f.k.is_multiple_of(16),
            "invalid shape"
        );
        ensure!(
            f.activation.len() == f.k / 2
                && f.weights.len() == f.n * f.k / 2
                && f.activation_scales.len() == f.k / 16
                && f.weight_scales.len() == f.n * f.k / 16,
            "invalid extents"
        );
        ensure!(
            f.activation_scales
                .iter()
                .chain(&f.weight_scales)
                .all(|&s| s <= 126),
            "invalid scales"
        );
        let inputs = [
            &f.activation,
            &f.weights,
            &f.activation_scales,
            &f.weight_scales,
        ]
        .into_iter()
        .map(|b| guarded(ctx, b))
        .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inputs,
            bf16: guarded(ctx, &vec![0xff; f.n * 2])?,
            raw: guarded(ctx, &vec![0xff; f.n * 4])?,
            n: f.n,
            k: f.k,
            factor,
        })
    }
    fn poison(&self, byte: u8) -> Result<()> {
        self.bf16.upload_at(GUARD, &vec![byte; self.n * 2])?;
        self.raw.upload_at(GUARD, &vec![byte; self.n * 4])
    }
    fn snapshot(&self) -> Result<Snapshot> {
        let bf16 = download(&self.bf16)?;
        let raw = download(&self.raw)?;
        Ok(Snapshot {
            bf16: payload(&bf16)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect(),
            raw: payload(&raw)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect(),
            guards: guards(&bf16) && guards(&raw),
        })
    }
    fn inputs_intact(&self, f: &Fixture) -> Result<bool> {
        for (buffer, expected) in self.inputs.iter().zip([
            &f.activation,
            &f.weights,
            &f.activation_scales,
            &f.weight_scales,
        ]) {
            let bytes = download(buffer)?;
            if !guards(&bytes) || payload(&bytes) != expected.as_slice() {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn launch(&self, function: &Function<'_, '_>) -> Result<()> {
        let mut pointers = self
            .inputs
            .iter()
            .map(|b| b.pointer() + GUARD as u64)
            .collect::<Vec<_>>();
        pointers.extend([
            self.bf16.pointer() + GUARD as u64,
            self.raw.pointer() + GUARD as u64,
        ]);
        let mut dimensions = [1_u32, self.n as u32, self.k as u32];
        let mut factor = self.factor;
        let mut args = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        args.extend(dimensions.iter_mut().map(|d| (d as *mut u32).cast()));
        args.push((&mut factor as *mut f32).cast());
        // SAFETY: Validated extents, finite scale domain, four-byte aligned interior
        // pointers, disjoint buffers and exact ten-argument ABI. Callers drain work.
        unsafe { function.launch([self.n.div_ceil(4) as u32, 1, 1], [128, 1, 1], 0, &mut args) }
    }
    fn execute(&self, ctx: &Context, function: &Function<'_, '_>) -> Result<()> {
        let launch = self.launch(function);
        let sync = ctx.synchronize();
        launch?;
        sync
    }
}

fn channels(n: usize, sampled: bool) -> Vec<usize> {
    if !sampled || n <= ORACLE_ROWS {
        return (0..n).collect();
    }
    (0..ORACLE_ROWS)
        .map(|i| i * (n - 1) / (ORACLE_ROWS - 1))
        .collect()
}
fn expected(f: &Fixture, globals: [f32; 2], rows: &[usize]) -> Result<Snapshot> {
    let mut packed = Vec::with_capacity(rows.len() * f.k / 2);
    let mut scales = Vec::with_capacity(rows.len() * f.k / 16);
    for &row in rows {
        packed.extend_from_slice(&f.weights[row * f.k / 2..(row + 1) * f.k / 2]);
        scales.extend_from_slice(&f.weight_scales[row * f.k / 16..(row + 1) * f.k / 16]);
    }
    let expected = oracle::run(
        Matrix {
            packed: &f.activation,
            scales: &f.activation_scales,
            rows: 1,
            global: globals[0],
        },
        Matrix {
            packed: &packed,
            scales: &scales,
            rows: rows.len(),
            global: globals[1],
        },
        f.k,
    )?;
    Ok(Snapshot {
        bf16: expected.normalized,
        raw: expected.unrounded.iter().map(|v| v.to_bits()).collect(),
        guards: true,
    })
}
fn compare(actual: &Snapshot, expected: &Snapshot, rows: &[usize]) -> Value {
    let raw = rows
        .iter()
        .zip(&expected.raw)
        .filter(|(r, v)| actual.raw[**r] != **v)
        .count();
    let bf16 = rows
        .iter()
        .zip(&expected.bf16)
        .filter(|(r, v)| actual.bf16[**r] != **v)
        .count();
    let finite = actual.raw.iter().all(|&v| f32::from_bits(v).is_finite())
        && actual
            .bf16
            .iter()
            .all(|&v| f32::from_bits(u32::from(v) << 16).is_finite());
    let extent = rows.len() == expected.raw.len() && rows.len() == expected.bf16.len();
    json!({"all_passed":extent && finite && actual.guards && expected.guards && raw==0 && bf16==0,
        "raw_fp32_bit_differences":raw,"bf16_bit_differences":bf16,"compared_outputs":rows.len(),
        "all_outputs_finite":finite,"output_guards_intact":actual.guards,"extent_match":extent})
}

fn case(
    ctx: &Context,
    module: &Module<'_>,
    f: &Fixture,
    globals: [f32; 2],
    sampled: bool,
    timed: bool,
) -> Result<Value> {
    let baseline = module.function(CONTROL)?;
    let candidate = module.function(CANDIDATE)?;
    let oracle_rows = channels(f.n, sampled);
    let oracle = expected(f, globals, &oracle_rows)?;
    let device = DeviceCase::new(ctx, f, globals)?;
    device.execute(ctx, &baseline)?;
    let control = device.snapshot()?;
    let control_oracle = compare(&control, &oracle, &oracle_rows);
    let all_rows = (0..f.n).collect::<Vec<_>>();
    let mut repeats = Vec::new();
    for poison in [0xff, 0x7f, 0xa5].into_iter().take(REPEATS) {
        device.poison(poison)?;
        device.execute(ctx, &candidate)?;
        let actual = device.snapshot()?;
        repeats.push(json!({"control":compare(&actual,&control,&all_rows),
            "oracle":compare(&actual,&oracle,&oracle_rows),"poison":poison}));
    }
    let inputs_intact = device.inputs_intact(f)?;
    let correct = inputs_intact
        && control_oracle["all_passed"] == true
        && repeats
            .iter()
            .all(|r| r["control"]["all_passed"] == true && r["oracle"]["all_passed"] == true);
    let timing = if correct && timed {
        timing(ctx, [&baseline, &candidate], &device)?
    } else {
        Value::Null
    };
    // Recheck both outputs after timing. Performance loops may not weaken correctness.
    device.poison(0xff)?;
    device.execute(ctx, &baseline)?;
    let control_repeat = compare(&device.snapshot()?, &control, &all_rows);
    device.poison(0x7f)?;
    device.execute(ctx, &candidate)?;
    let candidate_repeat = compare(&device.snapshot()?, &control, &all_rows);
    let intact_after = device.inputs_intact(f)?;
    Ok(
        json!({"name":f.name,"shape":[1,f.n,f.k],"globals":globals,"global_factor":device.factor,
        "all_passed":correct && intact_after && control_repeat["all_passed"]==true && candidate_repeat["all_passed"]==true,
        "control_oracle":control_oracle,"repeats":repeats,"control_repeat":control_repeat,
        "candidate_repeat":candidate_repeat,"inputs_and_guards_intact":inputs_intact && intact_after,
        "control_coverage":"all output FP32 and BF16 bits","oracle_channel_count":oracle_rows.len(),
        "oracle_channels":if sampled {json!(oracle_rows)} else {Value::Null},
        "oracle_coverage":if sampled {"257 evenly spaced channels including first/last; remaining control-only"}else{"all channels"},
        "timing":timing}),
    )
}

fn timing(
    ctx: &Context,
    functions: [&Function<'_, '_>; 2],
    device: &DeviceCase<'_>,
) -> Result<Value> {
    for function in functions {
        for _ in 0..WARMUPS {
            device.execute(ctx, function)?;
        }
    }
    let start = Event::new(ctx)?;
    let end = Event::new(ctx)?;
    let mut samples = [Vec::new(), Vec::new()];
    for sample in 0..SAMPLES {
        for index in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
            start.record()?;
            let result = device
                .launch(functions[index])
                .and_then(|()| end.record())
                .and_then(|()| end.synchronize());
            if let Err(error) = result {
                let _ = ctx.synchronize();
                return Err(error);
            }
            let elapsed = end.elapsed_since(&start)?;
            ensure!(
                elapsed.is_finite() && elapsed > 0.0,
                "invalid event duration"
            );
            samples[index].push(elapsed);
        }
    }
    Ok(
        json!({"warmups":WARMUPS,"alternating_order":true,"control_event_ms":samples[0],
        "candidate_event_ms":samples[1],"scope":"single operator, not model throughput"}),
    )
}
