//! Bounded synthetic qualification for encoded embedding and FP32 gate parameters.
//! No model artifact, profile admission, or timing claim.

use super::driver::{Buffer, Context, Function, Module};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    fp8_embedding_gather_reference as embedding, gdn_gates_f32_params_reference as gates,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD: usize = 256;
const EPSILON: f32 = 1e-6;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "native parameter trial requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let gather = module.function("fp8_embedding_gather")?;
    let norm = module.function("embedding_norm_bf16")?;
    let f32_gate = module.function("gdn_gates_f32_params")?;
    let bf16_gate = module.function("gdn_gates")?;
    let resources = json!({
        "fp8_embedding_gather": gather.resources()?,
        "embedding_norm_bf16": norm.resources()?,
        "gdn_gates_f32_params": f32_gate.resources()?,
        "gdn_gates": bf16_gate.resources()?,
    });
    let mut cases = Vec::new();
    for width in [1, 7, 257, 5120] {
        cases.push(case_result(
            "embedding",
            embedding_case(&context, &gather, &norm, width),
        ));
    }
    for (rows, low_log, low_bias) in [
        (1, false, false),
        (7, false, false),
        (7, true, false),
        (7, false, true),
        (7, true, true),
    ] {
        cases.push(case_result(
            "gates",
            gate_case(
                &context,
                &f32_gate,
                &bf16_gate,
                gate_fixture(rows, low_log, low_bias),
            ),
        ));
    }
    Ok(json!({
        "kind": "native-parameter-synthetic-check", "schema_version": 1,
        "device": info, "jit_log": module.jit_log(), "resources": resources,
        "all_passed": cases.iter().all(|v| v["all_passed"] == true), "cases": cases,
        "scope": "synthetic encoded gather, pre-norm BF16 rounding, identity-ID alias safety, and gate parameter precision only",
        "model_import_required": false, "model_executable": false,
        "full_ninfer_arithmetic_parity": false, "performance_claim": false,
        "guards": {"prefix_bytes": GUARD, "suffix_bytes": GUARD, "outputs_poisoned": true},
    }))
}

fn case_result(kind: &str, result: Result<Value>) -> Value {
    result.unwrap_or_else(
        |error| json!({"kind": kind, "all_passed": false, "error": format!("{error:#}")}),
    )
}

/// Naturally aligned interior pointers with disjoint readable/writable canaries.
struct Guarded<'ctx> {
    buffer: Buffer<'ctx>,
    initial: Vec<u8>,
}

impl<'ctx> Guarded<'ctx> {
    fn new(context: &'ctx Context, initial: Vec<u8>) -> Result<Self> {
        ensure!(!initial.is_empty(), "empty trial buffer");
        let buffer = Buffer::new(context, initial.len() + 2 * GUARD)?;
        let this = Self { buffer, initial };
        this.reset()?;
        Ok(this)
    }

    fn poison(context: &'ctx Context, count: usize, bf16: bool) -> Result<Self> {
        let pattern = if bf16 {
            vec![0xc1, 0x7f]
        } else {
            vec![0x45, 0x23, 0xc1, 0x7f]
        };
        Self::new(context, pattern.repeat(count))
    }

    fn reset(&self) -> Result<()> {
        let mut bytes = vec![0xa5; GUARD];
        bytes.extend_from_slice(&self.initial);
        bytes.extend_from_slice(&[0x5a; GUARD]);
        self.buffer.upload(&bytes)
    }

    fn pointer(&self) -> u64 {
        self.buffer.pointer() + GUARD as u64
    }

    fn read(&self) -> Result<Snapshot> {
        let mut bytes = vec![0; self.buffer.len()];
        self.buffer.download(&mut bytes)?;
        let end = GUARD + self.initial.len();
        let guards =
            bytes[..GUARD].iter().all(|&b| b == 0xa5) && bytes[end..].iter().all(|&b| b == 0x5a);
        Ok(Snapshot {
            payload: bytes[GUARD..end].to_vec(),
            guards,
        })
    }

    fn unchanged(&self) -> Result<bool> {
        let snapshot = self.read()?;
        Ok(snapshot.guards && snapshot.payload == self.initial)
    }
}

struct Snapshot {
    payload: Vec<u8>,
    guards: bool,
}
impl Snapshot {
    fn words(&self) -> Vec<u16> {
        self.payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect()
    }
    fn floats(&self) -> Vec<f32> {
        self.payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect()
    }
}

fn words_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn floats_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn ids_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn launch(
    context: &Context,
    function: &Function<'_, '_>,
    pointers: &mut [u64],
    dimensions: &mut [u32],
    mut epsilon: Option<f32>,
    blocks: usize,
) -> Result<()> {
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|d| (d as *mut u32).cast()));
    if let Some(value) = epsilon.as_mut() {
        args.push((value as *mut f32).cast());
    }
    let grid = [u32::try_from(blocks)?, 1, 1];
    // SAFETY: The bounded call sites below construct each symbol's exact ABI.
    // Guarded interiors have complete oracle-validated extents and alignment;
    // all buffers outlive synchronization, including after a failed launch.
    let launched = unsafe { function.launch(grid, [256, 1, 1], 0, &mut args) };
    let drained = context.synchronize();
    launched?;
    drained
}

struct EmbeddingFixture {
    codes: Vec<u8>,
    scales: Vec<u16>,
    tokens: Vec<u32>,
    norm: Vec<u16>,
    width: usize,
}
fn embedding_fixture(width: usize) -> EmbeddingFixture {
    let mut codes: Vec<_> = (0..8 * width)
        .map(|i| {
            let code = ((i * 17 + i / width * 11) % 256) as u8;
            if code & 0x7f == 0x7f { code ^ 1 } else { code }
        })
        .collect();
    for row in 0..8 {
        for (column, code) in [0x3c, 0xbc, 0x80, 0, 0x38, 0xb8, 1]
            .into_iter()
            .enumerate()
            .take(width)
        {
            codes[row * width + column] = code;
        }
    }
    EmbeddingFixture {
        codes,
        scales: vec![0x3f81, 0x3f83, 0x3f80, 0x4000, 0x3f00, 1, 0x3f80, 0x3f81],
        tokens: vec![7, 0, 1, 7, 2, 5],
        norm: (0..width)
            .map(|i| round_bf16([0.0, -1.0, 0.5, -0.5][i % 4]))
            .collect(),
        width,
    }
}

fn embedding_case(
    context: &Context,
    gather: &Function<'_, '_>,
    norm: &Function<'_, '_>,
    width: usize,
) -> Result<Value> {
    let f = embedding_fixture(width);
    let expected = embedding::gather(&f.codes, &f.scales, &f.tokens, f.scales.len(), f.width)?;
    let ids: Vec<_> = (0..f.tokens.len() as u32).collect();
    let norm_expected = crate::entry_reference::embedding_norm(
        &words_bytes(&expected),
        &ids,
        &words_bytes(&f.norm),
        width,
        EPSILON,
    )?;
    let inputs = [
        Guarded::new(context, f.codes)?,
        Guarded::new(context, words_bytes(&f.scales))?,
        Guarded::new(context, ids_bytes(&f.tokens))?,
        Guarded::new(context, ids_bytes(&ids))?,
        Guarded::new(context, words_bytes(&f.norm))?,
    ];
    let output = Guarded::poison(context, expected.len(), true)?;
    let mut pointers = [
        inputs[0].pointer(),
        inputs[1].pointer(),
        inputs[2].pointer(),
        output.pointer(),
    ];
    launch(
        context,
        gather,
        &mut pointers,
        &mut [width as u32],
        None,
        ids.len(),
    )?;
    let first = output.read()?;
    let exact = first.words() == expected;
    let norm_report = norm_case(
        context,
        norm,
        &output,
        &inputs[3],
        &inputs[4],
        &norm_expected,
        [ids.len(), width],
    )?;
    // Repeat into the same allocation, freshly poisoned. No timing is collected.
    output.reset()?;
    launch(
        context,
        gather,
        &mut pointers,
        &mut [width as u32],
        None,
        ids.len(),
    )?;
    let repeat = output.read()?;
    let inputs_unchanged = inputs
        .iter()
        .map(Guarded::unchanged)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .all(|v| v);
    let deterministic = repeat.payload == first.payload;
    let guards = first.guards && repeat.guards;
    Ok(json!({
        "kind": "embedding", "width": width, "rows": ids.len(), "vocabulary": f.scales.len(),
        "all_passed": exact && deterministic && guards && inputs_unchanged && norm_report["all_passed"] == true,
        "gather_exact_bf16": exact, "elements": expected.len(), "guards_intact": guards,
        "inputs_unchanged": inputs_unchanged, "repeat_bit_identical": deterministic, "normalization": norm_report,
        "fixtures": "endpoint/repeated tokens, +/- zero, +/- halfway ties, finite E4M3 extremes and BF16 subnormal row scale",
    }))
}

fn norm_case(
    context: &Context,
    norm: &Function<'_, '_>,
    table: &Guarded<'_>,
    ids: &Guarded<'_>,
    weights: &Guarded<'_>,
    expected: &crate::entry_reference::EntryReference,
    [rows, width]: [usize; 2],
) -> Result<Value> {
    let count = rows * width;
    let disjoint = [
        Guarded::poison(context, count, true)?,
        Guarded::poison(context, count, true)?,
        Guarded::poison(context, count, false)?,
    ];
    let aliased = [
        Guarded::poison(context, count, true)?,
        Guarded::poison(context, count, false)?,
    ];
    launch(
        context,
        norm,
        &mut [
            table.pointer(),
            ids.pointer(),
            weights.pointer(),
            disjoint[0].pointer(),
            disjoint[1].pointer(),
            disjoint[2].pointer(),
        ],
        &mut [width as u32],
        Some(EPSILON),
        rows,
    )?;
    let baseline = disjoint
        .iter()
        .map(Guarded::read)
        .collect::<Result<Vec<_>>>()?;
    let before_alias = table.read()?;
    // This is the exact stream entry alias: table==residual, identity row IDs only.
    launch(
        context,
        norm,
        &mut [
            table.pointer(),
            ids.pointer(),
            weights.pointer(),
            table.pointer(),
            aliased[0].pointer(),
            aliased[1].pointer(),
        ],
        &mut [width as u32],
        Some(EPSILON),
        rows,
    )?;
    let actual = aliased
        .iter()
        .map(Guarded::read)
        .collect::<Result<Vec<_>>>()?;
    let after_alias = table.read()?;
    let residual_exact =
        baseline[0].words() == expected.residual && after_alias.words() == expected.residual;
    let alias_exact = baseline[1].payload == actual[0].payload
        && baseline[2].payload == actual[1].payload
        && before_alias.payload == after_alias.payload;
    let guards = baseline.iter().chain(&actual).all(|v| v.guards)
        && before_alias.guards
        && after_alias.guards;
    let raw = baseline[2].floats();
    let values = baseline[1].words();
    let raw_report = compare_floats(&raw, &expected.unrounded, 2e-6, 2e-6);
    let rounded_report = compare_words(&values, &expected.normalized, 1);
    let rounding_exact = values
        .iter()
        .zip(&raw)
        .all(|(&bits, &value)| bits == round_bf16(value));
    Ok(json!({
        "all_passed": guards && alias_exact && residual_exact && rounding_exact
            && raw_report["passed"] == true && rounded_report["passed"] == true,
        "guards_intact": guards, "identity_alias_bit_identical": alias_exact,
        "residual_exact": residual_exact, "rounding_matches_raw": rounding_exact,
        "raw": raw_report, "bf16": rounded_report,
    }))
}

struct GateFixture {
    rows: usize,
    heads: usize,
    low_log: bool,
    low_bias: bool,
    a: Vec<u16>,
    b: Vec<u16>,
    logs: Vec<f32>,
    biases: Vec<f32>,
}
fn gate_fixture(rows: usize, low_log: bool, low_bias: bool) -> GateFixture {
    let heads = 48;
    GateFixture {
        rows,
        heads,
        low_log,
        low_bias,
        a: (0..rows * heads)
            .map(|i| round_bf16([-93.0, -88.0, -20.0, 0.0, 20.0, 21.0, 1.0, 2.0][i % 8]))
            .collect(),
        b: (0..rows * heads)
            .map(|i| round_bf16([-93.0, -88.0, -20.0, -2.0, 0.0, 2.0, 20.0, 80.0][i % 8]))
            .collect(),
        logs: (0..heads)
            .map(|i| [-1.5, 0.5, 1.0][i % 3] + if low_log { 0.0001 } else { 0.0 })
            .collect(),
        biases: (0..heads)
            .map(|i| [0.0, 0.5, -0.5][i % 3] + if low_bias { 0.0001 } else { 0.0 })
            .collect(),
    }
}

struct GateOutput<'ctx> {
    buffers: [Guarded<'ctx>; 3],
}
impl<'ctx> GateOutput<'ctx> {
    fn new(context: &'ctx Context, count: usize) -> Result<Self> {
        Ok(Self {
            buffers: [
                Guarded::poison(context, count, true)?,
                Guarded::poison(context, count, false)?,
                Guarded::poison(context, count, false)?,
            ],
        })
    }
    fn launch(
        &self,
        context: &Context,
        function: &Function<'_, '_>,
        inputs: [u64; 4],
        rows: usize,
        heads: usize,
    ) -> Result<()> {
        let mut pointers = inputs.to_vec();
        pointers.extend(self.buffers.iter().map(Guarded::pointer));
        launch(
            context,
            function,
            &mut pointers,
            &mut [rows as u32, heads as u32],
            None,
            (rows * heads).div_ceil(256),
        )
    }
    fn read(&self) -> Result<Vec<Snapshot>> {
        self.buffers.iter().map(Guarded::read).collect()
    }
    fn reset(&self) -> Result<()> {
        for buffer in &self.buffers {
            buffer.reset()?;
        }
        Ok(())
    }
}

fn gate_case(
    context: &Context,
    candidate: &Function<'_, '_>,
    baseline: &Function<'_, '_>,
    f: GateFixture,
) -> Result<Value> {
    let expected = gates::run(&f.a, &f.b, &f.logs, &f.biases, f.rows, f.heads)?;
    let log_words: Vec<_> = f.logs.iter().copied().map(round_bf16).collect();
    let bias_words: Vec<_> = f.biases.iter().copied().map(round_bf16).collect();
    let narrowed_logs: Vec<_> = log_words.iter().copied().map(bf16_to_f32).collect();
    let narrowed_biases: Vec<_> = bias_words.iter().copied().map(bf16_to_f32).collect();
    let expected_legacy = gates::run(
        &f.a,
        &f.b,
        &narrowed_logs,
        &narrowed_biases,
        f.rows,
        f.heads,
    )?;
    let inputs = [
        Guarded::new(context, words_bytes(&f.a))?,
        Guarded::new(context, words_bytes(&f.b))?,
        Guarded::new(context, floats_bytes(&f.logs))?,
        Guarded::new(context, floats_bytes(&f.biases))?,
        Guarded::new(context, words_bytes(&log_words))?,
        Guarded::new(context, words_bytes(&bias_words))?,
    ];
    let output = GateOutput::new(context, f.a.len())?;
    let control = GateOutput::new(context, f.a.len())?;
    let new_ptrs = [
        inputs[0].pointer(),
        inputs[1].pointer(),
        inputs[2].pointer(),
        inputs[3].pointer(),
    ];
    let old_ptrs = [
        inputs[0].pointer(),
        inputs[1].pointer(),
        inputs[4].pointer(),
        inputs[5].pointer(),
    ];
    output.launch(context, candidate, new_ptrs, f.rows, f.heads)?;
    control.launch(context, baseline, old_ptrs, f.rows, f.heads)?;
    let first = output.read()?;
    let old = control.read()?;
    let candidate_report = gate_metrics(&first, &expected);
    let baseline_report = gate_metrics(&old, &expected_legacy);
    let low_bits = f.low_log || f.low_bias;
    let exact_legacy = first.iter().zip(&old).all(|(a, b)| a.payload == b.payload);
    let beta_unchanged = first[0].payload == old[0].payload;
    let gate_differences = bit_differences(&first[1].payload, &old[1].payload);
    let decay_differences = bit_differences(&first[2].payload, &old[2].payload);
    let oracle_differences = expected
        .g
        .iter()
        .zip(&expected_legacy.g)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let precision_pass = if low_bits {
        gate_differences > 0 && decay_differences > 0 && oracle_differences > 0 && beta_unchanged
    } else {
        exact_legacy
    };
    output.reset()?;
    output.launch(context, candidate, new_ptrs, f.rows, f.heads)?;
    let repeat = output.read()?;
    let deterministic = first
        .iter()
        .zip(&repeat)
        .all(|(a, b)| a.payload == b.payload);
    let guards = first.iter().chain(&old).chain(&repeat).all(|v| v.guards);
    let inputs_unchanged = inputs
        .iter()
        .map(Guarded::unchanged)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .all(|v| v);
    Ok(json!({
        "kind": "gates", "rows": f.rows, "heads": f.heads,
        "low_f32_bits": {"a_log": f.low_log, "dt_bias": f.low_bias},
        "all_passed": precision_pass && deterministic && guards && inputs_unchanged
            && candidate_report["passed"] == true && baseline_report["passed"] == true,
        "candidate": candidate_report, "legacy_bf16": baseline_report,
        "representable_parameters_require_bit_identity": !low_bits,
        "all_outputs_bit_identical_to_legacy": exact_legacy,
        "beta_bit_identical_to_legacy": beta_unchanged,
        "gate_bits_different_from_narrowed": gate_differences,
        "decay_bits_different_from_narrowed": decay_differences,
        "oracle_gate_bits_different_from_narrowed": oracle_differences,
        "precision_check_passed": precision_pass,
        "guards_intact": guards, "inputs_unchanged": inputs_unchanged,
        "repeat_bit_identical": deterministic,
    }))
}

fn bit_differences(left: &[u8], right: &[u8]) -> usize {
    left.as_chunks::<4>()
        .0
        .iter()
        .zip(right.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count()
}

fn gate_metrics(actual: &[Snapshot], expected: &gates::Gates) -> Value {
    let beta = actual[0].words();
    let g = actual[1].floats();
    let decay = actual[2].floats();
    let beta_report = compare_words(&beta, &expected.beta, 1);
    let g_report = compare_floats(&g, &expected.g, 3e-6, 5e-6);
    let decay_report = compare_floats(&decay, &expected.decay, 3e-6, 5e-6);
    let domain = beta.iter().all(|&v| (0.0..=1.0).contains(&bf16_to_f32(v)))
        && g.iter().all(|&v| v.is_finite() && v <= 0.0)
        && decay.iter().all(|v| (0.0..=1.0).contains(v));
    json!({"passed": domain && beta_report["passed"] == true
        && g_report["passed"] == true && decay_report["passed"] == true,
        "domain_valid": domain, "beta": beta_report, "g": g_report, "decay": decay_report})
}

fn ordered_bf16(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}
fn compare_words(actual: &[u16], expected: &[u16], allowed: u16) -> Value {
    let mut maximum = 0;
    let mut mismatches = 0;
    let mut nonfinite = 0;
    for (&a, &b) in actual.iter().zip(expected) {
        let ulp = ordered_bf16(a).abs_diff(ordered_bf16(b));
        maximum = maximum.max(ulp);
        nonfinite += usize::from(!bf16_to_f32(a).is_finite());
        mismatches += usize::from(ulp > allowed);
    }
    json!({"passed": actual.len() == expected.len() && mismatches == 0 && nonfinite == 0,
        "elements": expected.len(), "nonfinite": nonfinite, "mismatches": mismatches,
        "maximum_bf16_ulp": maximum, "allowed_bf16_ulp": allowed})
}
fn compare_floats(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> Value {
    let mut maximum = 0.0_f32;
    let mut mismatches = 0;
    let mut nonfinite = 0;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            nonfinite += 1;
            continue;
        }
        let error = (a - b).abs();
        maximum = maximum.max(error);
        mismatches += usize::from(error > absolute + relative * b.abs());
    }
    json!({"passed": actual.len() == expected.len() && mismatches == 0 && nonfinite == 0,
        "elements": expected.len(), "nonfinite": nonfinite, "mismatches": mismatches,
        "maximum_absolute_error": maximum, "absolute_tolerance": absolute, "relative_tolerance": relative})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_fixtures_cover_ties_sign_and_subnormal_rows() {
        let f = embedding_fixture(7);
        let values = embedding::gather(&f.codes, &f.scales, &f.tokens, 8, 7).unwrap();
        assert_eq!(&values[..4], &[0x3fc2, 0xbfc2, 0x8000, 0]);
        assert_eq!(&values[14..18], &[0x3fc4, 0xbfc4, 0x8000, 0]);
        assert_eq!(values[35], 2);
        assert_eq!(values[37], 0x8000);
    }

    #[test]
    fn low_parameter_bits_are_observable_independently() {
        for (low_log, low_bias) in [(true, false), (false, true), (true, true)] {
            let f = gate_fixture(7, low_log, low_bias);
            let expected = gates::run(&f.a, &f.b, &f.logs, &f.biases, f.rows, f.heads).unwrap();
            let logs: Vec<_> = f.logs.iter().map(|&v| bf16_to_f32(round_bf16(v))).collect();
            let biases: Vec<_> = f
                .biases
                .iter()
                .map(|&v| bf16_to_f32(round_bf16(v)))
                .collect();
            let narrowed = gates::run(&f.a, &f.b, &logs, &biases, f.rows, f.heads).unwrap();
            assert_eq!(expected.beta, narrowed.beta);
            assert!(
                expected
                    .g
                    .iter()
                    .zip(&narrowed.g)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            );
            assert!(
                expected
                    .decay
                    .iter()
                    .zip(&narrowed.decay)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            );
        }
    }

    #[test]
    fn poison_and_missing_output_fail_comparison() {
        assert_eq!(compare_words(&[0x7fc1], &[0], 1)["passed"], false);
        assert_eq!(compare_words(&[], &[0], 1)["passed"], false);
        assert_eq!(
            compare_floats(&[f32::NAN], &[0.0], 1e-6, 1e-6)["passed"],
            false
        );
        assert_eq!(compare_floats(&[], &[0.0], 1e-6, 1e-6)["passed"], false);
        assert_eq!(
            compare_words(&[0xbf80, 0x8000], &[0xbf80, 0x8000], 0)["passed"],
            true
        );
    }
}
