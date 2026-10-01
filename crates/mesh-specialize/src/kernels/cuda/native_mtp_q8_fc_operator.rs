mod compare;
mod fixture;
pub(in crate::kernels::cuda) mod launch;
mod resident;

#[cfg(test)]
mod tests;

use super::driver::{Context, Module};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

const FC_K: usize = 10_240;
const FC_ROWS: usize = 5_120;
const FC_GROUP: usize = 32;
const FACTOR_BITS: [u16; 5] = [0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x4000];

#[derive(Clone, Copy)]
enum SparseInput {
    GroupSweep,
    LastK,
}

impl SparseInput {
    const fn name(self) -> &'static str {
        match self {
            Self::GroupSweep => "one-nonzero-lane-per-group",
            Self::LastK => "last-k-only",
        }
    }
}

fn sparse_input(pattern: SparseInput, tokens: usize) -> Result<Vec<u16>> {
    let count = tokens
        .checked_mul(FC_K)
        .context("FC activation count overflow")?;
    let mut input = Vec::new();
    input
        .try_reserve_exact(count)
        .context("FC activation allocation failed")?;
    input.resize(count, 0);
    match pattern {
        SparseInput::GroupSweep => {
            for token in 0..tokens {
                for group in 0..FC_K / FC_GROUP {
                    let index = token
                        .checked_mul(FC_K)
                        .and_then(|base| base.checked_add(group * FC_GROUP + group % FC_GROUP))
                        .context("FC group fixture index overflow")?;
                    input[index] = FACTOR_BITS[(group + token) % FACTOR_BITS.len()];
                }
            }
        }
        SparseInput::LastK => {
            for token in 0..tokens {
                let index = token
                    .checked_mul(FC_K)
                    .and_then(|base| base.checked_add(FC_K - 1))
                    .context("FC last-K fixture index overflow")?;
                input[index] = FACTOR_BITS[token % FACTOR_BITS.len()];
            }
        }
    }
    Ok(input)
}

fn compare_sparse(
    expected: &crate::native_mtp_q8_sliced_k_fc_reference::Q8SlicedKReference,
    outputs: &[Vec<u16>; 2],
    tokens: usize,
) -> Result<Value> {
    let count = tokens
        .checked_mul(FC_ROWS)
        .context("FC output count overflow")?;
    ensure!(
        expected.output_bf16.len() == count && outputs.iter().all(|output| output.len() == count),
        "FC output/reference extent mismatch"
    );
    let mut nonfinite = 0;
    let mut repeat_mismatches = 0;
    let mut exact_mismatches = 0;
    let mut failures = Vec::new();
    for (index, (&first, &second)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
        nonfinite += usize::from(first & 0x7f80 == 0x7f80) + usize::from(second & 0x7f80 == 0x7f80);
        repeat_mismatches += usize::from(first != second);
        let oracle = expected.output_bf16[index];
        exact_mismatches += usize::from(first != oracle) + usize::from(second != oracle);
        if (first != oracle || second != oracle) && failures.len() < 16 {
            failures.push(json!({
                "token": index / FC_ROWS, "row": index % FC_ROWS,
                "expected_bf16": oracle, "observed_bf16": [first, second],
            }));
        }
    }
    Ok(json!({
        "all_passed": nonfinite == 0 && repeat_mismatches == 0 && exact_mismatches == 0,
        "output_count_per_repeat": count, "repeats": 2,
        "nonfinite_outputs": nonfinite, "repeat_mismatches": repeat_mismatches,
        "exact_bf16_mismatches": exact_mismatches, "failures_first_16": failures,
        "gate": "all 5120 rows exactly match independent scheduled-FP32 BF16 reference; no tolerance",
    }))
}

fn reference_diagnostics(
    expected: &crate::native_mtp_q8_sliced_k_fc_reference::Q8SlicedKReference,
) -> Value {
    let max_scheduled_f64_delta = expected
        .scheduled_f32
        .iter()
        .zip(&expected.mathematical_f64)
        .map(|(&scheduled, &mathematical)| (f64::from(scheduled) - mathematical).abs())
        .fold(0.0_f64, f64::max);
    let max_error_bound = expected
        .mathematical_error_bound
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    json!({
        "scheduled_f32_vs_mathematical_f64_max_abs_delta": max_scheduled_f64_delta,
        "mathematical_error_bound_max_abs": max_error_bound,
        "bound_is_gpu_tolerance": false,
    })
}

pub(in crate::kernels) fn run(ptx: &str, device: i32) -> Result<Value> {
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        info.major >= 8,
        "Q8 FC requires BF16 MMA and cp.async on SM80+"
    );
    let module = Module::load(&context, ptx)?;
    let mut cases = Vec::new();
    for candidate in [fixture::Candidate::C4, fixture::Candidate::C8] {
        let resources = module.function(candidate.entry())?.resources()?;
        ensure!(
            resources.max_threads_per_block >= 256,
            "Q8 FC block exceeds function limit"
        );
        for kind in [
            fixture::Kind::Dense,
            fixture::Kind::LastK,
            fixture::Kind::Cancellation,
        ] {
            let result = run_case(&context, &module, (candidate, kind));
            let report = match result {
                Ok(report) => report,
                Err(error) => json!({"all_passed": false, "error": format!("{error:#}")}),
            };
            cases.push(json!({
                "entry": candidate.entry(), "tokens": candidate.tokens(),
                "fixture": kind.name(), "kernel_resources": resources, "result": report,
            }));
        }
    }
    Ok(json!({
        "kind": "native-mtp-q8-fc-sliced-k-synthetic-v1",
        "device": info, "jit_log": module.jit_log(),
        "grid": [320, 1, 1], "block": [256, 1, 1], "shape": [5120, 10240],
        "all_passed": cases.iter().all(|case| case["result"]["all_passed"] == true),
        "cases": cases,
        "scope": "synthetic FC candidates only; no real-weight, model, MTP admission or timing claim",
    }))
}

pub(in crate::kernels) fn run_resident(artifact: &Path, ptx: &str, device: i32) -> Result<Value> {
    resident::run(artifact, ptx, device)
}

fn run_case(
    context: &Context,
    module: &Module<'_>,
    case: (fixture::Candidate, fixture::Kind),
) -> Result<Value> {
    let fixture = fixture::Fixture::new(case.0, case.1)?;
    let expected = crate::native_mtp_q8_sliced_k_fc_reference::run(
        &fixture.object,
        &fixture.view,
        &fixture.input,
    )?;
    let outputs = launch::run(context, module, &fixture)?;
    compare::run(&fixture, &expected, &outputs)
}
