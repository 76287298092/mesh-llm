//! Bounded real-weight norm qualification. No source arithmetic or model admission.

#[path = "../../../reference/native_mtp_norm.rs"]
mod reference;
mod comparison;
mod fixtures;
#[cfg(test)]
mod tests;

use super::{
    driver::{Buffer, Context, Module},
    resident_native_mtp::ResidentNativeMtp,
    resident_norm::Norm,
};
use anyhow::{Result, ensure};
use comparison::{compare, diagnostic};
use reference::{Gain, Request, evaluate};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const EPSILON: f32 = 1e-6;

pub(super) fn run(
    context: &Context,
    module: &Module<'_>,
    native: &ResidentNativeMtp<'_>,
) -> Result<Value> {
    ensure!(native.belongs_to(context), "native norm trial context mismatch");
    ensure!(module.belongs_to(context), "native norm trial module context mismatch");
    let norms = &native.views().norms;
    let mut roles = Vec::with_capacity(7);
    for (role, view, width, heads) in [
        ("embedding", &norms.embedding, 5120, 1),
        ("hidden", &norms.hidden, 5120, 1),
        ("final", &norms.final_norm, 5120, 1),
        ("input", &norms.input, 5120, 1),
        ("post_attention", &norms.post_attention, 5120, 1),
        ("query", &norms.query, 256, 24),
        ("key", &norms.key, 256, 4),
    ] {
        let result = (|| -> Result<Value> {
            ensure!(view.elements == width, "native norm role width mismatch");
            let norm = Norm::from_native_mtp(native.norm(view)?, width, EPSILON)?;
            let parent = native.parent(&view.object_id)?;
            ensure!(parent.bytes() == u64::try_from(width * 2)?, "norm parent extent mismatch");
            let mut bytes = vec![0; width * 2];
            parent.read_range(0, &mut bytes)?;
            let resident_hash = hex::encode(Sha256::digest(&bytes));
            let hash_matches = resident_hash == parent.sha256();
            let weights = fixtures::words(&bytes);
            let mut cases = Vec::with_capacity(4);
            for tokens in [1, 5] {
                for fixture in [fixtures::Fixture::ExactSquares, fixtures::Fixture::DenseSigned] {
                    let rows = tokens * heads;
                    let input = fixtures::input(fixture, rows, width)?;
                    let result = run_case(
                        &Execution { context, module, norm: &norm },
                        &Case { input: &input, weights: &weights, rows },
                    );
                    cases.push(match result {
                        Ok(report) => json!({"tokens": tokens, "head_rows": rows,
                            "fixture": fixture, "report": report}),
                        Err(error) => json!({"tokens": tokens, "head_rows": rows,
                            "fixture": fixture, "error": format!("{error:#}"), "passed": false}),
                    });
                }
            }
            let passed = hash_matches && cases.iter().all(|case| case["report"]["passed"] == true);
            Ok(json!({"role": role, "width": width, "object_id": view.object_id,
                "source_parent_sha256": parent.sha256(), "resident_sha256": resident_hash,
                "source_resident_hash_matches": hash_matches, "cases": cases, "passed": passed}))
        })();
        roles.push(match result {
            Ok(report) => report,
            Err(error) => json!({"role": role, "width": width,
                "error": format!("{error:#}"), "passed": false}),
        });
    }
    let passed = roles.iter().all(|role| role["passed"] == true);
    let gain_observed = roles.iter().all(|role| role["cases"].as_array().is_some_and(|cases|
        cases.iter().all(|case| case["report"]["gain_semantics_observed"] == true)));
    Ok(json!({
        "kind": "native-mtp-norm-rust-schedule-qualification", "schema_version": 1,
        "all_passed": passed, "rust_schedule_numerically_qualified": passed,
        "source_gain_semantics": "unit_offset=true: FP32(1+BF16 weight)",
        "offset_gain_observed_in_all_cases": gain_observed,
        "native_weight_words_unchanged": true, "roles": roles, "epsilon": EPSILON,
        "repeat_executions_per_case": 2, "poison_initialization_available": false,
        "poison_runs": 0, "poison_limitation": "Norm::run allocates outputs internally",
        "maximum_readback_bytes": 61440, "ideal_fp64_is_diagnostic_only": true,
        "source_schedule": "BF16 pairs and rsqrtf; launcher unavailable locally",
        "rust_schedule": "256 scalar strided partials/shared tree/sqrt.rn/div.rn",
        "source_arithmetic_qualified": false, "native_mtp_admitted": false,
        "model_executable": false, "timing_claim": false,
    }))
}

struct Execution<'a, 'w, 'ctx> {
    context: &'a Context,
    module: &'a Module<'ctx>,
    norm: &'a Norm<'w, 'ctx>,
}

struct Case<'a> {
    input: &'a [u16],
    weights: &'a [u16],
    rows: usize,
}

fn run_case(execution: &Execution<'_, '_, '_>, case: &Case<'_>) -> Result<Value> {
    let request = Request { input: case.input, weight: case.weights, epsilon: EPSILON };
    let offset = evaluate(&request, Gain::Offset)?;
    let plain = evaluate(&request, Gain::Plain)?;
    let bytes: Vec<u8> = case.input.iter().flat_map(|word| word.to_le_bytes()).collect();
    let input = Buffer::new(execution.context, bytes.len())?;
    input.upload(&bytes)?;
    let mut runs = Vec::with_capacity(2);
    let mut outputs = Vec::with_capacity(2);
    for repeat in 0..2 {
        let result = (|| -> Result<Vec<u16>> {
            let output = execution.norm.run(execution.context, execution.module, &input, case.rows)?;
            let mut bytes = vec![0; case.input.len() * 2];
            output.download(&mut bytes)?;
            Ok(fixtures::words(&bytes))
        })();
        match result {
            Ok(words) => {
                runs.push(json!({"repeat": repeat,
                    "scheduled_offset": compare(&offset.scheduled, &words),
                    "scheduled_plain_gamma": compare(&plain.scheduled, &words),
                    "ideal_fp64": diagnostic(&offset.ideal, &words)}));
                outputs.push(Some(words));
            }
            Err(error) => {
                runs.push(json!({"repeat": repeat, "passed": false,
                    "error": format!("{error:#}")}));
                outputs.push(None);
            }
        }
    }
    let discriminating_words = compare(&offset.scheduled, &plain.scheduled).mismatches;
    let repeat_errors = outputs.iter().filter(|output| output.is_none()).count();
    let repeat = match (&outputs[0], &outputs[1]) {
        (Some(first), Some(second)) => Some(compare(first, second)),
        (Some(_), None) | (None, Some(_)) | (None, None) => None,
    };
    let passed = outputs.iter().all(|output| output.as_ref().is_some_and(|words|
        compare(&offset.scheduled, words).passed()))
        && repeat.as_ref().is_some_and(comparison::Comparison::passed);
    Ok(json!({"passed": passed, "output_words": case.input.len(), "runs": runs,
        "repeat_comparison": repeat,
        "repeat_mismatch_count": repeat.as_ref().map(|report| report.mismatches),
        "repeat_error_count": repeat_errors,
        "gain_semantics_observed": passed && discriminating_words > 0,
        "offset_plain_discriminating_words": discriminating_words}))
}
