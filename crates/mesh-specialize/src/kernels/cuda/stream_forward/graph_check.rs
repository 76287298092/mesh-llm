//! Exact eager stream versus actual full-model graph replay. Selected by
//! MESH_SPECIALIZE_EXECUTION=graph in qwen-stream-check. No residual-only probe.

use super::{
    StreamForward, StreamOutput,
    check::{kinds_equal, logit_sha256, region_hashes, to_map},
    graph_decode::{GraphDecode, ensure_exact},
};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    engine::sampling,
    kernels::{
        DecoderConfig, StreamCheckRequest, attention_profile,
        cuda::{
            driver::{Context, Module},
            resident_model::Session,
            resident_weights::ResidentWeights,
        },
        fp8_profile, nvfp4_profile,
    },
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut ModelArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    request: &StreamCheckRequest<'_>,
) -> Result<Value> {
    ensure_exact(attention_profile::current()?)?;
    ensure!(
        request.decode_steps >= 2,
        "graph check requires at least two actual replay positions"
    );
    ensure!(
        ptx.contains(".target sm_120a"),
        "graph check requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "graph check requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    // Preserve full tensor transport/layout diagnostics, outside all timings.
    let weight_readback = weights.verify()?;
    if weight_readback.iter().any(|item| item["matches"] != true) {
        return Ok(
            json!({"all_passed": false, "kind": "graph-forward-weight-readback-failure",
            "weight_readback": weight_readback}),
        );
    }
    let reference = StreamForward::new(&weights, &module, config, request.tokens.len())?;
    let mut candidate = StreamForward::new(&weights, &module, config, request.tokens.len())?;
    let stream_report = candidate.report();
    let mut left = Session::new(&context, config)?;
    let mut right = Session::new(&context, config)?;
    context.synchronize()?;
    let mut run = compare_sequence(&reference, &mut candidate, &mut left, &mut right, request)?;
    run["device"] = json!(info);
    run["stream_forward"] = stream_report;
    run["weight_readback"] = json!(weight_readback);
    run["weight_readback_scope"] =
        json!("All resident tensor bytes; excluded from all forward timings");
    run["configured_capacity"] = json!(config.capacity);
    Ok(run)
}

fn compare_sequence<'m, 'w, 'ctx>(
    reference: &StreamForward<'m, 'w, 'ctx>,
    candidate: &mut StreamForward<'m, 'w, 'ctx>,
    left: &mut Session<'ctx>,
    right: &mut Session<'ctx>,
    request: &StreamCheckRequest<'_>,
) -> Result<Value> {
    let start = Instant::now();
    let mut expected = reference.forward(request.tokens, left, true)?;
    let eager_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let actual = candidate.forward(request.tokens, right, true)?;
    let candidate_seconds = start.elapsed().as_secs_f64();
    let before = region_hashes(&right.state)?;
    let mut steps = vec![compare(
        0,
        &expected,
        &actual,
        &region_hashes(&left.state)?,
        &before,
        [eager_seconds, candidate_seconds],
    )?];
    let mut graph = GraphDecode::capture(candidate, right)?;
    let capture_check = compare(
        0,
        &actual,
        &graph.snapshot()?,
        &before,
        &region_hashes(graph.state())?,
        [0.0, 0.0],
    )?;
    for index in 1..=request.decode_steps {
        // Both executors consume the reference token, including after divergence.
        let input = expected.token;
        let start = Instant::now();
        expected = reference.forward(&[input], left, true)?;
        let eager_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let actual = graph.replay(input, true)?;
        let graph_seconds = start.elapsed().as_secs_f64();
        let mut comparison = compare(
            index,
            &expected,
            &actual,
            &region_hashes(&left.state)?,
            &region_hashes(graph.state())?,
            [eager_seconds, graph_seconds],
        )?;
        comparison["input_token_id"] = json!(input);
        steps.push(comparison);
    }
    let graph_report = graph.report();
    let replayed = graph_report["successful_replays"].as_u64() == Some(request.decode_steps as u64);
    let all_passed = replayed
        && capture_check["passed"] == true
        && steps.iter().all(|step| step["passed"] == true);
    Ok(json!({
        "schema_version": 1, "kind": "exact-stream-graph-equivalence-check",
        "arithmetic_profile": fp8_profile::current()?.name(),
        "attention_profile": attention_profile::current()?.name(),
        "nvfp4_profile": nvfp4_profile::current()?.name(),
        "gpu_greedy": true, "prompt_token_ids": request.tokens,
        "prompt_tokens": request.tokens.len(), "decode_steps": request.decode_steps,
        "steps": steps, "graph": graph_report, "actual_replay_count_matches": replayed,
        "capture_did_not_execute": capture_check,
        "state_scope": "ALL bytes in ALL state regions, including unused KV suffix, convolution history, recurrence and other regions",
        "timing_note": "Diagnostic wall times include full BF16 logit downloads; capture and instantiation are separate. Not a throughput benchmark.",
        "all_passed": all_passed,
    }))
}

fn compare(
    step: usize,
    expected: &StreamOutput,
    actual: &StreamOutput,
    left: &[(String, String)],
    right: &[(String, String)],
    seconds: [f64; 2],
) -> Result<Value> {
    let expected_logits = expected
        .logits
        .as_ref()
        .context("reference omitted full logits")?;
    let actual_logits = actual
        .logits
        .as_ref()
        .context("graph candidate omitted full logits")?;
    let reference_cpu = sampling::greedy(expected_logits)?;
    let candidate_cpu = sampling::greedy(actual_logits)?;
    let mismatched: Vec<&str> = left
        .iter()
        .filter(|entry| !right.contains(entry))
        .map(|entry| entry.0.as_str())
        .collect();
    let passed = expected_logits == actual_logits
        && left == right
        && expected.token == actual.token
        && expected.past == actual.past
        && reference_cpu == expected.token
        && candidate_cpu == actual.token;
    Ok(json!({
        "step": step, "execution": if step == 0 { "eager-prefill" } else { "graph-replay" },
        "eager_token": expected.token, "candidate_token": actual.token,
        "eager_cpu_token": reference_cpu, "candidate_cpu_token": candidate_cpu,
        "eager_past": expected.past, "candidate_past": actual.past,
        "logits_equal": expected_logits == actual_logits,
        "eager_logits_sha256": logit_sha256(expected_logits),
        "candidate_logits_sha256": logit_sha256(actual_logits),
        "state": {"equal": left == right, "kinds_equal": kinds_equal(&mismatched),
            "mismatched_regions": mismatched, "eager_sha256": to_map(left), "candidate_sha256": to_map(right)},
        "eager_seconds": seconds[0], "candidate_seconds": seconds[1], "passed": passed,
    }))
}

#[cfg(test)]
mod tests {
    use super::compare;
    use crate::kernels::cuda::stream_forward::StreamOutput;

    #[test]
    fn equivalence_requires_full_logits_and_every_region() {
        let left = StreamOutput {
            token: 0,
            past: 3,
            logits: Some(vec![0x4000, 0x3f80]),
        };
        let mut right = StreamOutput {
            token: 0,
            past: 3,
            logits: Some(vec![0x4000, 0x3f80]),
        };
        let hashes = vec![("layer.gdn.history".into(), "hash".into())];
        assert_eq!(
            compare(1, &left, &right, &hashes, &hashes, [0.0; 2]).unwrap()["passed"],
            true
        );
        right.logits.as_mut().unwrap()[1] = 0;
        assert_eq!(
            compare(1, &left, &right, &hashes, &hashes, [0.0; 2]).unwrap()["passed"],
            false
        );
        right.logits = left.logits.clone();
        assert_eq!(
            compare(1, &left, &right, &hashes, &[], [0.0; 2]).unwrap()["passed"],
            false
        );
    }
}
