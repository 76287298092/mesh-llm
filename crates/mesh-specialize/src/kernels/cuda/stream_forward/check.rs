//! Legacy versus stream forward equivalence on identical inputs and initial state.
//!
//! Both sessions start zeroed. Step 0 prefills the prompt; each later step feeds
//! the legacy greedy token to both executors, so inputs stay identical even after
//! a divergence. Every step records token equality, the SHA-256 of the BF16
//! logits, the SHA-256 of every persistent state region, and wall times. The
//! check passes only when everything is bit-identical.

use super::StreamForward;
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    engine::sampling,
    kernels::{
        DecoderConfig, StreamCheckRequest, attention_profile,
        cuda::{
            driver::{Context, Module},
            resident_model::{Model, Session},
            resident_state::ResidentState,
            resident_weights::ResidentWeights,
        },
        fp8_profile, nvfp4_profile,
    },
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut ModelArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    request: &StreamCheckRequest<'_>,
) -> Result<Value> {
    validate_request(config, request)?;
    if super::bench::Execution::current()? == super::bench::Execution::Graph {
        return super::graph_check::run(ptx, device, artifact, objects, config, request);
    }
    ensure!(
        ptx.contains(".target sm_120a"),
        "stream check requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "stream check requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    // Diagnostic-only: verify transport/layout before either executor uses the
    // weights. This is outside per-step timings and not in the benchmark path.
    let weight_readback = weights.verify()?;
    if weight_readback.iter().any(|item| item["matches"] != true) {
        return Ok(
            json!({"all_passed": false, "kind": "stream-forward-weight-readback-failure",
            "weight_readback": weight_readback}),
        );
    }
    let model = Model::new(&weights, config)?;
    let stream = StreamForward::new(&weights, &module, config, request.tokens.len())?;
    let harness = Harness {
        context: &context,
        module: &module,
        model: &model,
        stream: &stream,
    };
    let mut legacy = Session::new(&context, config)?;
    let mut candidate = Session::new(&context, config)?;
    context.synchronize()?;
    let mut steps = Vec::with_capacity(request.decode_steps + 1);
    let mut timings = Vec::with_capacity(request.decode_steps + 1);
    let mut input = request.tokens.to_vec();
    let mut all_passed = true;
    for index in 0..=request.decode_steps {
        let step = harness.step(index, &input, &mut legacy, &mut candidate)?;
        all_passed &= step.passed;
        timings.push(step.seconds);
        input = vec![step.legacy_token];
        steps.push(step.report);
    }
    Ok(json!({
        "schema_version": 1,
        "kind": "stream-forward-equivalence-check",
        "device": info,
        "arithmetic_profile": fp8_profile::current()?.name(),
        "attention_profile": attention_profile::current()?.name(),
        "nvfp4_profile": nvfp4_profile::current()?.name(),
        "legacy_gpu_greedy": crate::kernels::cuda::model_greedy::enabled()?,
        "stream_forward": stream.report(),
        "weight_readback": weight_readback,
        "weight_readback_scope": "All resident tensor bytes; excluded from forward timings",
        "configured_capacity": config.capacity,
        "prompt_token_ids": request.tokens,
        "prompt_tokens": request.tokens.len(),
        "decode_steps": request.decode_steps,
        "steps": steps,
        "timing": timing_summary(&timings),
        "timing_note": "Wall time of one forward between context synchronizes. Both sides include a full BF16 logit download: legacy for CPU greedy, stream as the diagnostic equivalence readback. Not a throughput benchmark; use qwen-model-bench with MESH_SPECIALIZE_EXECUTION=stream.",
        "all_passed": all_passed,
    }))
}

fn validate_request(config: &DecoderConfig, request: &StreamCheckRequest<'_>) -> Result<()> {
    ensure!(
        (1..=super::program::MAX_ROWS).contains(&request.tokens.len()),
        "stream check prompt must contain 1..={} tokens",
        super::program::MAX_ROWS
    );
    ensure!(
        request.decode_steps <= 512,
        "stream check decode steps must be at most 512"
    );
    ensure!(
        request
            .tokens
            .iter()
            .all(|&token| (token as usize) < config.vocabulary),
        "stream check prompt contains an ID outside the vocabulary"
    );
    let end = request
        .tokens
        .len()
        .checked_add(request.decode_steps)
        .context("stream check cursor overflows usize")?;
    ensure!(
        end <= config.capacity,
        "stream check prompt plus decode steps exceed configured capacity"
    );
    Ok(())
}

struct Harness<'a, 'm, 'w, 'ctx> {
    context: &'ctx Context,
    module: &'m Module<'ctx>,
    model: &'a Model<'w, 'ctx>,
    stream: &'a StreamForward<'m, 'w, 'ctx>,
}

struct StepResult {
    legacy_token: u32,
    passed: bool,
    seconds: [f64; 2],
    report: Value,
}

impl Harness<'_, '_, '_, '_> {
    fn step(
        &self,
        index: usize,
        input: &[u32],
        legacy: &mut Session<'_>,
        candidate: &mut Session<'_>,
    ) -> Result<StepResult> {
        self.context.synchronize()?;
        let start = Instant::now();
        let reference = self
            .model
            .forward(self.context, self.module, input, legacy, None)?;
        self.context.synchronize()?;
        let legacy_seconds = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let output = self.stream.forward(input, candidate, true)?;
        self.context.synchronize()?;
        let stream_seconds = start.elapsed().as_secs_f64();

        let logits = output.logits.context("stream forward omitted logits")?;
        let legacy_logits = logit_sha256(&reference.logits);
        let stream_logits = logit_sha256(&logits);
        let stream_cpu_token = sampling::greedy(&logits)?;
        let legacy_state = region_hashes(&legacy.state)?;
        let stream_state = region_hashes(&candidate.state)?;
        let mismatched: Vec<&str> = legacy_state
            .iter()
            .zip(&stream_state)
            .filter(|(left, right)| left != right)
            .map(|(left, _)| left.0.as_str())
            .collect();
        ensure!(
            legacy_state.len() == stream_state.len(),
            "session state layouts differ"
        );
        let token_equal = reference.token == output.token;
        let passed = token_equal
            && legacy_logits == stream_logits
            && mismatched.is_empty()
            && reference.past == output.past;
        let report = json!({
            "step": index,
            "input_tokens": input.len(),
            "legacy_token": reference.token,
            "stream_token": output.token,
            "stream_cpu_greedy_token": stream_cpu_token,
            "token_equal": token_equal,
            "legacy_past": reference.past,
            "stream_past": output.past,
            "legacy_logits_sha256": legacy_logits,
            "stream_logits_sha256": stream_logits,
            "logits_equal": legacy_logits == stream_logits,
            "state": {
                "kinds_equal": kinds_equal(&mismatched),
                "mismatched_regions": mismatched,
                "legacy_sha256": to_map(&legacy_state),
                "stream_sha256": to_map(&stream_state),
            },
            "legacy_seconds": legacy_seconds,
            "stream_seconds": stream_seconds,
            "passed": passed,
        });
        Ok(StepResult {
            legacy_token: reference.token,
            passed,
            seconds: [legacy_seconds, stream_seconds],
            report,
        })
    }
}

pub(super) fn logit_sha256(logits: &[u16]) -> String {
    let mut hash = Sha256::new();
    for word in logits {
        hash.update(word.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

pub(super) fn region_hashes(state: &ResidentState<'_>) -> Result<Vec<(String, String)>> {
    let mut hashes = Vec::with_capacity(state.layout().regions.len());
    for region in &state.layout().regions {
        let length = usize::try_from(region.length)
            .with_context(|| format!("state region {} does not fit usize", region.name))?;
        let mut bytes = vec![0; length];
        state.read_region(&region.name, &mut bytes)?;
        hashes.push((region.name.clone(), hex::encode(Sha256::digest(&bytes))));
    }
    Ok(hashes)
}

fn region_kind(name: &str) -> &'static str {
    if name.ends_with(".attention.k") || name.ends_with(".attention.v") {
        "kv"
    } else if name.ends_with(".gdn.history") {
        "conv"
    } else if name.ends_with(".gdn.recurrent") {
        "recurrent"
    } else {
        "other"
    }
}

pub(super) fn kinds_equal(mismatched: &[&str]) -> Value {
    let equal = |kind: &str| !mismatched.iter().any(|name| region_kind(name) == kind);
    json!({
        "kv": equal("kv"),
        "conv": equal("conv"),
        "recurrent": equal("recurrent"),
        "other": equal("other"),
    })
}

pub(super) fn to_map(hashes: &[(String, String)]) -> Value {
    Value::Object(
        hashes
            .iter()
            .map(|(name, hash)| (name.clone(), Value::String(hash.clone())))
            .collect::<Map<_, _>>(),
    )
}

fn timing_summary(timings: &[[f64; 2]]) -> Value {
    let mean = |values: &[[f64; 2]], side: usize| {
        (!values.is_empty())
            .then(|| values.iter().map(|pair| pair[side]).sum::<f64>() / values.len() as f64)
    };
    let (prefill, decode) = timings.split_at(timings.len().min(1));
    json!({
        "legacy_prefill_seconds": mean(prefill, 0),
        "stream_prefill_seconds": mean(prefill, 1),
        "legacy_decode_mean_seconds": mean(decode, 0),
        "stream_decode_mean_seconds": mean(decode, 1),
        "decode_steps_timed": decode.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::{kinds_equal, logit_sha256, region_kind, timing_summary};

    #[test]
    fn classifies_state_regions() {
        assert_eq!(region_kind("layers.03.attention.k"), "kv");
        assert_eq!(region_kind("layers.03.attention.v"), "kv");
        assert_eq!(region_kind("layers.00.gdn.history"), "conv");
        assert_eq!(region_kind("layers.00.gdn.recurrent"), "recurrent");
        let kinds = kinds_equal(&["layers.00.gdn.history"]);
        assert_eq!(kinds["conv"], false);
        assert_eq!(kinds["kv"], true);
    }

    #[test]
    fn hashes_logits_as_little_endian_bf16_words() {
        assert_eq!(logit_sha256(&[0x3f80]), logit_sha256(&[0x3f80]));
        assert_ne!(logit_sha256(&[0x3f80]), logit_sha256(&[0x803f]));
    }

    #[test]
    fn splits_prefill_from_decode_timing() {
        let summary = timing_summary(&[[4.0, 2.0], [1.0, 0.5], [3.0, 1.5]]);
        assert_eq!(summary["legacy_prefill_seconds"], 4.0);
        assert_eq!(summary["stream_decode_mean_seconds"], 1.0);
        assert_eq!(summary["decode_steps_timed"], 2);
    }
}
