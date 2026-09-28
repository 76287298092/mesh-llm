//! Fixed-length raw-token timing for the resident decoder model.

use super::{
    driver::{Context, Module},
    resident_model::{Model, SelectedOutput, Session},
    resident_weights::ResidentWeights,
    stream_forward::bench::{DecodeRunner, Execution, Runner},
};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    engine::layout::Layout,
    kernels::{DecoderConfig, ModelBenchRequest},
};
use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;

const MEMORY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const REQUIRED_KERNELS: [&str; 22] = [
    "embedding_norm_bf16",
    "fp8_quantize_bf16",
    "fp8_linear_exact",
    "fp8_linear_exact4",
    "fp8_prefill_exact",
    "fp8_verify_exact",
    "bf16_linear_decode",
    "causal_conv4_bf16",
    "gdn_qk_norm",
    "gdn_gates",
    "gdn_recurrent",
    "gdn_gated_rms_norm",
    "residual_norm_bf16",
    "nvfp4_quantize_bf16",
    "nvfp4_linear",
    "nvfp4_decode_exact",
    "mlp_silu_product",
    "residual_add_bf16",
    "attention_qk_prepare",
    "attention_kv_append",
    "causal_attention_bf16",
    "attention_gate_bf16",
];

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut ModelArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    request: &ModelBenchRequest<'_>,
) -> Result<Value> {
    ensure!(
        !super::nvfp4_projection_audit::enabled()?,
        "NVFP4 audit is diagnostic; use qwen-model-profile"
    );
    ensure!(
        !crate::kernels::attention_profile::current()?.is_audit(),
        "attention audit is diagnostic; use qwen-model-profile"
    );
    ensure!(
        !crate::kernels::fp8_profile::current()?.is_audit(),
        "projection audit is diagnostic; use qwen-model-profile rather than a throughput benchmark"
    );
    let final_cursor = validate_request(
        request.tokens,
        config.vocabulary,
        config.capacity,
        config.layers.len(),
        request.output_tokens,
        request.repetitions,
    )?;
    ensure!(
        ptx.contains(".target sm_120a"),
        "model bench requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "model bench requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    validate_kernels(&module)?;

    let layout = Layout::new(
        objects
            .iter()
            .map(|object| (object.name.clone(), object.length)),
    )?;
    let memory_before = context.memory()?;
    let mut memory = MemoryTracker::new();
    memory.record_values("before_model_allocations", None, None, memory_before);
    validate_admission(&layout, config, memory_before.0)?;

    let weights = ResidentWeights::load(&context, artifact, objects)?;
    memory.record(&context, "after_weights_loaded", None, None)?;
    let model = Model::new(&weights, config)?;
    let execution = Execution::current()?;
    let mut runner = Runner::new(
        execution,
        &model,
        &weights,
        &module,
        config,
        request.tokens.len(),
    )?;
    let warmup = run_warmup(
        &context,
        &module,
        &mut runner,
        config,
        request.tokens,
        &mut memory,
    )?;

    let mut repetitions = Vec::with_capacity(request.repetitions);
    for index in 0..request.repetitions {
        repetitions.push(run_repetition(
            &context,
            &module,
            &mut runner,
            config,
            request,
            index + 1,
            &mut memory,
        )?);
    }

    let execution_report = runner.report();
    drop(runner);
    drop(model);
    drop(weights);
    context.synchronize()?;
    let memory_after_release = memory.record(&context, "after_model_release", None, None)?;
    let after_release_free = memory_after_release["free_bytes"]
        .as_u64()
        .context("memory snapshot omitted free_bytes")?;
    let persistent_allocation_bytes = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .context("persistent allocation byte count overflows u64")?;

    Ok(json!({
        "schema_version": 1,
        "kind": "resident-model-fixed-token-benchmark",
        "completed": true,
        "device": info,
        "arithmetic_profile": crate::kernels::fp8_profile::current()?.name(),
        "attention_profile":crate::kernels::attention_profile::current()?.name(),
        "nvfp4_profile":crate::kernels::nvfp4_profile::current()?.name(),
        "execution": execution_report,
        "gpu_greedy": execution != Execution::Legacy || super::model_greedy::enabled()?,
        "configured_gpu_greedy": super::model_greedy::enabled()?, "mlp_workspace":super::model_workspace::enabled()?, "fp8_split_k":super::resident_fp8_splitk::configured_splits()?,
        "configured_capacity": config.capacity,
        "prompt_token_ids": request.tokens,
        "prompt_tokens": request.tokens.len(),
        "fixed_output_tokens": request.output_tokens,
        "repetition_count": request.repetitions,
        "warmup": warmup,
        "warmup_protocol_change": "All executions now warm the full requested prompt plus one actual decode on a disposable fresh session; graph mode captures and replays during warmup. Former protocol warmed one token without decode. Timed repetitions still use fresh sessions and unchanged requested output sequences.",
        "repetitions": repetitions,
        "allocation_bytes": {
            "weight_arena_payload": layout.bytes,
            "state_arena_payload": config.state_layout.bytes,
            "persistent_arena_payload": persistent_allocation_bytes,
        },
        "memory": {
            "samples": memory.samples,
            "minimum_sampled_free_bytes": memory.minimum_free_bytes,
            "after_release_free_bytes": after_release_free,
            "arena_memory_release_observed": after_release_free >= memory_before.0 as u64,
            "transient_peak_measured": false,
        },
        "profile": {
            "kind": "fixed-length raw-token engineering measurement",
            "eos_termination_ignored": true,
            "prefill_includes_final_logits_and_first_greedy_token": true,
            "timing_excludes": ["weight loading", "session allocation", "memory sampling", "JIT warmup", "graph capture and instantiation (separately reported per repetition)"],
            "qualification_claim": false,
            "serving_claim": false,
            "memory_note": "CUDA memory values are checkpoints, not a transient peak measurement.",
        },
        "requested_final_cursor_past": final_cursor,
    }))
}

fn validate_request(
    tokens: &[u32],
    vocabulary: usize,
    capacity: usize,
    layers: usize,
    output_tokens: usize,
    repetitions: usize,
) -> Result<usize> {
    ensure!(
        layers == 64,
        "model bench requires exactly 64 decoder layers"
    );
    ensure!(
        (1..=2048).contains(&capacity),
        "model bench capacity must be in 1..=2048"
    );
    ensure!(
        (1..=512).contains(&tokens.len()),
        "model bench prompt must contain 1..=512 tokens"
    );
    ensure!(
        (2..=512).contains(&output_tokens),
        "model bench output length must be in 2..=512 tokens"
    );
    ensure!(
        (1..=3).contains(&repetitions),
        "model bench repetitions must be in 1..=3"
    );
    ensure!(
        tokens.iter().all(|&token| (token as usize) < vocabulary),
        "model bench prompt contains an ID outside the vocabulary"
    );
    let final_cursor = tokens
        .len()
        .checked_add(output_tokens - 1)
        .context("model bench prompt plus generated-token cursor overflows usize")?;
    ensure!(
        final_cursor <= capacity,
        "model bench prompt and fixed output exceed configured capacity"
    );
    Ok(final_cursor)
}

fn validate_admission(layout: &Layout, config: &DecoderConfig, free_bytes: usize) -> Result<()> {
    let needed = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .and_then(|bytes| bytes.checked_add(MEMORY_RESERVE_BYTES))
        .context("model bench admission byte count overflows u64")?;
    ensure!(
        needed <= u64::try_from(free_bytes)?,
        "insufficient CUDA free memory for weights, state, and 1 GiB workspace reserve"
    );
    Ok(())
}

fn validate_kernels(module: &Module<'_>) -> Result<()> {
    for name in REQUIRED_KERNELS {
        module
            .function(name)
            .with_context(|| format!("load model bench kernel {name}"))?;
    }
    Ok(())
}

struct MemoryTracker {
    samples: Vec<Value>,
    minimum_free_bytes: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            minimum_free_bytes: usize::MAX,
        }
    }

    fn record(
        &mut self,
        context: &Context,
        stage: &str,
        repetition: Option<usize>,
        step: Option<usize>,
    ) -> Result<Value> {
        let memory = context.memory()?;
        Ok(self.record_values(stage, repetition, step, memory))
    }

    fn record_values(
        &mut self,
        stage: &str,
        repetition: Option<usize>,
        step: Option<usize>,
        (free_bytes, total_bytes): (usize, usize),
    ) -> Value {
        self.minimum_free_bytes = self.minimum_free_bytes.min(free_bytes);
        let sample = json!({
            "stage": stage,
            "repetition": repetition,
            "step": step,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "device_used_bytes_snapshot": total_bytes.saturating_sub(free_bytes),
        });
        self.samples.push(sample.clone());
        sample
    }
}

#[derive(Serialize)]
struct Warmup {
    protocol: &'static str,
    input_token_ids: Vec<u32>,
    prefill_output_token_id: u32,
    output_token_id: u32,
    final_cursor_past: usize,
    decode_steps: usize,
    graph: Option<Value>,
}

fn run_warmup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    model: &mut Runner<'_, '_, '_, 'ctx>,
    config: &DecoderConfig,
    tokens: &[u32],
    memory: &mut MemoryTracker,
) -> Result<Warmup> {
    let mut session = Session::new(context, config)?;
    context.synchronize()?;
    memory.record(context, "warmup_session_allocated", None, None)?;
    let (prefill, _) = timed_forward(context, module, model, tokens, &mut session)?;
    memory.record(context, "warmup_prefill_completed", None, None)?;
    let mut decoder = model.prepare_decode(&mut session)?;
    let (output, _) = timed_decode(context, module, &mut decoder, prefill.token)?;
    let graph = decoder.report();
    drop(decoder);
    memory.record(context, "warmup_forward_completed", None, None)?;
    let result = Warmup {
        protocol: "full-requested-prompt-plus-one-actual-decode-v1",
        input_token_ids: tokens.to_vec(),
        prefill_output_token_id: prefill.token,
        output_token_id: output.token,
        final_cursor_past: session.cursor.past(),
        decode_steps: 1,
        graph,
    };
    ensure!(
        result.final_cursor_past == tokens.len() + 1,
        "warmup cursor mismatch"
    );
    drop(session);
    context.synchronize()?;
    memory.record(context, "warmup_session_released", None, None)?;
    Ok(result)
}

fn run_repetition<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    model: &mut Runner<'_, '_, '_, 'ctx>,
    config: &DecoderConfig,
    request: &ModelBenchRequest<'_>,
    repetition: usize,
    memory: &mut MemoryTracker,
) -> Result<Value> {
    let mut session = Session::new(context, config)?;
    context.synchronize()?;
    memory.record(
        context,
        "repetition_session_allocated",
        Some(repetition),
        None,
    )?;

    let sequence_start = Instant::now();
    let (prefill, prefill_seconds) =
        timed_forward(context, module, model, request.tokens, &mut session)?;
    memory.record(
        context,
        "prefill_forward_completed",
        Some(repetition),
        Some(0),
    )?;
    let mut generated = Vec::with_capacity(request.output_tokens);
    let mut previous_token = prefill.token;
    generated.push(previous_token);
    let prefill_past = prefill.past;

    // Graph setup is outside both prefill and every decode interval. The lease
    // binds exactly this fresh session until all replay work and graph handles end.
    let decode_start = Instant::now();
    let mut decoder = model.prepare_decode(&mut session)?;
    let decode_setup_seconds = decode_start.elapsed().as_secs_f64();
    let mut decode_interval_seconds = Vec::with_capacity(request.output_tokens - 1);
    for step in 1..request.output_tokens {
        let (output, seconds) = timed_decode(context, module, &mut decoder, previous_token)?;
        previous_token = output.token;
        generated.push(previous_token);
        decode_interval_seconds.push(seconds);

        memory.record(
            context,
            "decode_forward_completed",
            Some(repetition),
            Some(step),
        )?;
    }

    let decode_wall_seconds = decode_start.elapsed().as_secs_f64();
    let sequence_wall_seconds = sequence_start.elapsed().as_secs_f64();
    let graph_report = decoder.report();
    drop(decoder);
    let final_cursor_past = session.cursor.past();
    let expected_final_cursor = request
        .tokens
        .len()
        .checked_add(request.output_tokens - 1)
        .context("expected model bench cursor overflows usize")?;
    ensure!(
        final_cursor_past == expected_final_cursor,
        "model bench cursor ended at {final_cursor_past}, expected {expected_final_cursor}"
    );
    let decode_seconds = decode_interval_seconds.iter().sum::<f64>();
    let report = json!({
        "repetition": repetition,
        "prefill_input_tokens": request.tokens.len(),
        "prefill_seconds": prefill_seconds,
        "prefill_input_tokens_per_second": rate(request.tokens.len(), prefill_seconds),
        "prefill_cursor_past": prefill_past,
        "decode_intervals": decode_interval_seconds,
        "decode_interval_count": request.output_tokens - 1,
        "decode_total_seconds": decode_seconds,
        "decode_tokens_per_second": rate(request.output_tokens - 1, decode_seconds),
        "decode_setup_seconds": decode_setup_seconds,
        "decode_setup_inclusive_seconds": decode_setup_seconds + decode_seconds,
        "decode_setup_inclusive_tokens_per_second": rate(request.output_tokens - 1, decode_setup_seconds + decode_seconds),
        "decode_wall_seconds_including_setup": decode_wall_seconds,
        "decode_wall_tokens_per_second_including_setup": rate(request.output_tokens - 1, decode_wall_seconds),
        "sequence_wall_seconds": sequence_wall_seconds,
        "sequence_output_tokens_per_second": rate(request.output_tokens, sequence_wall_seconds),
        "wall_scope": "Sequence starts before eager prefill; decode wall starts before graph setup; both end after last decode memory checkpoint. Includes per-session capture/instantiate, host loop and memory sampling, excludes session/weight allocation and final graph destruction. Setup-inclusive interval rate excludes memory sampling, wall rate includes it.",
        "generated_token_ids": generated,
        "final_cursor_past": final_cursor_past,
        "cursor_poisoned": session.cursor.is_poisoned(),
        "graph": graph_report,
    });

    drop(session);
    context.synchronize()?;
    memory.record(
        context,
        "repetition_session_released",
        Some(repetition),
        None,
    )?;
    Ok(report)
}

fn timed_decode(
    context: &Context,
    module: &Module<'_>,
    decoder: &mut DecodeRunner<'_, '_, '_, '_, '_>,
    token: u32,
) -> Result<(SelectedOutput, f64)> {
    let graph = decoder.is_graph();
    if !graph {
        context.synchronize()?;
    }
    let start = Instant::now();
    let forward = decoder.forward(context, module, token);
    // Graph replay already drains its stream on success and failure. Retain the
    // original context boundaries for eager measurements, not for graph replay.
    let synchronization = if graph { Ok(()) } else { context.synchronize() };
    let seconds = start.elapsed().as_secs_f64();
    let output = forward.and_then(|output| synchronization.map(|()| output))?;
    Ok((output, seconds))
}

fn timed_forward(
    context: &Context,
    module: &Module<'_>,
    model: &Runner<'_, '_, '_, '_>,
    tokens: &[u32],
    session: &mut Session<'_>,
) -> Result<(SelectedOutput, f64)> {
    context.synchronize()?;
    let start = Instant::now();
    let forward = model.forward_selected(context, module, tokens, session);
    let synchronization = context.synchronize();
    let seconds = start.elapsed().as_secs_f64();
    let output = match (forward, synchronization) {
        (Ok(output), Ok(())) => output,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(error)) => return Err(error).context("synchronize completed model forward"),
        (Err(error), Err(sync_error)) => {
            return Err(error.context(format!(
                "model forward also failed to synchronize: {sync_error:#}"
            )));
        }
    };
    Ok((output, seconds))
}

fn rate(tokens: usize, seconds: f64) -> Option<f64> {
    (seconds.is_finite() && seconds > 0.0).then(|| tokens as f64 / seconds)
}

#[cfg(test)]
mod tests {
    use super::validate_request;

    #[test]
    fn validates_bounded_fixed_length_requests() {
        assert_eq!(validate_request(&[1], 10, 512, 64, 512, 1).unwrap(), 512);
        assert_eq!(validate_request(&[1, 2], 10, 32, 64, 4, 2).unwrap(), 5);
        assert!(validate_request(&[], 10, 32, 64, 2, 1).is_err());
        assert!(validate_request(&[1; 513], 256, 1024, 64, 2, 1).is_err());
        assert!(validate_request(&[10], 10, 32, 64, 2, 1).is_err());
        assert!(validate_request(&[1], 10, 2, 64, 3, 1).is_err());
        assert!(validate_request(&[1], 10, 32, 63, 2, 1).is_err());
        assert!(validate_request(&[1], 10, 32, 64, 1, 1).is_err());
        assert!(validate_request(&[1], 10, 1024, 64, 513, 1).is_err());
        assert!(validate_request(&[1], 10, 32, 64, 2, 0).is_err());
        assert!(validate_request(&[1], 10, 32, 64, 2, 4).is_err());
    }
}
