//! Per-kernel CUDA-event profile for one resident decode after a short prefix.

use super::{
    driver::{Buffer, Context, Module},
    resident_model::{Model, Output, Session},
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::layout::Layout,
    kernels::DecoderConfig,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

const MAX_PREFIX_TOKENS: usize = 512;
const MAX_CAPACITY: usize = 513;
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
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    tokens: &[u32],
) -> Result<Value> {
    validate_request(
        tokens,
        config.vocabulary,
        config.capacity,
        config.layers.len(),
    )?;
    ensure!(
        ptx.contains(".target sm_120a"),
        "profile requires SM120a PTX"
    );
    let layout = Layout::new(
        objects
            .iter()
            .map(|object| (object.name.clone(), object.length)),
    )?;

    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "profile requires selected SM120 device"
    );
    let module = Module::load(&context, ptx)?;
    validate_kernels(&module)?;
    let memory_before = context.memory()?;
    validate_admission(&layout, config, memory_before.0)?;

    let weights = ResidentWeights::load(&context, artifact, objects)?;
    let model = Model::new(&weights, config)?;
    let warmup = run_warmup(&context, &module, &model, config, tokens[0])?;
    let control = run_unprofiled(&context, &module, &model, config, tokens, false)?;
    let partitioned = run_unprofiled(&context, &module, &model, config, tokens, true)?;
    let partition = json!({
        "prefill_logits_bit_exact": control.prefill_logits == partitioned.prefill_logits,
        "whole_prefill_token": control.prefill_token,
        "token_prefill_token": partitioned.prefill_token,
        "decode_inputs_equal":control.prefill_token==partitioned.prefill_token,
        "layer_tail_drift":control.layer_tails.iter().zip(&partitioned.layer_tails).enumerate().map(|(layer,(a,b))|json!({"layer":layer,"drift":logit_drift(a,b)})).collect::<Vec<_>>(),
        "prefill_logit_drift": logit_drift(&control.prefill_logits, &partitioned.prefill_logits),
        "decode_logit_drift": logit_drift(&control.output.logits, &partitioned.output.logits),
        "decode_logits_bit_exact": control.output.logits == partitioned.output.logits,
        "state_bit_exact": control.state_sha256 == partitioned.state_sha256,
        "whole_state_sha256": control.state_sha256,
        "token_state_sha256": partitioned.state_sha256,
        "cursor_exact": control.cursor_past == partitioned.cursor_past,
        "prefill_bf16_differences": control.prefill_logits.iter().zip(&partitioned.prefill_logits).filter(|(a,b)|a!=b).count(),
        "decode_bf16_differences": control.output.logits.iter().zip(&partitioned.output.logits).filter(|(a,b)|a!=b).count(),
    });
    let partition_exact = control.prefill_logits == partitioned.prefill_logits
        && control.output.logits == partitioned.output.logits
        && control.state_sha256 == partitioned.state_sha256
        && control.cursor_past == partitioned.cursor_past;
    drop(partitioned);
    let profiled = run_profiled(&context, &module, &model, config, tokens)?;
    validate_kernel_profile(&profiled.kernel_profile)?;
    validate_kernel_profile(&profiled.prefill_kernel_profile)?;

    let DecodeRun {
        prefill_logits: control_prefill_logits,
        prefill_token: control_prefill_token,
        prefill_past: control_prefill_past,
        output: control_output,
        wall_seconds: unprofiled_seconds,
        state_sha256: control_state_sha256,
        cursor_past: control_cursor_past,
        kernel_profile: _,
        prefill_kernel_profile: _,
        layer_tails: _,
    } = control;
    let DecodeRun {
        prefill_logits: profiled_prefill_logits,
        prefill_token: profiled_prefill_token,
        prefill_past: profiled_prefill_past,
        output: profiled_output,
        wall_seconds: profiled_seconds,
        state_sha256: profiled_state_sha256,
        cursor_past: profiled_cursor_past,
        kernel_profile,
        prefill_kernel_profile,
        layer_tails: _,
    } = profiled;
    let prefill_exact = control_prefill_logits == profiled_prefill_logits
        && control_prefill_token == profiled_prefill_token
        && control_prefill_past == profiled_prefill_past;
    let exact_output_and_state = control_output.logits == profiled_output.logits
        && control_output.token == profiled_output.token
        && control_output.past == profiled_output.past
        && control_state_sha256 == profiled_state_sha256;
    let resulting_past = profiled_output.past;
    let decode_input_token = profiled_prefill_token;
    drop(control_prefill_logits);
    drop(profiled_prefill_logits);
    drop(control_output);
    drop(profiled_output);
    drop(model);
    drop(weights);
    context.synchronize()?;
    let memory_after_free = context.memory()?;
    let memory_released = memory_after_free.0 >= memory_before.0;
    let persistent_bytes = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .context("profile persistent allocation size overflows u64")?;
    let past_exact = resulting_past == tokens.len() + 1
        && control_cursor_past == resulting_past
        && profiled_cursor_past == resulting_past;
    let all_passed =
        prefill_exact && exact_output_and_state && past_exact && memory_released && partition_exact;

    Ok(json!({
        "schema_version": 1,
        "kind": "resident-model-single-decode-kernel-profile",
        "completed": true,
        "all_passed": all_passed,
        "device": info,
        "arithmetic_profile": crate::kernels::fp8_profile::current()?.name(),
        "prefix_token_ids": tokens,
        "decode_input_token": decode_input_token,
        "resulting_past": resulting_past,
        "exact_prefill_logits": prefill_exact,
        "whole_vs_token_partition": partition,
        "whole_vs_token_partition_exact": partition_exact,
        "exact_output_and_state": exact_output_and_state,
        "control_state_sha256": control_state_sha256,
        "profiled_state_sha256": profiled_state_sha256,
        "control_cursor_past": control_cursor_past,
        "profiled_cursor_past": profiled_cursor_past,
        "unprofiled_decode_wall_seconds": unprofiled_seconds,
        "profiled_decode_wall_seconds": profiled_seconds,
        "kernel_profile": kernel_profile,
        "prefill_kernel_profile": prefill_kernel_profile,
        "warmup": warmup,
        "allocation_bytes": {
            "weight_arena": layout.bytes,
            "state_arena": config.state_layout.bytes,
            "persistent_total": persistent_bytes,
        },
        "memory_before": {"free_bytes": memory_before.0, "total_bytes": memory_before.1},
        "memory_after_free": {"free_bytes": memory_after_free.0, "total_bytes": memory_after_free.1},
        "memory_released": memory_released,
        "scope": "Profiled full-prefix execution and one full-decoder token; event instrumentation is not model throughput or an independent quality check.",
    }))
}

struct DecodeRun {
    layer_tails: Vec<Vec<u16>>,
    prefill_logits: Vec<u16>,
    prefill_token: u32,
    prefill_past: usize,
    output: Output,
    wall_seconds: f64,
    state_sha256: String,
    cursor_past: usize,
    kernel_profile: Value,
    prefill_kernel_profile: Value,
}

fn validate_request(
    tokens: &[u32],
    vocabulary: usize,
    capacity: usize,
    layers: usize,
) -> Result<()> {
    ensure!(
        (1..=MAX_PREFIX_TOKENS).contains(&tokens.len()),
        "profile prefix must contain 1..={MAX_PREFIX_TOKENS} tokens"
    );
    ensure!(
        tokens.iter().all(|&token| (token as usize) < vocabulary),
        "profile prefix contains a token outside the vocabulary"
    );
    let expected_capacity = tokens
        .len()
        .checked_add(1)
        .context("profile capacity overflows usize")?;
    ensure!(
        capacity == expected_capacity && capacity <= MAX_CAPACITY,
        "profile capacity must equal prefix length plus one and be at most {MAX_CAPACITY}"
    );
    ensure!(layers == 64, "profile requires exactly 64 decoder layers");
    Ok(())
}

fn validate_kernels(module: &Module<'_>) -> Result<()> {
    for name in REQUIRED_KERNELS {
        module
            .function(name)
            .with_context(|| format!("load profile kernel {name}"))?;
    }
    Ok(())
}

fn validate_admission(layout: &Layout, config: &DecoderConfig, free_bytes: usize) -> Result<()> {
    let required = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .and_then(|bytes| bytes.checked_add(MEMORY_RESERVE_BYTES))
        .context("profile admission size overflows u64")?;
    ensure!(
        required <= u64::try_from(free_bytes)?,
        "insufficient CUDA memory for weights, state, and 1 GiB reserve"
    );
    Ok(())
}

fn run_warmup(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    config: &DecoderConfig,
    token: u32,
) -> Result<Value> {
    let mut session = Session::new(context, config)?;
    context.synchronize()?;
    let output = model.forward(context, module, &[token], &mut session, None)?;
    context.synchronize()?;
    ensure!(
        output.past == 1 && session.cursor.past() == 1 && !session.cursor.is_poisoned(),
        "profile warmup did not commit one clean token"
    );
    let report = json!({
        "input_token": token,
        "output_token": output.token,
        "past": output.past,
        "timed": false,
    });
    drop(output);
    drop(session);
    context.synchronize()?;
    Ok(report)
}

fn run_unprofiled(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    config: &DecoderConfig,
    tokens: &[u32],
    token_prefill: bool,
) -> Result<DecodeRun> {
    let mut session = Session::new(context, config)?;
    let mut layer_tails = Vec::new();
    let mut observer = |_layer: usize, buffer: &Buffer<'_>| -> Result<()> {
        let bytes = config
            .hidden
            .checked_mul(2)
            .context("layer tail size overflow")?;
        ensure!(buffer.len() >= bytes, "layer tail buffer too short");
        let mut tail = vec![0; bytes];
        buffer.download_at(buffer.len() - bytes, &mut tail)?;
        layer_tails.push(
            tail.as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect(),
        );
        Ok(())
    };
    let (prefill_logits, prefill_token, prefill_past) = if token_prefill {
        let mut last = None;
        for (index, &token) in tokens.iter().enumerate() {
            let watch = if index + 1 == tokens.len() {
                Some(&mut observer as &mut super::resident_model::Observer<'_>)
            } else {
                None
            };
            last = Some(model.forward(context, module, &[token], &mut session, watch)?);
        }
        let output = last.context("empty profile prefix")?;
        check_committed(&session, &output, tokens.len())?;
        (output.logits, output.token, output.past)
    } else {
        let output = model.forward(context, module, tokens, &mut session, Some(&mut observer))?;
        check_committed(&session, &output, tokens.len())?;
        (output.logits, output.token, output.past)
    };
    let (output, wall_seconds) =
        timed_forward(context, module, model, prefill_token, &mut session)?;
    check_committed(&session, &output, tokens.len() + 1)?;
    let state_sha256 = state_hash(&session)?;
    let cursor_past = session.cursor.past();
    drop(session);
    context.synchronize()?;
    Ok(DecodeRun {
        layer_tails,
        prefill_logits,
        prefill_token,
        prefill_past,
        output,
        wall_seconds,
        state_sha256,
        cursor_past,
        kernel_profile: Value::Null,
        prefill_kernel_profile: Value::Null,
    })
}

fn run_profiled(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    config: &DecoderConfig,
    tokens: &[u32],
) -> Result<DecodeRun> {
    let mut session = Session::new(context, config)?;
    let ((prefill_logits, prefill_token, prefill_past), prefill_kernel_profile) =
        super::launch_profile::capture(context, || {
            run_prefill(context, module, model, tokens, &mut session)
        })?;
    let (output, kernel_profile, wall_seconds) =
        profiled_forward(context, module, model, prefill_token, &mut session)?;
    check_committed(&session, &output, tokens.len() + 1)?;
    let state_sha256 = state_hash(&session)?;
    let cursor_past = session.cursor.past();
    drop(session);
    context.synchronize()?;
    Ok(DecodeRun {
        layer_tails: Vec::new(),
        prefill_logits,
        prefill_token,
        prefill_past,
        output,
        wall_seconds,
        state_sha256,
        cursor_past,
        kernel_profile,
        prefill_kernel_profile,
    })
}

fn run_prefill(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    tokens: &[u32],
    session: &mut Session<'_>,
) -> Result<(Vec<u16>, u32, usize)> {
    let output = model.forward(context, module, tokens, session, None)?;
    context.synchronize()?;
    check_committed(session, &output, tokens.len())?;
    Ok((output.logits, output.token, output.past))
}

fn check_committed(session: &Session<'_>, output: &Output, expected_past: usize) -> Result<()> {
    ensure!(
        output.past == expected_past
            && session.cursor.past() == expected_past
            && !session.cursor.is_poisoned(),
        "profile forward did not commit the expected clean cursor"
    );
    Ok(())
}

fn timed_forward(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    token: u32,
    session: &mut Session<'_>,
) -> Result<(Output, f64)> {
    context.synchronize()?;
    let start = Instant::now();
    let forward = model.forward(context, module, &[token], session, None);
    let synchronization = context.synchronize();
    let seconds = start.elapsed().as_secs_f64();
    let output = combine_forward_and_sync(forward, synchronization)?;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "invalid decode wall time"
    );
    Ok((output, seconds))
}

fn profiled_forward(
    context: &Context,
    module: &Module<'_>,
    model: &Model<'_, '_>,
    token: u32,
    session: &mut Session<'_>,
) -> Result<(Output, Value, f64)> {
    context.synchronize()?;
    let start = Instant::now();
    let capture = super::launch_profile::capture(context, || {
        model.forward(context, module, &[token], session, None)
    });
    let synchronization = context.synchronize();
    let seconds = start.elapsed().as_secs_f64();
    let (output, profile) = combine_forward_and_sync(capture, synchronization)?;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "invalid profiled decode wall time"
    );
    Ok((output, profile, seconds))
}

fn combine_forward_and_sync<T>(forward: Result<T>, synchronization: Result<()>) -> Result<T> {
    match (forward, synchronization) {
        (Ok(output), Ok(())) => Ok(output),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error).context("synchronize completed model forward"),
        (Err(error), Err(sync_error)) => Err(error.context(format!(
            "model forward also failed to synchronize: {sync_error:#}"
        ))),
    }
}

fn validate_kernel_profile(profile: &Value) -> Result<()> {
    let launch_count = profile
        .get("launch_count")
        .and_then(Value::as_u64)
        .context("kernel profile omitted a u64 launch_count")?;
    let total_gpu_ms = profile
        .get("total_gpu_ms")
        .and_then(Value::as_f64)
        .context("kernel profile omitted total_gpu_ms")?;
    ensure!(launch_count > 0, "profile captured no kernel launches");
    ensure!(
        total_gpu_ms.is_finite() && total_gpu_ms >= 0.0,
        "kernel profile has invalid total GPU time"
    );
    ensure!(
        profile.get("groups").and_then(Value::as_array).is_some(),
        "kernel profile omitted grouped launch timings"
    );
    Ok(())
}

fn state_hash(session: &Session<'_>) -> Result<String> {
    let mut hash = Sha256::new();
    for region in &session.state.layout().regions {
        let length = usize::try_from(region.length)
            .with_context(|| format!("state region {} does not fit usize", region.name))?;
        let mut bytes = vec![0; length];
        session.state.read_region(&region.name, &mut bytes)?;
        hash.update(region.name.as_bytes());
        hash.update(bytes);
    }
    Ok(hex::encode(hash.finalize()))
}

fn logit_drift(actual: &[u16], expected: &[u16]) -> Value {
    let mut error = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut finite = actual.len() == expected.len();
    for (&a, &b) in actual.iter().zip(expected) {
        let a = f64::from(f32::from_bits(u32::from(a) << 16));
        let b = f64::from(f32::from_bits(u32::from(b) << 16));
        finite &= a.is_finite() && b.is_finite();
        error += (a - b).powi(2);
        left_norm += a * a;
        right_norm += b * b;
        dot += a * b;
        max_error = max_error.max((a - b).abs());
    }
    json!({"finite":finite,"normalized_l2":(error/right_norm.max(1e-30)).sqrt(),"cosine":dot/(left_norm*right_norm).sqrt().max(1e-30),"max_abs_error":max_error,"diagnostic_only":true})
}

#[cfg(test)]
mod tests {
    use super::{MAX_CAPACITY, MAX_PREFIX_TOKENS, validate_kernel_profile, validate_request};
    use serde_json::json;

    #[test]
    fn validates_prefix_vocabulary_capacity_and_fixed_depth() {
        assert!(validate_request(&[7], 10, 2, 64).is_ok());
        assert!(validate_request(&[7; MAX_PREFIX_TOKENS], 10, MAX_CAPACITY, 64).is_ok());
        assert!(validate_request(&[], 10, 1, 64).is_err());
        assert!(validate_request(&[10], 10, 2, 64).is_err());
        assert!(validate_request(&[7], 10, 1, 64).is_err());
        assert!(validate_request(&[7; MAX_PREFIX_TOKENS + 1], 10, MAX_CAPACITY, 64).is_err());
        assert!(validate_request(&[7], 10, 2, 63).is_err());
    }

    #[test]
    fn requires_nonempty_finite_kernel_profile() {
        let valid = json!({"launch_count": 2, "total_gpu_ms": 0.1, "groups": []});
        assert!(validate_kernel_profile(&valid).is_ok());
        assert!(
            validate_kernel_profile(&json!({"launch_count": 0, "total_gpu_ms": 0.1, "groups": []}))
                .is_err()
        );
        assert!(
            validate_kernel_profile(
                &json!({"launch_count": 1, "total_gpu_ms": -1.0, "groups": []})
            )
            .is_err()
        );
        assert!(validate_kernel_profile(&json!({"launch_count": 1, "total_gpu_ms": 0.1})).is_err());
    }
}
