//! Separate bounded measurement lane: real prefill partitions over one session.
//! No changes to arithmetic, existing benchmark bounds, or execution defaults.

use super::{StreamForward, StreamOutput, ensure_supported_profiles, plan::ArenaPlan, program};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    engine::{
        layout::Layout,
        prefill_chunks::{Plan, checked_end},
    },
    kernels::{
        ChunkedBenchRequest, DecoderConfig, attention_profile,
        cuda::{
            driver::{Context, Module},
            resident_model::Session,
            resident_weights::ResidentWeights,
        },
        fp8_profile, nvfp4_profile,
    },
    packages::qwen3_8_27b::{decoder, inventory, schedule},
};
use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, time::Instant};

const MARGIN_BYTES: u64 = 512 * 1024 * 1024;

pub(in crate::kernels) fn run(
    path: &Path,
    ptx: &str,
    device: i32,
    request: &ChunkedBenchRequest<'_>,
    report: &mut Value,
) -> Result<()> {
    ensure!(
        report.is_object(),
        "chunked benchmark report must be an object"
    );
    let plan = Plan::new(
        request.tokens.len(),
        request.chunk_size,
        request.output_tokens,
    )?;
    ensure!(
        (1..=5).contains(&request.repetitions),
        "repetitions must be 1..=5"
    );
    let config = decoder::config(plan.capacity)?;
    plan.validate_capacity(config.capacity)?;
    ensure!(
        request
            .tokens
            .iter()
            .all(|&id| (id as usize) < config.vocabulary),
        "input token outside vocabulary"
    );
    ensure_supported_profiles()?;
    ensure!(
        ptx.contains(".target sm_120a"),
        "chunked benchmark requires SM120a PTX"
    );
    report["plan"] = json!(plan);
    report["configured_capacity"] = json!(config.capacity);
    report["profiles"] = json!({
        "fp8": fp8_profile::current()?.name(),
        "nvfp4": nvfp4_profile::current()?.name(),
        "attention": attention_profile::current()?.name(),
        "execution": "stream", "gpu_greedy": true,
        "mlp_workspace": false, "fp8_split_k": false, "nvfp4_audit": false,
    });
    let mut artifact = ModelArtifact::open(path)?;
    report["identity"] = json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    report["source_checkpoint"] = json!(artifact.directory().source);
    report["recipe_sha256"] = json!(artifact.directory().recipe_sha256);
    inventory::validate(artifact.directory())?;
    let objects = schedule::text_objects(artifact.directory())?;
    // The diagnostic always executes two decode forwards, even when the timed
    // request has only two outputs. Its separate capacity never alters the bench.
    let diagnostic = if request.tokens.len() <= 512 {
        Some(decoder::config(checked_end(
            request.tokens.len(),
            2,
            usize::MAX,
        )?)?)
    } else {
        None
    };
    let budget = Budget::new(&objects, &config, diagnostic.as_ref(), &plan)?;
    report["memory_admission"] = json!(budget);
    report["phase"] = json!("context_and_admission");
    let context = Context::new(device)?;
    report["device"] = json!(context.info());
    ensure!(
        (context.info().major, context.info().minor) == (12, 0),
        "chunked benchmark requires SM120"
    );
    admit(&context, budget.required_bytes, report, "before_module")?;
    let module = Module::load(&context, ptx)?;
    admit(
        &context,
        budget.required_bytes,
        report,
        "before_allocations",
    )?;
    report["phase"] = json!("weight_upload");
    let weights = ResidentWeights::load(&context, &mut artifact, &objects)?;
    run_repetitions(&weights, &module, &config, request, &plan, report)?;
    report["phase"] = json!("partition_check");
    if let Some(config) = diagnostic {
        admit(
            &context,
            sum(&[budget.diagnostic_working_bytes, MARGIN_BYTES])?,
            report,
            "before_diagnostics_weights_already_resident",
        )?;
        partition_check(&weights, &module, &config, request, &plan, report)?;
    } else {
        report["partition_check"] = json!({
            "status": "not_run", "passed": null,
            "reason": "one-shot StreamForward is bounded to 512 prompt rows; long prompt partition equivalence is untested",
        });
    }
    report["phase"] = json!("complete");
    report["completed"] = json!(true);
    Ok(())
}

#[derive(Serialize)]
struct Budget {
    weight_bytes: u64,
    benchmark_state_bytes: u64,
    benchmark_arena_bytes: u64,
    benchmark_rope_bytes: u64,
    benchmark_attention_workspace_bytes: u64,
    diagnostic_attention_workspace_bytes: u64,
    diagnostic_working_bytes: u64,
    margin_bytes: u64,
    required_bytes: u64,
}

impl Budget {
    fn new(
        objects: &[Object],
        config: &DecoderConfig,
        diagnostic: Option<&DecoderConfig>,
        plan: &Plan,
    ) -> Result<Self> {
        let weight_bytes = Layout::new(objects.iter().map(|o| (o.name.clone(), o.length)))?.bytes;
        let benchmark_state_bytes = canonical_state_bytes(config)?;
        let benchmark_arena_bytes = arena_bytes(config, plan.max_rows)?;
        let benchmark_rope_bytes = rope_bytes(config)?;
        let profile = attention_profile::current()?;
        let benchmark_attention_workspace_bytes =
            attention_workspace_bytes(config, plan.max_rows, profile)?;
        let working = sum(&[
            benchmark_state_bytes,
            benchmark_arena_bytes,
            benchmark_rope_bytes,
            benchmark_attention_workspace_bytes,
        ])?;
        let diagnostic_attention_workspace_bytes = if let Some(config) = diagnostic {
            sum(&[
                attention_workspace_bytes(config, plan.max_rows, profile)?,
                attention_workspace_bytes(config, plan.prompt_tokens, profile)?,
            ])?
        } else {
            0
        };
        let diagnostic_working_bytes = if let Some(config) = diagnostic {
            // Two streams and two fresh sessions coexist only during diagnostics.
            sum(&[
                canonical_state_bytes(config)?,
                canonical_state_bytes(config)?,
                arena_bytes(config, plan.max_rows)?,
                arena_bytes(config, plan.prompt_tokens)?,
                rope_bytes(config)?,
                rope_bytes(config)?,
                diagnostic_attention_workspace_bytes,
            ])?
        } else {
            0
        };
        Ok(Self {
            weight_bytes,
            benchmark_state_bytes,
            benchmark_arena_bytes,
            benchmark_rope_bytes,
            benchmark_attention_workspace_bytes,
            diagnostic_attention_workspace_bytes,
            diagnostic_working_bytes,
            margin_bytes: MARGIN_BYTES,
            required_bytes: sum(&[
                weight_bytes,
                working.max(diagnostic_working_bytes),
                MARGIN_BYTES,
            ])?,
        })
    }
}

fn sum(bytes: &[u64]) -> Result<u64> {
    bytes.iter().try_fold(0_u64, |total, &bytes| {
        total
            .checked_add(bytes)
            .context("memory admission overflows u64")
    })
}

fn canonical_state_bytes(config: &DecoderConfig) -> Result<u64> {
    let layout = Layout::new(
        config
            .state_layout
            .regions
            .iter()
            .map(|r| (r.name.clone(), r.length)),
    )?;
    ensure!(layout == config.state_layout, "noncanonical state layout");
    Ok(layout.bytes)
}

fn arena_bytes(config: &DecoderConfig, rows: usize) -> Result<u64> {
    let shapes = program::Shapes::from_config(config)?;
    let plan = ArenaPlan::place(&program::forward_program(&shapes, rows)?)?;
    Ok(u64::try_from(plan.total_bytes)?)
}

fn rope_bytes(config: &DecoderConfig) -> Result<u64> {
    let half = u64::try_from(config.attention_shape.rotary_dim / 2)?;
    u64::try_from(config.capacity)?
        .checked_mul(half)
        .and_then(|x| x.checked_mul(4))
        .context("RoPE admission overflow")
}

fn attention_workspace_bytes(
    config: &DecoderConfig,
    rows: usize,
    profile: attention_profile::Profile,
) -> Result<u64> {
    if profile.uses_staged(1) {
        let schedule = crate::kernels::attention_staged_plan::CoefficientSchedule::current()?;
        let plan = crate::kernels::attention_staged_plan::Plan::new_with_schedule(
            [1, 24, 4, 256, 0, config.capacity],
            schedule,
        )?;
        return Ok(u64::try_from(plan.workspace_bytes)?);
    }
    if profile != attention_profile::Profile::SplitDecode {
        return Ok(0);
    }
    Ok(u64::try_from(
        crate::kernels::attention_v2_plan::persistent_workspace_bytes(rows, config.capacity)?,
    )?)
}

fn admit(context: &Context, required: u64, report: &mut Value, label: &str) -> Result<()> {
    let (free, total) = context.memory()?;
    report["memory_admission"][label] = json!({"free_bytes": free, "total_bytes": total});
    ensure!(
        required <= u64::try_from(free)?,
        "chunked benchmark requires {required} free bytes including margin, found {free}"
    );
    Ok(())
}

fn run_repetitions(
    weights: &ResidentWeights<'_>,
    module: &Module<'_>,
    config: &DecoderConfig,
    request: &ChunkedBenchRequest<'_>,
    plan: &Plan,
    report: &mut Value,
) -> Result<()> {
    let stream = StreamForward::new(weights, module, config, plan.max_rows)?;
    report["stream_forward"] = stream.report();
    report["phase"] = json!("warmup");
    let warmup = sequence(&stream, config, request.tokens, plan, 2)?;
    report["warmup"] = json!({
        "completed": true, "prompt_tokens": plan.prompt_tokens, "output_tokens": 2,
        "chunks": plan.chunks.len(), "excluded_from_measurements": true,
        "sequence_wall_seconds": warmup.sequence_wall_seconds,
    });
    report["phase"] = json!("timed_repetitions");
    report["repetitions"] = json!([]);
    for index in 0..request.repetitions {
        report["active_repetition"] = json!(index);
        let measurement = sequence(&stream, config, request.tokens, plan, request.output_tokens)?;
        report["repetitions"]
            .as_array_mut()
            .context("missing repetitions array")?
            .push(json!(measurement));
    }
    report["active_repetition"] = Value::Null;
    Ok(())
}

#[derive(Serialize)]
struct ChunkTiming {
    start_row: usize,
    end_row: usize,
    rows: usize,
    past_before: usize,
    past_after: usize,
    start_seconds: f64,
    end_seconds: f64,
    seconds: f64,
}

#[derive(Serialize)]
struct Measurement {
    chunks: Vec<ChunkTiming>,
    prefill_sum_seconds: f64,
    prefill_wall_seconds: f64,
    prefill_tokens_per_second: f64,
    decode_intervals_seconds: Vec<f64>,
    decode_interval_count: usize,
    decode_sum_seconds: f64,
    decode_wall_seconds: f64,
    decode_tokens_per_second: f64,
    generated_token_ids: Vec<u32>,
    final_past: usize,
    sequence_wall_seconds: f64,
}

fn sequence(
    stream: &StreamForward<'_, '_, '_>,
    config: &DecoderConfig,
    tokens: &[u32],
    plan: &Plan,
    outputs: usize,
) -> Result<Measurement> {
    ensure!(
        (2..=plan.output_tokens).contains(&outputs),
        "invalid sequence output count"
    );
    let required = sum(&[config.state_layout.bytes, MARGIN_BYTES])?;
    ensure!(
        required <= u64::try_from(stream.context.memory()?.0)?,
        "insufficient free memory for fresh session plus margin"
    );
    let mut session = Session::new(stream.context, config)?;
    stream.context.synchronize()?;
    let start = Instant::now();
    let mut chunks = Vec::with_capacity(plan.chunks.len());
    let mut final_token = None;
    for chunk in &plan.chunks {
        ensure!(
            session.cursor.past() == chunk.start,
            "prefill cursor disagrees with plan"
        );
        let expected = checked_end(chunk.start, chunk.rows(), config.capacity)?;
        let chunk_start = Instant::now();
        let output = stream.forward(&tokens[chunk.start..chunk.end], &mut session, false)?;
        let chunk_end = Instant::now();
        ensure!(output.past == expected, "prefill cursor advance mismatch");
        chunks.push(ChunkTiming {
            start_row: chunk.start,
            end_row: chunk.end,
            rows: chunk.rows(),
            past_before: chunk.start,
            past_after: output.past,
            start_seconds: chunk_start.duration_since(start).as_secs_f64(),
            end_seconds: chunk_end.duration_since(start).as_secs_f64(),
            seconds: chunk_end.duration_since(chunk_start).as_secs_f64(),
        });
        // Intermediate last-row head selections are deliberately discarded.
        final_token = Some(output.token);
    }
    let prefill_wall_seconds = start.elapsed().as_secs_f64();
    let mut generated_token_ids = vec![final_token.context("prefill produced no final token")?];
    let mut decode_intervals_seconds = Vec::with_capacity(outputs - 1);
    let decode_start = Instant::now();
    for _ in 1..outputs {
        let expected = checked_end(session.cursor.past(), 1, config.capacity)?;
        let token = *generated_token_ids.last().context("no decode input")?;
        let step_start = Instant::now();
        let output = stream.forward(&[token], &mut session, false)?;
        decode_intervals_seconds.push(step_start.elapsed().as_secs_f64());
        ensure!(output.past == expected, "decode cursor advance mismatch");
        generated_token_ids.push(output.token);
    }
    let decode_wall_seconds = decode_start.elapsed().as_secs_f64();
    let sequence_wall_seconds = start.elapsed().as_secs_f64();
    let decode_sum_seconds: f64 = decode_intervals_seconds.iter().sum();
    let expected = checked_end(tokens.len(), outputs - 1, config.capacity)?;
    ensure!(
        session.cursor.past() == expected && generated_token_ids.len() == outputs,
        "fixed-output sequence incomplete"
    );
    Ok(Measurement {
        prefill_sum_seconds: chunks.iter().map(|chunk| chunk.seconds).sum(),
        chunks,
        prefill_wall_seconds,
        prefill_tokens_per_second: tokens.len() as f64 / prefill_wall_seconds,
        decode_interval_count: decode_intervals_seconds.len(),
        decode_intervals_seconds,
        decode_sum_seconds,
        decode_wall_seconds,
        decode_tokens_per_second: (outputs - 1) as f64 / decode_sum_seconds,
        generated_token_ids,
        final_past: session.cursor.past(),
        sequence_wall_seconds,
    })
}

fn partition_check(
    weights: &ResidentWeights<'_>,
    module: &Module<'_>,
    config: &DecoderConfig,
    request: &ChunkedBenchRequest<'_>,
    plan: &Plan,
    report: &mut Value,
) -> Result<()> {
    report["partition_check"] = json!({
        "status": "running", "passed": null, "configured_capacity": config.capacity,
        "decode_steps": 2, "steps": [], "excluded_from_timing": true,
        "comparison": "exact BF16 logits, selected token, cursor and SHA256 of every initialized state region",
        "input_policy": "both decode paths consume the one-shot selected token, even after divergence",
        "state_scope": "initialized KV prefix only; complete convolution history and recurrent regions; unused KV tail excluded",
    });
    let one_shot = StreamForward::new(weights, module, config, plan.prompt_tokens)?;
    let chunked = StreamForward::new(weights, module, config, plan.max_rows)?;
    let mut reference = Session::new(weights.context(), config)?;
    let mut candidate = Session::new(weights.context(), config)?;
    weights.context().synchronize()?;
    let mut expected = one_shot.forward(request.tokens, &mut reference, true)?;
    let mut actual = None;
    for chunk in &plan.chunks {
        ensure!(
            candidate.cursor.past() == chunk.start,
            "diagnostic prefill cursor mismatch"
        );
        let last = chunk.end == plan.prompt_tokens;
        actual = Some(chunked.forward(
            &request.tokens[chunk.start..chunk.end],
            &mut candidate,
            last,
        )?);
    }
    let mut actual = actual.context("diagnostic prefill produced no token")?;
    let mut passed = true;
    let mut input = request.tokens.to_vec();
    for step in 0..=2 {
        if step > 0 {
            input = vec![expected.token];
            expected = one_shot.forward(&input, &mut reference, true)?;
            actual = chunked.forward(&input, &mut candidate, true)?;
        }
        let check = compare_step(step, &input, &expected, &actual, &reference, &candidate)?;
        passed &= check["passed"] == true;
        report["partition_check"]["steps"]
            .as_array_mut()
            .context("missing diagnostic steps")?
            .push(check);
    }
    report["partition_check"]["status"] = json!("completed");
    // Numerical partition differences are retained as a strict failed check,
    // not turned into a benchmark execution error or a relaxed tolerance pass.
    report["partition_check"]["passed"] = json!(passed);
    Ok(())
}

fn compare_step(
    index: usize,
    input: &[u32],
    expected: &StreamOutput,
    actual: &StreamOutput,
    reference: &Session<'_>,
    candidate: &Session<'_>,
) -> Result<Value> {
    let left = expected
        .logits
        .as_ref()
        .context("reference omitted diagnostic logits")?;
    let right = actual
        .logits
        .as_ref()
        .context("candidate omitted diagnostic logits")?;
    let reference_state = state_hashes(reference)?;
    let candidate_state = state_hashes(candidate)?;
    let mismatched: Vec<_> = reference_state
        .iter()
        .filter(|(key, value)| candidate_state.get(*key) != Some(*value))
        .map(|(key, _)| key)
        .collect();
    let passed = left == right
        && expected.token == actual.token
        && expected.past == actual.past
        && reference_state == candidate_state;
    Ok(json!({
        "step": index, "input_token_ids": input,
        "one_shot_token": expected.token, "chunked_token": actual.token,
        "token_equal": expected.token == actual.token,
        "one_shot_past": expected.past, "chunked_past": actual.past,
        "logits_equal": left == right,
        "one_shot_logits_sha256": logit_hash(left), "chunked_logits_sha256": logit_hash(right),
        "state": {"equal": reference_state == candidate_state, "mismatched_regions": mismatched,
            "one_shot": reference_state, "chunked": candidate_state},
        "passed": passed,
    }))
}

#[derive(Serialize, PartialEq)]
struct RegionHash {
    bytes_hashed: usize,
    sha256: String,
}

fn state_hashes(session: &Session<'_>) -> Result<BTreeMap<String, RegionHash>> {
    let mut hashes = BTreeMap::new();
    for region in &session.state.layout().regions {
        let length = usize::try_from(region.length)?;
        let mut bytes = vec![0; length];
        session.state.read_region(&region.name, &mut bytes)?;
        let initialized =
            if region.name.ends_with(".attention.k") || region.name.ends_with(".attention.v") {
                let capacity = session.cursor.capacity();
                ensure!(
                    capacity > 0 && length % capacity == 0,
                    "KV region not divisible by capacity"
                );
                session
                    .cursor
                    .past()
                    .checked_mul(length / capacity)
                    .context("initialized KV extent overflow")?
            } else {
                length
            };
        ensure!(initialized <= length, "initialized state exceeds region");
        hashes.insert(
            region.name.clone(),
            RegionHash {
                bytes_hashed: initialized,
                sha256: hex::encode(Sha256::digest(&bytes[..initialized])),
            },
        );
    }
    Ok(hashes)
}

fn logit_hash(logits: &[u16]) -> String {
    let mut hash = Sha256::new();
    for word in logits {
        hash.update(word.to_le_bytes());
    }
    hex::encode(hash.finalize())
}
