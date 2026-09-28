//! Windowed teacher-forced scoring of fixed token streams. Every window runs
//! from a freshly allocated, zeroed state and cursor: no KV or recurrent state
//! carries between windows or streams.
use super::{
    driver::{Buffer, Context, Module},
    resident_model::{Model, Session},
    resident_score::{self, Scorer},
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    engine::{
        layout::Layout,
        teacher_scoring::{self, RECORD_BYTES, TOP_K, Window},
    },
    kernels::{DecoderConfig, ModelScoreRequest, ScoreSink},
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env::VarError, ops::Range, time::Instant};

/// Qualified scoring context ceiling, independent of the forward chunk size.
pub(in crate::kernels) const MAX_CONTEXT: usize = 512;
const FORWARD_ROWS_ENV: &str = "MESH_SPECIALIZE_SCORE_FORWARD_ROWS";
const DEFAULT_FORWARD_ROWS: usize = 512;

struct ForwardChunk {
    inputs: Range<usize>,
    byte_offset: usize,
    bytes: usize,
}

struct ForwardPlan {
    bytes: usize,
    chunks: Vec<ForwardChunk>,
}

impl ForwardPlan {
    fn new(rows: usize, width: usize, forward_rows: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT).contains(&rows),
            "scoring window rows must be 1..={MAX_CONTEXT}"
        );
        ensure!(
            (1..=DEFAULT_FORWARD_ROWS).contains(&forward_rows),
            "{FORWARD_ROWS_ENV} must be an integer in 1..={DEFAULT_FORWARD_ROWS}"
        );
        ensure!(width > 0, "scoring hidden width must be positive");
        let row_bytes = width
            .checked_mul(2)
            .context("hidden row byte size overflows")?;
        let bytes = rows
            .checked_mul(row_bytes)
            .context("hidden window byte size overflows")?;
        let chunks = (0..rows)
            .step_by(forward_rows)
            .map(|start| {
                let end = (start + forward_rows).min(rows);
                ForwardChunk {
                    inputs: start..end,
                    byte_offset: start * row_bytes,
                    bytes: (end - start) * row_bytes,
                }
            })
            .collect();
        Ok(Self { bytes, chunks })
    }
}

fn parse_forward_rows(value: Result<String, VarError>) -> Result<usize> {
    match value {
        Err(VarError::NotPresent) => Ok(DEFAULT_FORWARD_ROWS),
        Ok(value) if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
            let rows = value.parse::<usize>().with_context(|| {
                format!("{FORWARD_ROWS_ENV} must be an integer in 1..={DEFAULT_FORWARD_ROWS}")
            })?;
            ensure!(
                (1..=DEFAULT_FORWARD_ROWS).contains(&rows),
                "{FORWARD_ROWS_ENV} must be an integer in 1..={DEFAULT_FORWARD_ROWS}"
            );
            Ok(rows)
        }
        _ => anyhow::bail!("{FORWARD_ROWS_ENV} must be an integer in 1..={DEFAULT_FORWARD_ROWS}"),
    }
}
const MEMORY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const SCORING_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Default)]
struct Aggregate {
    scored: usize,
    total_nll: f64,
}

impl Aggregate {
    fn add(&mut self, scored: usize, total_nll: f64) {
        self.scored += scored;
        self.total_nll += total_nll;
    }
    fn json(&self) -> Value {
        let mean = if self.scored == 0 {
            Value::Null
        } else {
            json!(self.total_nll / self.scored as f64)
        };
        json!({"scored_tokens": self.scored, "total_nll": self.total_nll, "mean_nll": mean})
    }
}

struct Plan<'a> {
    stream: &'a crate::kernels::ScoreStream,
    windows: Vec<Window>,
}

struct Runner<'r, 'm, 'w, 'ctx> {
    context: &'ctx Context,
    module: &'r Module<'ctx>,
    model: &'m Model<'w, 'ctx>,
    scorer: Scorer<'m, 'w, 'ctx>,
    config: &'r DecoderConfig,
    check: Option<Value>,
    hash_logits: bool,
    forward_rows: usize,
}

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut ModelArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    request: &ModelScoreRequest<'_>,
    sink: &mut ScoreSink<'_>,
) -> Result<Value> {
    let plans = plan(request, config)?;
    validate_score_execution(std::env::var("MESH_SPECIALIZE_EXECUTION"))?;
    let hash_logits = logit_hash_enabled()?;
    let forward_rows = parse_forward_rows(std::env::var(FORWARD_ROWS_ENV))?;
    ensure!(
        ptx.contains(".target sm_120a"),
        "scoring requires SM120a PTX"
    );
    let layout = Layout::new(objects.iter().map(|o| (o.name.clone(), o.length)))?;
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "scoring requires selected SM120 device"
    );
    let module = Module::load(&context, ptx)?;
    module
        .function(resident_score::KERNEL)
        .context("load row log-probability kernel")?;
    validate_admission(&layout, config, context.memory()?.0)?;
    let load_started = Instant::now();
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    let model = Model::new(&weights, config)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let mut runner = Runner {
        context: &context,
        module: &module,
        model: &model,
        scorer: Scorer::new(model.head(), config.hidden, config.vocabulary)?,
        config,
        check: None,
        hash_logits,
        forward_rows,
    };
    let started = Instant::now();
    let mut streams = Vec::new();
    let mut domains = BTreeMap::<String, Aggregate>::new();
    let mut overall = Aggregate::default();
    for (index, plan) in plans.iter().enumerate() {
        let (report, aggregate) = runner.score_stream(index, plan, sink)?;
        domains
            .entry(plan.stream.domain.clone())
            .or_default()
            .add(aggregate.scored, aggregate.total_nll);
        overall.add(aggregate.scored, aggregate.total_nll);
        streams.push(report);
    }
    let score_seconds = started.elapsed().as_secs_f64();
    let check = runner.check.take().unwrap_or(Value::Null);
    let chunk_rows = runner.scorer.chunk_rows();
    drop(runner);
    drop(model);
    drop(weights);
    context.synchronize()?;
    let domain_reports = domains
        .iter()
        .map(|(domain, aggregate)| {
            let mut item = aggregate.json();
            item["domain"] = json!(domain);
            item
        })
        .collect::<Vec<_>>();
    let tokens_per_second = overall.scored as f64 / score_seconds.max(f64::MIN_POSITIVE);
    Ok(json!({
        "schema_version": 1,
        "kind": "resident-teacher-forced-scores",
        "execution": "legacy",
        "all_passed": check["passed"] == true,
        "corpus_id": request.corpus_id,
        "context_tokens": request.context,
        "stride_tokens": request.stride,
        "metric": {"name": "fixed-window truncated-context causal perplexity", "log_base": "natural"},
        "state_policy": "fresh zeroed ResidentState and cursor per window; no carry-over between windows or streams",
        "record_format": record_format(),
        "profiles": profiles()?,
        "environment": environment(),
        "head_chunk_rows": chunk_rows,
        "forward_rows": forward_rows,
        "full_logit_hash": {"enabled": hash_logits,
            "scope": "SHA-256 of every little-endian BF16 vocabulary value at scored positions, in window order"},
        "device": info,
        "check": check,
        "streams": streams,
        "domains": domain_reports,
        "overall": overall.json(),
        "timing": {"load_seconds": load_seconds, "score_seconds": score_seconds,
            "scored_tokens_per_second": tokens_per_second},
        "scope": "Teacher-forced scoring of fixed token streams; score timing includes the forward, head, statistics and host transfers and is not a prefill benchmark.",
    }))
}

fn plan<'a>(request: &'a ModelScoreRequest<'a>, config: &DecoderConfig) -> Result<Vec<Plan<'a>>> {
    ensure!(
        (2..=MAX_CONTEXT).contains(&request.context),
        "context must be 2..={MAX_CONTEXT} (qualified scoring context ceiling)"
    );
    ensure!(
        config.capacity == request.context && config.layers.len() == 64,
        "scoring decoder capacity must equal the context over 64 layers"
    );
    ensure!(!request.streams.is_empty(), "no streams to score");
    let mut identifiers = std::collections::BTreeSet::new();
    let mut plans = Vec::new();
    for stream in request.streams {
        ensure!(
            identifiers.insert(stream.id.as_str()) && !stream.id.is_empty(),
            "stream ids must be unique and nonempty"
        );
        ensure!(
            stream
                .tokens
                .iter()
                .all(|&t| (t as usize) < config.vocabulary),
            "stream {} contains a token outside the vocabulary",
            stream.id
        );
        let windows =
            teacher_scoring::plan_windows(stream.tokens.len(), request.context, request.stride)
                .with_context(|| format!("plan windows for stream {}", stream.id))?;
        plans.push(Plan { stream, windows });
    }
    Ok(plans)
}

impl<'ctx> Runner<'_, '_, '_, 'ctx> {
    /// Preserve all session state within a window, but never across windows.
    fn forward_window(&self, inputs: &[u32]) -> Result<Buffer<'ctx>> {
        let plan = ForwardPlan::new(inputs.len(), self.config.hidden, self.forward_rows)?;
        let mut session = Session::new(self.context, self.config)?;
        if plan.chunks.len() == 1 {
            // Keep the default path unchanged: no extra allocation or copy.
            return self
                .model
                .forward_hidden(self.context, self.module, inputs, &mut session);
        }
        let hidden = Buffer::new(self.context, plan.bytes)?;
        for chunk in plan.chunks {
            let output = self.model.forward_hidden(
                self.context,
                self.module,
                &inputs[chunk.inputs],
                &mut session,
            )?;
            ensure!(
                output.len() == chunk.bytes,
                "forward hidden chunk extent mismatch"
            );
            // Synchronous device copy completes before the chunk is dropped.
            hidden.copy_from_at(chunk.byte_offset, &output, 0, chunk.bytes)?;
        }
        Ok(hidden)
    }

    fn score_stream(
        &mut self,
        index: usize,
        plan: &Plan<'_>,
        sink: &mut ScoreSink<'_>,
    ) -> Result<(Value, Aggregate)> {
        let started = Instant::now();
        let mut aggregate = Aggregate::default();
        let mut windows = Vec::new();
        let mut logit_hash = self.hash_logits.then(Sha256::new);
        for (window_index, window) in plan.windows.iter().enumerate() {
            let window_started = Instant::now();
            let (scored, total_nll, bytes) =
                self.score_window(plan.stream.tokens.as_slice(), window, logit_hash.as_mut())?;
            sink(index, &bytes).with_context(|| format!("write records for {}", plan.stream.id))?;
            aggregate.add(scored, total_nll);
            let mut report = Aggregate { scored, total_nll }.json();
            report["index"] = json!(window_index);
            report["input_begin"] = json!(window.input_begin);
            report["input_end"] = json!(window.input_end);
            report["target_begin"] = json!(window.target_begin);
            report["target_end"] = json!(window.target_end);
            report["first_target"] = json!(window.target_begin - window.input_begin);
            report["seconds"] = json!(window_started.elapsed().as_secs_f64());
            windows.push(report);
        }
        let mut report = aggregate.json();
        report["id"] = json!(plan.stream.id);
        report["domain"] = json!(plan.stream.domain);
        report["input_tokens"] = json!(plan.stream.tokens.len());
        report["input_tokens_sha256"] = json!(token_digest(&plan.stream.tokens));
        report["full_logits_sha256"] = json!(logit_hash.map(|h| hex::encode(h.finalize())));
        report["unscored_tokens"] = json!(1);
        report["record_count"] = json!(aggregate.scored);
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        report["windows"] = json!(windows);
        Ok((report, aggregate))
    }

    /// One window from fresh state. Returns (scored, total NLL, encoded records).
    fn score_window(
        &mut self,
        tokens: &[u32],
        window: &Window,
        logit_hash: Option<&mut Sha256>,
    ) -> Result<(usize, f64, Vec<u8>)> {
        let inputs = &tokens[window.input_begin..window.input_end];
        let targets = &tokens[window.target_begin..window.target_end];
        let hidden = self.forward_window(inputs)?;
        let rows = inputs.len();
        if self.check.is_none() {
            self.check = Some(self.scorer.check(
                self.context,
                self.module,
                &hidden,
                rows,
                window.first_row(),
                targets,
            )?);
        }
        let records = self.scorer.score(
            self.context,
            self.module,
            &hidden,
            window.first_row(),
            targets,
            logit_hash,
        )?;
        ensure!(
            records.len() == targets.len()
                && records.iter().zip(targets).all(|(r, &t)| r.target == t),
            "scored records do not match window targets"
        );
        let total_nll = -records
            .iter()
            .map(|record| f64::from(record.target_logprob))
            .sum::<f64>();
        ensure!(total_nll.is_finite(), "window NLL is not finite");
        let mut bytes = Vec::with_capacity(records.len() * RECORD_BYTES);
        for record in &records {
            record.encode_into(&mut bytes);
        }
        Ok((records.len(), total_nll, bytes))
    }
}

fn validate_admission(layout: &Layout, config: &DecoderConfig, free_bytes: usize) -> Result<()> {
    let required = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .and_then(|bytes| bytes.checked_add(MEMORY_RESERVE_BYTES + SCORING_SCRATCH_BYTES))
        .context("scoring admission size overflows u64")?;
    ensure!(
        required <= u64::try_from(free_bytes)?,
        "insufficient CUDA memory for weights, state, scoring scratch and 1 GiB reserve"
    );
    Ok(())
}

fn record_format() -> Value {
    json!({
        "byte_order": "little-endian",
        "record_bytes": RECORD_BYTES,
        "fields": [
            {"name": "target_id", "type": "u32", "count": 1},
            {"name": "target_logprob", "type": "f32", "count": 1},
            {"name": "logsumexp", "type": "f32", "count": 1},
            {"name": "top_ids", "type": "u32", "count": TOP_K},
            {"name": "top_logprobs", "type": "f32", "count": TOP_K},
        ],
        "order": "scored positions in stream order; top-k by logit descending, lower id first on ties",
        "arithmetic": "BF16 head logits; FP32 exp of (x - max), FP64 accumulation and log, FP32 outputs",
    })
}

fn profiles() -> Result<Value> {
    Ok(json!({
        "fp8": crate::kernels::fp8_profile::current()?.name(),
        "nvfp4": crate::kernels::nvfp4_profile::current()?.name(),
        "attention": crate::kernels::attention_profile::current()?.name(),
        "mlp_workspace": super::model_workspace::enabled()?,
        "fp8_split_k": super::resident_fp8_splitk::configured_splits()?,
    }))
}

fn environment() -> Value {
    let mut variables = std::env::vars()
        .filter(|(name, _)| name.starts_with("MESH_SPECIALIZE_"))
        .collect::<Vec<_>>();
    variables.sort();
    Value::Object(
        variables
            .into_iter()
            .map(|(name, value)| (name, Value::String(value)))
            .collect::<Map<_, _>>(),
    )
}

fn validate_score_execution(value: Result<String, VarError>) -> Result<()> {
    match value {
        Err(VarError::NotPresent) => Ok(()),
        Ok(value) if value == "legacy" => Ok(()),
        _ => anyhow::bail!(
            "teacher-forced scoring uses legacy execution; unset MESH_SPECIALIZE_EXECUTION or select legacy"
        ),
    }
}

fn logit_hash_enabled() -> Result<bool> {
    match std::env::var("MESH_SPECIALIZE_SCORE_LOGITS_HASH").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("off") => Ok(false),
        Ok("on") => Ok(true),
        _ => anyhow::bail!("MESH_SPECIALIZE_SCORE_LOGITS_HASH must be on or off"),
    }
}

fn token_digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_FORWARD_ROWS, FORWARD_ROWS_ENV, ForwardPlan, VarError, parse_forward_rows,
        token_digest, validate_score_execution,
    };
    use crate::engine::teacher_scoring::plan_windows;

    #[test]
    fn scoring_does_not_silently_claim_graph_or_stream_execution() {
        assert!(validate_score_execution(Err(VarError::NotPresent)).is_ok());
        assert!(validate_score_execution(Ok("legacy".into())).is_ok());
        for value in ["stream", "graph", "unknown"] {
            assert!(validate_score_execution(Ok(value.into())).is_err());
        }
    }

    #[test]
    fn forward_rows_defaults_and_bounds() {
        assert_eq!(parse_forward_rows(Err(VarError::NotPresent)).unwrap(), 512);
        for rows in 1..=512 {
            assert_eq!(parse_forward_rows(Ok(rows.to_string())).unwrap(), rows);
        }
    }

    #[test]
    fn forward_rows_rejects_bad_environment_without_mutating_it() {
        for value in [
            "",
            "0",
            "513",
            "-1",
            "+1",
            "1.0",
            "1e2",
            " 1",
            "1 ",
            "on",
            "１",
            "999999999999999999999999999999999",
        ] {
            let error = parse_forward_rows(Ok(value.to_owned())).unwrap_err();
            assert!(error.to_string().contains(FORWARD_ROWS_ENV));
        }
        assert!(parse_forward_rows(Err(VarError::NotUnicode(std::ffi::OsString::new()))).is_err());
    }

    #[test]
    fn forward_plan_preserves_default_and_bounds_every_schedule() {
        for rows in [1, 2, 7, 511, 512] {
            assert_eq!(
                ForwardPlan::new(rows, 17, DEFAULT_FORWARD_ROWS)
                    .unwrap()
                    .chunks
                    .len(),
                1
            );
            for forward_rows in 1..=512 {
                let plan = ForwardPlan::new(rows, 17, forward_rows).unwrap();
                assert_eq!(plan.bytes, rows * 34);
                assert_eq!(plan.chunks.len(), rows.div_ceil(forward_rows));
                let mut next_row = 0;
                let mut next_byte = 0;
                for chunk in plan.chunks {
                    assert_eq!(chunk.inputs.start, next_row);
                    assert_eq!(chunk.byte_offset, next_byte);
                    assert!((1..=forward_rows).contains(&chunk.inputs.len()));
                    assert_eq!(chunk.bytes, chunk.inputs.len() * 34);
                    next_row = chunk.inputs.end;
                    next_byte += chunk.bytes;
                }
                assert_eq!(next_row, rows);
                assert_eq!(next_byte, plan.bytes);
            }
        }
    }

    #[test]
    fn forward_plan_rejects_invalid_extents_and_overflow() {
        for (rows, width, forward_rows) in [
            (0, 17, 1),
            (513, 17, 1),
            (2, 0, 1),
            (2, 17, 0),
            (2, 17, 513),
            (2, usize::MAX, 1),
            (512, usize::MAX / 2, 1),
        ] {
            assert!(ForwardPlan::new(rows, width, forward_rows).is_err());
        }
    }

    #[test]
    fn chunk_offsets_keep_window_target_rows_and_reset_at_each_window() {
        let tokens: Vec<_> = (0..700).collect();
        for forward_rows in [1, 7, 8, 127, 512] {
            let mut predictions = Vec::new();
            for window in plan_windows(tokens.len(), 512, 256).unwrap() {
                let inputs = &tokens[window.input_begin..window.input_end];
                let plan = ForwardPlan::new(inputs.len(), 1, forward_rows).unwrap();
                assert_eq!(plan.chunks[0].byte_offset, 0);
                let mut hidden_rows = vec![usize::MAX; inputs.len()];
                for chunk in plan.chunks {
                    let start = chunk.byte_offset / 2;
                    hidden_rows[start..start + chunk.bytes / 2]
                        .copy_from_slice(&inputs[chunk.inputs]);
                }
                predictions.extend_from_slice(
                    &hidden_rows[window.first_row()..window.first_row() + window.scored()],
                );
            }
            assert_eq!(predictions, tokens[..tokens.len() - 1]);
        }
    }

    #[test]
    fn token_hash_covers_unscored_context_and_order() {
        assert_ne!(token_digest(&[1, 2, 3]), token_digest(&[4, 2, 3]));
        assert_ne!(token_digest(&[1, 2]), token_digest(&[2, 1]));
        assert_eq!(token_digest(&[1, 2]), token_digest(&[1, 2]));
    }
}
