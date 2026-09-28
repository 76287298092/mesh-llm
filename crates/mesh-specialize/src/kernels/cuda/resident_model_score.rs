//! Windowed teacher-forced scoring of fixed token streams. Every window runs
//! from a freshly allocated, zeroed state and cursor: no KV or recurrent state
//! carries between windows or streams.
use super::{
    driver::{Context, Module},
    resident_model::{Model, Session},
    resident_score::{self, Scorer},
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::{
        layout::Layout,
        teacher_scoring::{self, RECORD_BYTES, TOP_K, Window},
    },
    kernels::{DecoderConfig, ModelScoreRequest, ScoreSink},
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, time::Instant};

/// Current single-forward limit; larger contexts need chunked prefill.
pub(in crate::kernels) const MAX_CONTEXT: usize = 512;
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
}

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    request: &ModelScoreRequest<'_>,
    sink: &mut ScoreSink<'_>,
) -> Result<Value> {
    let plans = plan(request, config)?;
    ensure!(ptx.contains(".target sm_120a"), "scoring requires SM120a PTX");
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
        "context must be 2..={MAX_CONTEXT} until chunked prefill exists (current single-forward limit)"
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
        let windows = teacher_scoring::plan_windows(
            stream.tokens.len(),
            request.context,
            request.stride,
        )
        .with_context(|| format!("plan windows for stream {}", stream.id))?;
        plans.push(Plan { stream, windows });
    }
    Ok(plans)
}

impl Runner<'_, '_, '_, '_> {
    fn score_stream(
        &mut self,
        index: usize,
        plan: &Plan<'_>,
        sink: &mut ScoreSink<'_>,
    ) -> Result<(Value, Aggregate)> {
        let started = Instant::now();
        let mut aggregate = Aggregate::default();
        let mut windows = Vec::new();
        for (window_index, window) in plan.windows.iter().enumerate() {
            let window_started = Instant::now();
            let (scored, total_nll, bytes) = self.score_window(plan.stream.tokens.as_slice(), window)?;
            sink(index, &bytes).with_context(|| format!("write records for {}", plan.stream.id))?;
            aggregate.add(scored, total_nll);
            let mut report = Aggregate {
                scored,
                total_nll,
            }
            .json();
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
        report["unscored_tokens"] = json!(1);
        report["record_count"] = json!(aggregate.scored);
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        report["windows"] = json!(windows);
        Ok((report, aggregate))
    }

    /// One window from fresh state. Returns (scored, total NLL, encoded records).
    fn score_window(&mut self, tokens: &[u32], window: &Window) -> Result<(usize, f64, Vec<u8>)> {
        let inputs = &tokens[window.input_begin..window.input_end];
        let targets = &tokens[window.target_begin..window.target_end];
        let mut session = Session::new(self.context, self.config)?;
        let hidden = self
            .model
            .forward_hidden(self.context, self.module, inputs, &mut session)?;
        drop(session);
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
            rows,
            window.first_row(),
            targets,
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
