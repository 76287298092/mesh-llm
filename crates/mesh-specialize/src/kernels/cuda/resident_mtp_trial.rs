//! Real-weight MTP oracle, partition, rejection and greedy-output qualification.
use super::{
    driver::{Buffer, Context, Module},
    resident_model::{Model, Session},
    resident_mtp, resident_speculation,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    decoder_ops_reference as ops,
    engine::sampling,
    kernels::{DecoderConfig, SpeculationRequest},
    packages::qwen3_8_27b::mtp::Reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    reference: &Reference,
    request: &SpeculationRequest<'_>,
) -> Result<Value> {
    ensure!(ptx.contains(".target sm_120a"), "MTP requires SM120a PTX");
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!((info.major, info.minor) == (12, 0), "MTP requires SM120");
    let module = Module::load(&ctx, ptx)?;
    let fp8_probe = super::fp8_exact_trial::run(&ctx, &module)?;
    ensure!(
        fp8_probe["all_passed"] == true,
        "MTP projection probe failed: {fp8_probe}"
    );
    let before = ctx.memory()?;
    let weights = ResidentWeights::load(&ctx, artifact, objects)?;
    let target = Model::new(&weights, config)?;
    let draft = resident_mtp::Model::new(&weights, config)?;
    let head = check_head(&ctx, &module, &draft, config, reference)?;
    ensure!(
        head["all_passed"] == true,
        "independent MTP head or partition check failed: {head}"
    );
    let runner = Runner {
        ctx: &ctx,
        module: &module,
        target: &target,
        draft: &draft,
        config,
        request,
    };
    let control = runner.control()?;
    let verification_profile = runner.verification_profile(&control)?;
    let forced_acceptance = resident_speculation::run(
        &ctx,
        &module,
        &target,
        &draft,
        config,
        &resident_speculation::Request {
            tokens: request.tokens,
            output_tokens: request.output_tokens,
            depth: 1,
            forced_first_draft: Some(control.tokens[1]),
        },
    )?;
    ensure!(
        forced_acceptance.all_accepted_rounds > 0,
        "forced correct draft did not exercise a positive all-accepted round"
    );
    let acceptance_report = compare_run(&forced_acceptance, &control)?;
    ensure!(
        acceptance_report["all_passed"] == true,
        "forced all-accepted run differs from target-only output/state: {acceptance_report}"
    );
    drop(forced_acceptance);
    let forced = runner.speculate(Some(
        (control.tokens[1] + 1) % u32::try_from(config.vocabulary)?,
    ))?;
    ensure!(
        forced.replay_rows > 0,
        "forced wrong draft did not exercise target replay"
    );
    let forced_report = compare_run(&forced, &control)?;
    ensure!(
        forced_report["all_passed"] == true,
        "forced rejection did not preserve target output/state: {forced_report}"
    );
    drop(forced);
    let mut trials = Vec::new();
    for _ in 0..request.repetitions {
        let run = runner.speculate(None)?;
        let report = compare_run(&run, &control)?;
        ensure!(
            report["all_passed"] == true,
            "MTP greedy equivalence failed: {report}"
        );
        trials.push(report);
    }
    drop(draft);
    drop(target);
    drop(weights);
    ctx.synchronize()?;
    let after = ctx.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"resident-greedy-mtp-qualification","all_passed":true,
        "device":info,"fp8_probe":fp8_probe,"head":head,"prompt_token_ids":request.tokens,"output_tokens":request.output_tokens,
        "depth":request.depth,"control":control.report(),"verification_profile":verification_profile,"forced_acceptance":acceptance_report,"forced_rejection":forced_report,"trials":trials,
        "memory":{"before_free_bytes":before.0,"after_release_free_bytes":after.0,"arena_memory_release_observed":after.0>=before.0},
        "scope":"bounded greedy single sequence; no stochastic or serving qualification"}),
    )
}

fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    let mut raw = vec![0; buffer.len()];
    buffer.download(&mut raw)?;
    ops::words(&raw)
}
fn upload<'a>(ctx: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    let b = Buffer::new(ctx, words.len() * 2)?;
    b.upload(&ops::bytes(words))?;
    Ok(b)
}
fn state_hash(state: &ResidentState<'_>) -> Result<String> {
    let mut hash = Sha256::new();
    for region in &state.layout().regions {
        let mut bytes = vec![0; usize::try_from(region.length)?];
        state.read_region(&region.name, &mut bytes)?;
        hash.update(region.name.as_bytes());
        hash.update(bytes);
    }
    Ok(hex::encode(hash.finalize()))
}
fn comparison(actual: &[u16], expected: &[u16], width: usize) -> Result<Value> {
    let f = |v: &[u16]| {
        v.iter()
            .map(|&w| f32::from_bits(u32::from(w) << 16))
            .collect::<Vec<_>>()
    };
    let mut result =
        crate::layer_comparison_reference::compare_partitioned(&f(actual), &f(expected), width)?;
    result["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    Ok(result)
}
fn check_head(
    ctx: &Context,
    module: &Module<'_>,
    draft: &resident_mtp::Model<'_, '_>,
    config: &DecoderConfig,
    r: &Reference,
) -> Result<Value> {
    let rows = r.shifted_tokens.len();
    ensure!(
        r.raw_target_hidden.len() == rows * config.hidden,
        "MTP reference hidden extent mismatch"
    );
    let raw = upload(ctx, &r.raw_target_hidden)?;
    let input = draft.target_hidden(ctx, module, &raw, rows)?;
    let mut whole = resident_mtp::Session::new(ctx, config)?;
    let (output, kernel_profile) = super::launch_profile::capture(ctx, || {
        draft.forward(ctx, module, &r.shifted_tokens, &input, &mut whole)
    })?;
    let hidden = words(&output.hidden)?;
    let hidden_report = comparison(&hidden, &r.hidden, config.hidden)?;
    let logits_report = comparison(&output.logits, &r.logits, config.vocabulary)?;
    let cache_report = check_cache(&whole.state, config, r)?;
    let mut split = resident_mtp::Session::new(ctx, config)?;
    let mut split_hidden = Vec::new();
    let mut last_logits = Vec::new();
    for (i, &token) in r.shifted_tokens.iter().enumerate() {
        let row = Buffer::new(ctx, config.hidden * 2)?;
        row.copy_from_at(0, &input, i * config.hidden * 2, config.hidden * 2)?;
        let out = draft.forward(ctx, module, &[token], &row, &mut split)?;
        split_hidden.extend(words(&out.hidden)?);
        last_logits = out.logits;
    }
    let partition = hidden == split_hidden
        && output.logits == last_logits
        && state_hash(&whole.state)? == state_hash(&split.state)?;
    let greedy = output.token == sampling::greedy(&r.logits)?;
    Ok(
        json!({"all_passed":hidden_report["all_passed"]==true && logits_report["all_passed"]==true && cache_report["all_passed"]==true && partition && greedy,
        "kernel_profile":kernel_profile,"hidden":hidden_report,"logits":logits_report,"cache":cache_report,"whole_token_partition_bit_exact":partition,"greedy_matches_reference":greedy}),
    )
}
fn check_cache(state: &ResidentState<'_>, config: &DecoderConfig, r: &Reference) -> Result<Value> {
    let width = config.attention_shape.kv_heads * config.attention_shape.head_width;
    let mut reports = Vec::new();
    for (suffix, expected) in [("k", &r.key), ("v", &r.value)] {
        let mut bytes = vec![0; config.capacity * width * 2];
        state.read_region(&format!("mtp.attention.{suffix}"), &mut bytes)?;
        let actual = ops::words(&bytes)?;
        let report = comparison(&actual[..expected.len()], expected, width)?;
        ensure!(
            actual[expected.len()..].iter().all(|&v| v == 0),
            "MTP wrote outside initialized cache prefix"
        );
        reports.push(report);
    }
    Ok(json!({"all_passed":reports.iter().all(|r|r["all_passed"]==true),"regions":reports}))
}

struct Control {
    tokens: Vec<u32>,
    state: String,
    past: usize,
    prefill: f64,
    decode: f64,
}
impl Control {
    fn report(&self) -> Value {
        json!({"tokens":self.tokens,"state_sha256":self.state,"past":self.past,"prefill_seconds":self.prefill,"decode_seconds":self.decode,"decode_tokens_per_second":(self.tokens.len()-1) as f64/self.decode})
    }
}
struct Runner<'a, 'w, 'ctx> {
    ctx: &'ctx Context,
    module: &'a Module<'ctx>,
    target: &'a Model<'w, 'ctx>,
    draft: &'a resident_mtp::Model<'w, 'ctx>,
    config: &'a DecoderConfig,
    request: &'a SpeculationRequest<'a>,
}
impl<'ctx> Runner<'_, '_, 'ctx> {
    fn control(&self) -> Result<Control> {
        let mut session = Session::new(self.ctx, self.config)?;
        let tick = Instant::now();
        let first = self.target.forward(
            self.ctx,
            self.module,
            self.request.tokens,
            &mut session,
            None,
        )?;
        let prefill = tick.elapsed().as_secs_f64();
        let mut tokens = vec![first.token];
        let tick = Instant::now();
        for _ in 1..self.request.output_tokens {
            let out = self.target.forward(
                self.ctx,
                self.module,
                &[*tokens.last().unwrap()],
                &mut session,
                None,
            )?;
            tokens.push(out.token);
        }
        let decode = tick.elapsed().as_secs_f64();
        Ok(Control {
            tokens,
            state: state_hash(&session.state)?,
            past: session.cursor.past(),
            prefill,
            decode,
        })
    }
    fn verification_profile(&self, control: &Control) -> Result<Value> {
        let mut session = Session::new(self.ctx, self.config)?;
        self.target.forward(
            self.ctx,
            self.module,
            self.request.tokens,
            &mut session,
            None,
        )?;
        let rows = (self.request.depth + 1).min(control.tokens.len() - 1);
        let (output, profile) = super::launch_profile::capture(self.ctx, || {
            self.target.forward_detailed(
                self.ctx,
                self.module,
                &control.tokens[..rows],
                &mut session,
                super::resident_model::LogitsSelection::All,
                None,
            )
        })?;
        ensure!(
            output.tokens == control.tokens[1..=rows],
            "profiled verification tokens differ from target-only decode"
        );
        Ok(
            json!({"rows":rows,"tokens_match_control":true,"kernel_profile":profile,
            "scope":"separate instrumented verification of a known target prefix; excluded from unprofiled timings"}),
        )
    }
    fn speculate(&self, forced: Option<u32>) -> Result<resident_speculation::Run<'ctx>> {
        resident_speculation::run(
            self.ctx,
            self.module,
            self.target,
            self.draft,
            self.config,
            &resident_speculation::Request {
                tokens: self.request.tokens,
                output_tokens: self.request.output_tokens,
                depth: self.request.depth,
                forced_first_draft: forced,
            },
        )
    }
}

fn compare_run(run: &resident_speculation::Run<'_>, control: &Control) -> Result<Value> {
    let hash = state_hash(&run.target_session.state)?;
    let exact = run.tokens == control.tokens
        && hash == control.state
        && run.target_session.cursor.past() == control.past;
    Ok(
        json!({"all_passed":exact,"tokens":run.tokens,"target_state_sha256":hash,"target_past":run.target_session.cursor.past(),
        "rounds":run.rounds,"all_accepted_rounds":run.all_accepted_rounds,"drafted":run.drafted,"accepted":run.accepted,"verify_rows":run.verify_rows,"replay_rows":run.replay_rows,
        "prefill_seconds":run.prefill_seconds,"decode_seconds":run.decode_seconds,"phase_seconds":run.phase_seconds,
        "decode_tokens_per_second":(run.tokens.len()-1) as f64/run.decode_seconds,
        "accepted_fraction":if run.drafted>0 {run.accepted as f64/run.drafted as f64}else{0.0},
        "speedup_over_control":control.decode/run.decode_seconds,
        "total_generation_seconds":run.prefill_seconds+run.decode_seconds,
        "total_generation_speedup":(control.prefill+control.decode)/(run.prefill_seconds+run.decode_seconds)}),
    )
}
