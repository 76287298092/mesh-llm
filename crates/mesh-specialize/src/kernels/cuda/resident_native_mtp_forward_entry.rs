//! Bounded real-target inputs for native repeat/state consistency, not admission.

mod inputs;
mod preflight;

use super::{
    driver::{Context, Module},
    resident_model::Model as TargetModel,
    resident_native_mtp::ResidentNativeMtp,
    resident_native_mtp_forward::Model as NativeModel,
    resident_native_mtp_forward_trial,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    kernels::DecoderConfig,
    packages::qwen3_8_27b::target_batch_trial::Fixture,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) struct Request<'a> {
    pub artifact: &'a mut ModelArtifact,
    pub objects: &'a [Object],
    pub config: &'a DecoderConfig,
    pub ptx: &'a str,
    pub device: i32,
    pub fixture: &'a Fixture,
}

pub(in crate::kernels) fn run(mut request: Request<'_>) -> Result<Value> {
    let result = execute(&mut request);
    let mut report = match result {
        Ok(report) => report,
        Err(error) => json!({"all_passed": false, "error": format!("{error:#}")}),
    };
    report["identity"] = json!(request.artifact.identity());
    report["model_source"] = request.artifact.verification_report();
    report["fixture_alignment"] = inputs::alignment_report(request.fixture);
    for claim in [
        "source_arithmetic_qualified",
        "native_mtp_admitted",
        "model_executable",
        "timing_claim",
    ] {
        report[claim] = json!(false);
    }
    report["native_base_prefill"] = json!(
        "complete sequential T1 draft steps; correctness-only serial prefill, no batched speedup"
    );
    report["hidden_source_contract"] = json!(
        "canonical target DetailedOutput.hidden is the raw final decoder residual, before the target head final norm; native target_hidden applies that final norm. Field semantics are traced, not independent source arithmetic equality."
    );
    Ok(report)
}

fn execute(request: &mut Request<'_>) -> Result<Value> {
    preflight::validate(request)?;
    match &mut *request.artifact {
        ModelArtifact::Ninfer(source) => {
            source.native_mtp_views()?;
        }
        ModelArtifact::Mspec(_) => {
            anyhow::bail!("native forward trial requires a verified NInfer source")
        }
    }
    let context = Context::new(request.device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "native forward trial requires SM120"
    );
    let module = Module::load(&context, request.ptx)?;
    preflight::functions(&module)?;
    let weights = ResidentWeights::load(&context, request.artifact, request.objects)?;
    let native = match &mut *request.artifact {
        ModelArtifact::Ninfer(source) => ResidentNativeMtp::load(&context, source)?,
        ModelArtifact::Mspec(_) => {
            anyhow::bail!("native forward trial requires a verified NInfer source")
        }
    };
    let target = TargetModel::new(&weights, request.config)?;
    let draft = NativeModel::new(&weights, &native, request.config)?;
    let engines = inputs::Engines {
        context: &context,
        module: &module,
        target: &target,
        draft: &draft,
        config: request.config,
    };
    let prepared = engines.prepare(request.fixture)?;
    let mut report =
        resident_native_mtp_forward_trial::run(resident_native_mtp_forward_trial::Request {
            model: &draft,
            context: &context,
            module: &module,
            config: request.config,
            base: &prepared.base,
            tokens: &prepared.tokens,
            normalized_target_hidden: &prepared.hidden,
            continuation_token: request.fixture.continuation[1],
            normalized_continuation_hidden: &prepared.continuation_hidden,
            proposal_tokens: &native.views().proposal_tokens,
        })?;
    report["device"] = json!(info);
    report["native_parent_verification"] = json!(native.layout().regions.iter().map(|region| {
        let parent = native.parent(&region.name)?;
        Ok(json!({"object_id": parent.object_id(), "bytes": parent.bytes(), "sha256": parent.sha256()}))
    }).collect::<Result<Vec<_>>>()?);
    report["attention_profile"] = json!(crate::kernels::attention_profile::current()?.name());
    report["nvfp4_mlp_schedule"] = json!(crate::kernels::nvfp4_mlp_schedule::current()?.name());
    Ok(report)
}
