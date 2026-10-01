//! Callable loaded-model measurements, deliberately not registered in a runtime.

mod accounting;
mod evidence;
mod execution;

use super::{
    driver::{Context, Module},
    resident_model::Model,
    resident_native_mtp_forward::Model as NativeModel,
};
use crate::kernels::DecoderConfig;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) use accounting::{EvidenceStatus, Prerequisites};

pub(super) struct LoadedModels<'a, 'w, 'ctx> {
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub target: &'a Model<'w, 'ctx>,
    pub draft: &'a NativeModel<'w, 'ctx>,
    pub config: &'a DecoderConfig,
}

pub(super) struct Request<'a> {
    pub prompt: &'a [u32],
    pub output_tokens: usize,
    pub warmups: usize,
    pub repetitions: usize,
    pub prerequisites: Prerequisites,
}

pub(super) fn run(models: &LoadedModels<'_, '_, '_>, request: &Request<'_>) -> Result<Value> {
    validate(models, request)?;
    let profiles = profiles()?;
    let mut warmups = Vec::with_capacity(request.warmups);
    for index in 0..request.warmups {
        warmups.push(matched(models, request, index + 1)?);
    }
    let mut repetitions = Vec::with_capacity(request.repetitions);
    for index in 0..request.repetitions {
        repetitions.push(matched(models, request, index + 1)?);
    }
    ensure!(
        profiles == self::profiles()?,
        "benchmark profiles changed during runs"
    );
    let matched_evidence = warmups
        .iter()
        .chain(&repetitions)
        .all(|sample| sample.matched);
    Ok(json!({
        "schema_version": 1, "kind": "loaded-target-versus-native-mtp-fixed-output",
        "completed": true, "device": models.context.info(), "profiles": profiles,
        "prompt_token_ids": request.prompt, "fixed_output_tokens": request.output_tokens,
        "capacity": models.config.capacity, "prerequisites": request.prerequisites,
        "warmup_count": request.warmups, "repetition_count": request.repetitions,
        "warmups": warmups, "repetitions": repetitions,
        "matched_evidence_passed": matched_evidence,
        "performance_qualified": request.prerequisites.qualifies(matched_evidence),
        "native_mtp_admitted": false,
        "method": {
            "baseline": "Model::forward prompt, then Model::forward_decode with decode=true for every generated input; host full logits greedy selection",
            "native": "resident_speculation::run_native, unforced depths 1 and 4, complete draft/verify/recovery/teacher/fork loop",
            "warmup": "full requested prompt and full fixed output generation for every route on disposable fresh sessions, before repetitions",
            "sessions": "fresh per route per warmup/repetition; same borrowed loaded weights/context/module/config and captured profiles",
            "timers": "Instant host seconds; context synchronize before and after outer intervals and baseline prefill/decode; native internal host timers returned unchanged",
            "device_elapsed_seconds": null,
            "decode_numerator": "actual emitted outputs minus first output selected during prefill",
            "eos_termination_ignored": true,
            "final_output_pending_unconsumed": true,
            "continuation": "one additional ordinary decode on each final target fork, using that route's pending final token; logits and every state region compared outside timing",
            "state": "SHA256 of all bytes in every named region, including inactive capacity; exact byte differences retained",
            "timing_excludes": ["weight loading", "post-run state readback/comparison", "continuation forks and decode", "session destruction"],
            "proposal_vocabulary_rows": 131072, "target_vocabulary": 248320,
            "proposal_mapping": "native forward returns mapped target token IDs; benchmark never reinterprets proposal row as a target ID",
            "native_serial_routes": "prefill above five rows is serial; native rows 2..4 are serial; no acceleration claim",
            "ninfer_measurement": null, "quality_pass_claim": false,
            "speedup_claim": false, "source_parity_claim": false,
            "admission": "measurement component only; no runtime admission decision"
        }
    }))
}

#[derive(serde::Serialize)]
struct Matched {
    index: usize,
    matched: bool,
    target: Value,
    native: Vec<Value>,
}

fn matched(
    models: &LoadedModels<'_, '_, '_>,
    request: &Request<'_>,
    index: usize,
) -> Result<Matched> {
    let baseline = execution::baseline(models, request)?;
    let target = execution::describe(models, &baseline)?;
    let mut native = Vec::with_capacity(2);
    let mut matched = baseline.tokens.len() == request.output_tokens
        && baseline.target.cursor.past() == request.prompt.len() + request.output_tokens - 1
        && !baseline.target.cursor.is_poisoned();
    for depth in [1, 4] {
        let candidate = execution::native(models, request, depth)?;
        let comparison = evidence::compare(models, &baseline, &candidate)?;
        matched &= comparison.matches();
        native.push(json!({"depth": depth, "run": execution::describe(models, &candidate)?, "comparison": comparison}));
        drop(candidate);
        models.context.synchronize()?;
    }
    drop(baseline);
    models.context.synchronize()?;
    Ok(Matched {
        index,
        matched,
        target,
        native,
    })
}

fn validate(models: &LoadedModels<'_, '_, '_>, request: &Request<'_>) -> Result<()> {
    let config = models.config;
    ensure!(
        config.layers.len() == 64 && config.vocabulary == 248_320,
        "benchmark requires the complete 64-layer target248320 model"
    );
    ensure!(
        (1..=2048).contains(&config.capacity),
        "benchmark capacity must be 1..=2048"
    );
    ensure!(
        (1..=512).contains(&request.prompt.len()),
        "prompt must contain 1..=512 tokens"
    );
    ensure!(
        (2..=128).contains(&request.output_tokens),
        "matched output must be 2..=128 tokens"
    );
    ensure!(
        (1..=3).contains(&request.repetitions) && (1..=3).contains(&request.warmups),
        "warmups and repetitions must be 1..=3"
    );
    ensure!(
        request.prompt.iter().all(|&token| token < 248_320),
        "prompt token outside target vocabulary"
    );
    let continuation_past = request.prompt.len().checked_add(request.output_tokens);
    ensure!(
        continuation_past.is_some_and(|past| past <= config.capacity),
        "fixed generation plus continuation exceeds capacity"
    );
    ensure!(
        models.target.belongs_to(models.context) && models.module.belongs_to(models.context),
        "loaded target/module context mismatch"
    );
    request.prerequisites.validate()
}

fn profiles() -> Result<Value> {
    Ok(json!({
        "fp8": crate::kernels::fp8_profile::current()?.name(),
        "attention": crate::kernels::attention_profile::current()?.name(),
        "nvfp4": crate::kernels::nvfp4_profile::current()?.name(),
        "nvfp4_decode": crate::kernels::nvfp4_decode_schedule::current()?.name(),
        "nvfp4_mlp": crate::kernels::nvfp4_mlp_schedule::current()?.name(),
        "ab": crate::kernels::ab_schedule::current()?.name(),
        "compact_recovery": super::resident_speculation::compact_recovery_enabled()?
    }))
}
