//! Whole-model comparison against independently saved CPU layer/logit evidence.
use super::{
    driver::{Buffer, Context, Module},
    resident_model::{Model, Session},
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::{layout::Layout, sampling},
    entry_reference::bf16_to_f32,
    kernels::DecoderConfig,
    packages::qwen3_8_27b::model_reference::ModelReference,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &DecoderConfig,
    reference: &ModelReference,
) -> Result<Value> {
    validate_reference(artifact, config, reference)?;
    ensure!(
        ptx.contains(".target sm_120a"),
        "model trial requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "model trial requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    for name in [
        "embedding_norm_bf16",
        "fp8_quantize_bf16",
        "fp8_linear_exact",
        "bf16_linear_decode",
        "causal_conv4_bf16",
        "gdn_qk_norm",
        "gdn_gates",
        "gdn_recurrent",
        "gdn_gated_rms_norm",
        "residual_norm_bf16",
        "nvfp4_quantize_bf16",
        "nvfp4_linear",
        "mlp_silu_product",
        "residual_add_bf16",
        "attention_qk_prepare",
        "attention_kv_append",
        "causal_attention_bf16",
        "attention_gate_bf16",
    ] {
        module
            .function(name)
            .with_context(|| format!("load model kernel {name}"))?;
    }
    let before = context.memory()?;
    let attention_probes = super::attention_core::fixtures(&context, &module)?;
    ensure!(
        attention_probes.iter().all(|r| r["all_passed"] == true),
        "attention fixtures failed"
    );
    let nvfp4_tail_probes = super::nvfp4_linear::fixtures(&context, &module)?;
    let fp8_exact_probe = super::fp8_exact_trial::run(&context, &module)?;
    if fp8_exact_probe["all_passed"] != true {
        return Ok(
            json!({"all_passed":false,"full_model_executed":false,"fp8_exact_probe":fp8_exact_probe}),
        );
    }
    let activation_probe = super::silu_trial::run(&context, &module)?;
    if activation_probe["all_passed"] != true {
        return Ok(
            json!({"all_passed":false,"full_model_executed":false,"activation_probe":activation_probe}),
        );
    }
    let layout = Layout::new(objects.iter().map(|o| (o.name.clone(), o.length)))?;
    let bf16_probe = super::bf16_trial::run(&context, &module)?;
    if bf16_probe["all_passed"] != true {
        return Ok(
            json!({"all_passed":false,"full_model_executed":false,"bf16_probe":bf16_probe,"activation_probe":activation_probe}),
        );
    }
    let needed = layout
        .bytes
        .checked_add(config.state_layout.bytes)
        .and_then(|v| v.checked_add(1024 * 1024 * 1024))
        .context("model admission overflow")?;
    ensure!(
        needed <= u64::try_from(before.0)?,
        "insufficient model trial memory"
    );
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    if let Some(diagnostic) = &reference.diagnostic {
        let mut report = super::resident_layer_diagnostic::run(
            &context,
            &module,
            &weights,
            config,
            reference.tokens.len(),
            diagnostic,
        )?;
        report["activation_probe"] = activation_probe;
        report["bf16_probe"] = bf16_probe;
        return Ok(report);
    }
    let model = Model::new(&weights, config)?;
    let mut session = Session::new(&context, config)?;
    let mut layers = Vec::new();
    let mut observer = |index: usize, buffer: &Buffer<'_>| -> Result<()> {
        let actual = words(buffer)?;
        let mut result = compare(&actual, &reference.layer_outputs[index], config.hidden)?;
        result["layer"] = json!(index);
        layers.push(result);
        Ok(())
    };
    let output = model.forward(
        &context,
        &module,
        &reference.tokens,
        &mut session,
        Some(&mut observer),
    );
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return Ok(
                json!({"all_passed":false,"error":format!("{error:#}"),"layers":layers,"session_poisoned":session.cursor.is_poisoned(),"full_model_executed":false}),
            );
        }
    };
    let logits = compare(&output.logits, &reference.logits, config.vocabulary)?;
    let expected_token = sampling::greedy(&reference.logits)?;
    let whole_state = state_hash(&session)?;
    let memory = context.memory()?;
    let prefix_correct = output.past == reference.tokens.len() && !session.cursor.is_poisoned();
    drop(session);
    let mut split = Session::new(&context, config)?;
    let mut last = None;
    for &token in &reference.tokens {
        last = Some(model.forward(&context, &module, &[token], &mut split, None)?);
    }
    let last = last.context("empty token fixture")?;
    let partition_exact = last.logits == output.logits && state_hash(&split)? == whole_state;
    drop(split);
    let mut failed = Session::new(&context, config)?;
    let mut injected = |_index: usize, _buffer: &Buffer<'_>| -> Result<()> {
        anyhow::bail!("injected trial observer failure")
    };
    ensure!(
        model
            .forward(
                &context,
                &module,
                &reference.tokens[..1],
                &mut failed,
                Some(&mut injected)
            )
            .is_err(),
        "injected model error did not fail"
    );
    let poison_rejected = failed.cursor.is_poisoned()
        && model
            .forward(&context, &module, &reference.tokens[..1], &mut failed, None)
            .is_err();
    drop(failed);
    drop(model);
    drop(weights);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"qwen-resident-full-model-correctness","device":info,"tokens":reference.tokens,"layers":layers,"logits":logits,"activation_probe":activation_probe,"bf16_probe":bf16_probe,"fp8_exact_probe":fp8_exact_probe,"nvfp4_tail_probes":nvfp4_tail_probes,"attention_probes":attention_probes,
        "selected_token":output.token,"reference_token":expected_token,"greedy_token_exact":output.token==expected_token,
        "whole_vs_token_logits_and_state_bit_exact":partition_exact,"prefix_committed_correctly":prefix_correct,"failed_session_reuse_rejected":poison_rejected,
        "all_passed":layers.iter().all(|r|r["all_passed"]==true)&&logits["all_passed"]==true&&output.token==expected_token&&partition_exact&&prefix_correct&&poison_rejected&&after.0>=before.0,
        "full_model_executed":true,"model_layers_executed":config.layers.len(),"weight_arena_bytes":layout.bytes,"state_arena_bytes":config.state_layout.bytes,
        "memory_before":{"free_bytes":before.0,"total_bytes":before.1},"memory_after_sequence":{"free_bytes":memory.0,"total_bytes":memory.1},"memory_after_free":{"free_bytes":after.0,"total_bytes":after.1},
        "model_prefill_tokens_per_second":null,"model_decode_tokens_per_second":null,"model_context_tokens":null,
        "scope":"Correctness with hidden observers and state hashing; no model throughput measurement"}),
    )
}

fn validate_reference(
    artifact: &VerifiedArtifact,
    config: &DecoderConfig,
    reference: &ModelReference,
) -> Result<()> {
    ensure!(
        reference.schema_version == 1
            && reference.model_id == artifact.identity().model_id
            && reference.weights_id == artifact.identity().weights_id,
        "model reference identity mismatch"
    );
    ensure!(
        (1..=17).contains(&reference.tokens.len()) && reference.tokens.len() <= config.capacity,
        "invalid model reference token count"
    );
    ensure!(
        reference.layer_outputs.len() == config.layers.len()
            && reference
                .layer_outputs
                .iter()
                .all(|v| v.len() == reference.tokens.len() * config.hidden),
        "model reference layer extents mismatch"
    );
    ensure!(
        reference.logits.len() == config.vocabulary,
        "model reference vocabulary extent mismatch"
    );
    sampling::greedy(&reference.logits)?;
    Ok(())
}
fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    let mut raw = vec![0; buffer.len()];
    buffer.download(&mut raw)?;
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect())
}
fn compare(actual: &[u16], expected: &[u16], width: usize) -> Result<Value> {
    let decode = |v: &[u16]| v.iter().map(|&w| bf16_to_f32(w)).collect::<Vec<_>>();
    let mut result = crate::layer_comparison_reference::compare_partitioned(
        &decode(actual),
        &decode(expected),
        width,
    )?;
    result["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    Ok(result)
}
fn state_hash(session: &Session<'_>) -> Result<String> {
    let mut hash = Sha256::new();
    for region in &session.state.layout().regions {
        let mut bytes = vec![0; usize::try_from(region.length)?];
        session.state.read_region(&region.name, &mut bytes)?;
        hash.update(region.name.as_bytes());
        hash.update(bytes);
    }
    Ok(hex::encode(hash.finalize()))
}
