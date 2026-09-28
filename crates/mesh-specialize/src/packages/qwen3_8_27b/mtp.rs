//! Independent MTP reference and bounded speculative execution qualification.
use super::{model_reference::ModelReference, model_weights as weights};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    decoder_ops_reference as ops,
    engine::sampling,
    kernels::SpeculationRequest,
    mtp_reference, resident_entry_reference,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub schema_version: u32,
    pub model_id: String,
    pub weights_id: String,
    pub shifted_tokens: Vec<u32>,
    pub raw_target_hidden: Vec<u16>,
    pub hidden: Vec<u16>,
    pub logits: Vec<u16>,
    pub key: Vec<u16>,
    pub value: Vec<u16>,
}

pub fn reference(path: &Path, target: &ModelReference) -> Result<Reference> {
    ensure!(
        target.schema_version == 1
            && (1..=17).contains(&target.tokens.len())
            && target.layer_outputs.len() == 64,
        "invalid target reference for MTP"
    );
    let mut artifact = ModelArtifact::open(path)?;
    artifact.require_mtp()?;
    super::inventory::validate(artifact.directory())?;
    ensure!(
        target.model_id == artifact.identity().model_id
            && target.weights_id == artifact.identity().weights_id,
        "MTP target reference identity mismatch"
    );
    let config = super::decoder::config(target.tokens.len())?;
    let raw_target_hidden = target.layer_outputs[63].clone();
    let mut shifted_tokens = target.tokens[1..].to_vec();
    shifted_tokens.push(sampling::greedy(&target.logits)?);
    let weights = load_weights(&mut artifact, &shifted_tokens)?;
    let output = mtp_reference::run(
        &raw_target_hidden,
        shifted_tokens.len(),
        &config.attention_shape,
        &weights,
    )?;
    Ok(Reference {
        schema_version: 1,
        model_id: target.model_id.clone(),
        weights_id: target.weights_id.clone(),
        shifted_tokens,
        raw_target_hidden,
        hidden: output.hidden,
        logits: output.logits,
        key: output.key,
        value: output.value,
    })
}

fn load_weights(artifact: &mut ModelArtifact, tokens: &[u32]) -> Result<mtp_reference::Weights> {
    let prefix = "tensors/mtp.layers.0";
    let attention = format!("{prefix}.self_attn");
    Ok(mtp_reference::Weights {
        target_norm: weights::words(artifact, "tensors/model.language_model.norm.weight")?,
        embedding_rows: ops::words(&resident_entry_reference::embedding_rows(
            artifact,
            "tensors/model.language_model.embed_tokens.weight",
            5120,
            tokens,
        )?)?,
        pre_embedding_norm: weights::words(artifact, "tensors/mtp.pre_fc_norm_embedding.weight")?,
        pre_hidden_norm: weights::words(artifact, "tensors/mtp.pre_fc_norm_hidden.weight")?,
        fc: weights::bf16(artifact, "tensors/mtp.fc", 5120)?,
        input_norm: weights::words(artifact, &format!("{prefix}.input_layernorm.weight"))?,
        post_norm: weights::words(
            artifact,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        q: weights::bf16(artifact, &format!("{attention}.q_proj"), 12288)?,
        k: weights::bf16(artifact, &format!("{attention}.k_proj"), 1024)?,
        v: weights::bf16(artifact, &format!("{attention}.v_proj"), 1024)?,
        out: weights::bf16(artifact, &format!("{attention}.o_proj"), 5120)?,
        q_norm: weights::words(artifact, &format!("{attention}.q_norm.weight"))?,
        k_norm: weights::words(artifact, &format!("{attention}.k_norm.weight"))?,
        gate: weights::bf16(artifact, &format!("{prefix}.mlp.gate_proj"), 17408)?,
        up: weights::bf16(artifact, &format!("{prefix}.mlp.up_proj"), 17408)?,
        down: weights::bf16(artifact, &format!("{prefix}.mlp.down_proj"), 5120)?,
        final_norm: weights::words(artifact, "tensors/mtp.norm.weight")?,
        head: weights::fp8(artifact, "tensors/lm_head", 248320)?,
    })
}

fn objects(artifact: &ModelArtifact) -> Result<Vec<Object>> {
    let mut objects = super::schedule::text_objects(artifact.directory())?;
    let mtp = artifact
        .directory()
        .objects
        .iter()
        .filter(|o| o.name.starts_with("tensors/mtp."));
    objects.extend(mtp.cloned());
    ensure!(objects.len() == 1635, "MTP tensor inventory mismatch");
    objects.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(objects)
}

pub fn trial(
    path: &Path,
    ptx: &str,
    device: i32,
    reference: &Reference,
    request: &SpeculationRequest<'_>,
) -> Result<serde_json::Value> {
    ensure!(
        reference.schema_version == 1 && (1..=17).contains(&reference.shifted_tokens.len()),
        "invalid MTP reference"
    );
    ensure!(
        (1..=512).contains(&request.tokens.len())
            && (3..=128).contains(&request.output_tokens)
            && (1..=4).contains(&request.depth)
            && (1..=3).contains(&request.repetitions),
        "MTP request exceeds bounds"
    );
    let mut artifact = ModelArtifact::open(path)?;
    artifact.require_mtp()?;
    super::inventory::validate(artifact.directory())?;
    ensure!(
        reference.model_id == artifact.identity().model_id
            && reference.weights_id == artifact.identity().weights_id,
        "MTP reference identity mismatch"
    );
    let objects = objects(&artifact)?;
    let config = super::decoder::config(
        (request.tokens.len() + request.output_tokens + request.depth)
            .max(reference.shifted_tokens.len()),
    )?;
    let mut report = crate::kernels::mtp_trial(
        ptx,
        device,
        &mut artifact,
        &objects,
        &config,
        reference,
        request,
    )?;
    report["identity"] = serde_json::json!(artifact.identity());
    Ok(report)
}
