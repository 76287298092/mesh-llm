//! Independent whole-model CPU evidence, generated while the GPU service stays online.
use super::model_weights as weights;
use crate::{
    artifact::reader::VerifiedArtifact, decoder_attention_reference, decoder_gdn_reference,
    decoder_ops_reference as ops, kernels::DecoderBlockKind, resident_entry_reference,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelReference {
    pub schema_version: u32,
    pub model_id: String,
    pub weights_id: String,
    pub tokens: Vec<u32>,
    pub layer_outputs: Vec<Vec<u16>>,
    pub logits: Vec<u16>,
    pub elapsed_seconds: f64,
    pub layer_seconds: Vec<f64>,
}

pub fn trial(
    path: &Path,
    reference: &ModelReference,
    ptx: &str,
    device: i32,
) -> Result<serde_json::Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let config = super::decoder::config(reference.tokens.len())?;
    let mut report =
        crate::kernels::model_check(ptx, device, &mut artifact, &objects, &config, reference)?;
    report["identity"] = serde_json::json!(artifact.identity());
    Ok(report)
}

pub fn run(
    path: &Path,
    tokens: &[u32],
    mut progress: impl FnMut(usize, f64) -> Result<()>,
) -> Result<ModelReference> {
    ensure!(
        (1..=17).contains(&tokens.len()),
        "model reference is bounded to 1..17 tokens"
    );
    let start = Instant::now();
    let mut artifact = VerifiedArtifact::open(path)?;
    super::inventory::validate(artifact.directory())?;
    let config = super::decoder::config(tokens.len())?;
    let mut hidden = ops::words(&resident_entry_reference::embedding_rows(
        &mut artifact,
        &config.embedding_table,
        config.hidden,
        tokens,
    )?)?;
    let mut layer_outputs = Vec::new();
    let mut layer_seconds = Vec::new();
    for (index, layer) in config.layers.iter().enumerate() {
        let tick = Instant::now();
        hidden = match layer.block {
            DecoderBlockKind::Gdn => decoder_gdn_reference::run(
                &hidden,
                tokens.len(),
                &config.gdn_shape,
                &weights::gdn(&mut artifact, index)?,
            )?,
            DecoderBlockKind::Attention => decoder_attention_reference::run(
                &hidden,
                tokens.len(),
                &config.attention_shape,
                &weights::attention(&mut artifact, index)?,
            )?,
        };
        layer_outputs.push(hidden.clone());
        layer_seconds.push(tick.elapsed().as_secs_f64());
        progress(index, tick.elapsed().as_secs_f64())?;
    }
    let final_row = &hidden[(tokens.len() - 1) * config.hidden..];
    let norm = weights::words(&mut artifact, &config.final_norm)?;
    let normalized = ops::normalize(final_row, &norm, 1, config.hidden)?;
    let logits = ops::fp8(
        &normalized,
        &weights::fp8(&mut artifact, &config.head_prefix, config.vocabulary)?,
        1,
        config.hidden,
    )?;
    Ok(ModelReference {
        schema_version: 1,
        model_id: artifact.identity().model_id.clone(),
        weights_id: artifact.identity().weights_id.clone(),
        tokens: tokens.to_vec(),
        layer_outputs,
        logits,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        layer_seconds,
    })
}
