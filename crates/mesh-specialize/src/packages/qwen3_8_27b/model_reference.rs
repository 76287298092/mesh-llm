//! Independent whole-model CPU evidence, generated while the GPU service stays online.
use super::model_weights as weights;
use crate::{
    artifact::model_source::ModelArtifact, decoder_attention_reference, decoder_gdn_reference,
    decoder_ops_reference as ops, kernels::DecoderBlockKind, resident_entry_reference,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::{path::Path, time::Instant};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerDiagnostic {
    pub layer: usize,
    pub hidden: Vec<u16>,
    pub stages: BTreeMap<String, Vec<u16>>,
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<LayerDiagnostic>,
}

/// Enrich existing independent model evidence with one layer's scalar boundaries.
pub fn add_layer_diagnostic(
    path: &Path,
    reference: &mut ModelReference,
    layer: usize,
) -> Result<()> {
    ensure!(
        reference.schema_version == 1 && (1..=17).contains(&reference.tokens.len()),
        "invalid diagnostic model reference"
    );
    let mut artifact = ModelArtifact::open(path)?;
    artifact.require_legacy_reference()?;
    super::inventory::validate(artifact.directory())?;
    ensure!(
        reference.model_id == artifact.identity().model_id
            && reference.weights_id == artifact.identity().weights_id,
        "diagnostic reference identity mismatch"
    );
    let config = super::decoder::config(reference.tokens.len())?;
    ensure!(
        layer > 0 && layer < config.layers.len(),
        "diagnostic requires layer 1..63"
    );
    ensure!(
        reference.layer_outputs.len() == config.layers.len(),
        "diagnostic layer count mismatch"
    );
    let hidden = reference.layer_outputs[layer - 1].clone();
    let mut stages = BTreeMap::new();
    let mut observer = |name: &str, values: &[u16]| {
        stages.insert(name.to_owned(), values.to_vec());
        Ok(())
    };
    let output = match config.layers[layer].block {
        DecoderBlockKind::Gdn => decoder_gdn_reference::run_observed(
            &hidden,
            reference.tokens.len(),
            &config.gdn_shape,
            &weights::gdn(&mut artifact, layer)?,
            &mut observer,
        )?,
        DecoderBlockKind::Attention => decoder_attention_reference::run_observed(
            &hidden,
            reference.tokens.len(),
            &config.attention_shape,
            &weights::attention(&mut artifact, layer)?,
            &mut observer,
        )?,
    };
    ensure!(
        output == reference.layer_outputs[layer],
        "diagnostic recomputation differs from saved CPU output"
    );
    reference.diagnostic = Some(LayerDiagnostic {
        layer,
        hidden,
        stages,
    });
    Ok(())
}

pub fn trial(
    path: &Path,
    reference: &ModelReference,
    ptx: &str,
    device: i32,
) -> Result<serde_json::Value> {
    let mut artifact = ModelArtifact::open(path)?;
    artifact.require_legacy_reference()?;
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
    let mut artifact = ModelArtifact::open(path)?;
    artifact.require_legacy_reference()?;
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
        diagnostic: None,
    })
}
