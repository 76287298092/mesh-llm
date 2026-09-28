//! Resident layer-three attention qualification from independent checkpoint inputs.
use crate::{
    artifact::model_source::ModelArtifact,
    kernels::{ResidentAttentionCase, ResidentAttentionConfig, ResidentAttentionShape},
};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = ModelArtifact::open(path)?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let input = super::attention::load_input(&mut artifact)?;
    let mut cases = Vec::new();
    for (tokens, positions) in input.entry.batches.iter().zip(&input.positions) {
        cases.push(ResidentAttentionCase {
            tokens: tokens.clone(),
            reference: crate::qwen_attention_layer_reference::run(&input, tokens, positions)?,
        });
    }
    drop(input);
    let config = ResidentAttentionConfig {
        shape: ResidentAttentionShape {
            hidden: 5120,
            intermediate: 17408,
            query_heads: 24,
            kv_heads: 4,
            head_width: 256,
            rotary_dim: 64,
            rope_theta: 1e7,
        },
        prefix: "tensors/model.language_model.layers.3".into(),
        state_prefix: "layers.03".into(),
        table_name: "tensors/model.language_model.embed_tokens.weight".into(),
        vocabulary: 248320,
        capacity: 20,
        state_layout: super::schedule::Schedule::new(20)?.states,
    };
    let mut report = crate::kernels::resident_attention_check(
        ptx,
        device,
        &mut artifact,
        &objects,
        &config,
        &cases,
    )?;
    report["identity"] = json!(artifact.identity());
    report["model_executable"] = json!(false);
    report["scope"] = json!(
        "Resident layer three using embedding rows as synthetic hidden input; no preceding layers or logits. Whole/chunk/token K/V persistence."
    );
    for key in [
        "model_prefill_tokens_per_second",
        "model_decode_tokens_per_second",
        "model_context_tokens",
    ] {
        report[key] = Value::Null;
    }
    Ok(report)
}
