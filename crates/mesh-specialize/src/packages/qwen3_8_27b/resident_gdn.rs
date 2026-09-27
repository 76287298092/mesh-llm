//! Independent layer-zero validation of the reference-free resident decoder path.
use crate::{
    artifact::reader::VerifiedArtifact,
    kernels::{GdnShape, ResidentGdnCase, ResidentGdnConfig},
};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let input = super::projections::load_input(&mut artifact)?;
    let mut cases = Vec::new();
    for tokens in &input.entry.batches {
        cases.push(ResidentGdnCase {
            tokens: tokens.clone(),
            reference: crate::qwen_gdn_layer_reference::run(&input, tokens)?,
        });
    }
    drop(input);
    let config = ResidentGdnConfig {
        shape: GdnShape {
            hidden: 5120,
            key_heads: 16,
            value_heads: 48,
            head_width: 128,
            intermediate: 17408,
        },
        prefix: "tensors/model.language_model.layers.0".into(),
        state_prefix: "layers.00".into(),
        table_name: "tensors/model.language_model.embed_tokens.weight".into(),
        vocabulary: 248320,
        state_layout: super::schedule::Schedule::new(17)?.states,
    };
    let mut report =
        crate::kernels::resident_gdn_check(ptx, device, &mut artifact, &objects, &config, &cases)?;
    report["identity"] = json!(artifact.identity());
    report["model_executable"] = json!(false);
    report["full_model_executed"] = json!(false);
    report["scope"] = json!(
        "Resident layer zero with persistent convolution/recurrent state; whole/chunk/token execution. No later decoder layers or logits."
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
