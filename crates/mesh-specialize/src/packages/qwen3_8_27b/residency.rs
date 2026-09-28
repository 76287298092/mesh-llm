//! Persistent text-weight and state-capacity qualification, before model execution.

use super::schedule::{Schedule, text_objects};
use crate::{artifact::model_source::ModelArtifact, entry_reference, kernels::ResidentEntryInput};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

const TABLE: &str = "tensors/model.language_model.embed_tokens.weight";
const NORM: &str = "tensors/model.language_model.layers.0.input_layernorm.weight";

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = ModelArtifact::open(path)?;
    let objects = text_objects(artifact.directory())?;
    let schedule = Schedule::new(131_072)?;
    let tokens = vec![0, 248_319];
    let rows =
        crate::resident_entry_reference::embedding_rows(&mut artifact, TABLE, 5120, &tokens)?;
    let mut norm = Vec::new();
    artifact.copy_object(NORM, &mut norm)?;
    let reference = entry_reference::embedding_norm(&rows, &[0, 1], &norm, 5120, 1e-6)?;
    let entry = ResidentEntryInput {
        table_name: TABLE.into(),
        norm_name: NORM.into(),
        tokens,
        width: 5120,
        epsilon: 1e-6,
        reference,
    };
    let mut report = crate::kernels::residency_check(
        ptx,
        device,
        &mut artifact,
        &objects,
        &schedule.states,
        &entry,
    )?;
    report["identity"] = json!(artifact.identity());
    report["schedule"] = json!(schedule);
    report["allocated_context_capacity"] = json!(schedule.context_capacity);
    report["mtp_weights_loaded"] = json!(false);
    report["model_executable"] = json!(false);
    report["full_model_executed"] = json!(false);
    report["context_note"] =
        json!("BF16 KV capacity allocated and zero checked; no inference at this context length");
    for name in [
        "model_prefill_tokens_per_second",
        "model_decode_tokens_per_second",
        "model_context_tokens",
    ] {
        report[name] = Value::Null;
    }
    Ok(report)
}
