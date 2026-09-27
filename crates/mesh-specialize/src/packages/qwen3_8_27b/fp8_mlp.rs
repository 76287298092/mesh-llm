//! Last-eight-layer FP8 MLP qualification using synthetic normalized hidden input.

use crate::{
    artifact::reader::VerifiedArtifact,
    fp8_mlp_reference::{self, Projection, Weights},
    kernels::Fp8MlpCase,
};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

pub fn trial(path: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut artifact = VerifiedArtifact::open(path)?;
    let objects = super::schedule::text_objects(artifact.directory())?;
    let mut cases = Vec::new();
    for layer in [56, 63] {
        let prefix = format!("tensors/model.language_model.layers.{layer}.mlp");
        let weights = Weights {
            gate: load(&mut artifact, &prefix, "gate_proj", 17408)?,
            up: load(&mut artifact, &prefix, "up_proj", 17408)?,
            down: load(&mut artifact, &prefix, "down_proj", 5120)?,
        };
        for rows in [1, 17] {
            let input: Vec<_> = (0..rows * 5120)
                .map(|i| {
                    crate::entry_reference::round_bf16(
                        ((i * 17 + i / 5120 * 31) % 127) as f32 / 64.0 - 0.984375,
                    )
                })
                .collect();
            let reference = fp8_mlp_reference::run(&input, rows, 5120, &weights)?;
            cases.push(Fp8MlpCase {
                prefix: prefix.clone(),
                rows,
                width: 5120,
                channels: 17408,
                input,
                reference,
            });
        }
    }
    let mut report = crate::kernels::fp8_mlp_check(ptx, device, &mut artifact, &objects, &cases)?;
    report["identity"] = json!(artifact.identity());
    report["model_executable"] = json!(false);
    report["full_model_executed"] = json!(false);
    report["synthetic_normalized_hidden_input"] = json!(true);
    report["scope"] = json!(
        "Resident FP8 gate/up -> BF16 SiLU product -> FP8 down branches for layers 56 and 63. Attention, residual and full model are not executed in this trial."
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

fn load(
    artifact: &mut VerifiedArtifact,
    prefix: &str,
    name: &str,
    channels: usize,
) -> Result<Projection> {
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    artifact.copy_object(&format!("{prefix}.{name}.weight"), &mut weights)?;
    artifact.copy_object(&format!("{prefix}.{name}.weight_scale"), &mut scales)?;
    anyhow::ensure!(
        scales.len() == channels * 2,
        "FP8 MLP scale byte extent mismatch"
    );
    Ok(Projection {
        weights,
        scales: scales
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect(),
        channels,
    })
}
