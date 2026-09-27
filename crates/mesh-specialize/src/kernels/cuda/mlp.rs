//! Resident NVFP4 feed-forward path and second decoder residual boundary.
use super::{
    driver::{Context, Module},
    mlp_activation, nvfp4_linear, residual_add,
    residual_norm::CheckedNorm,
};
use crate::{kernels::Nvfp4Mlp, qwen_attention_layer_reference::Stage};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn validate(weights: &Nvfp4Mlp, width: usize) -> Result<()> {
    nvfp4_linear::validate(&weights.gate, width)?;
    nvfp4_linear::validate(&weights.up, width)?;
    ensure!(
        weights.gate.channels == weights.up.channels,
        "MLP gate/up channels differ"
    );
    nvfp4_linear::validate(&weights.down, weights.gate.channels)?;
    ensure!(
        weights.down.channels == width,
        "MLP down channels differ from hidden width"
    );
    Ok(())
}
pub(super) struct CheckedMlp {
    pub(super) stages: Vec<Stage>,
    pub(super) words: Vec<u16>,
    pub(super) report: Value,
}
pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: &CheckedNorm<'_>,
    weights: &Nvfp4Mlp,
    shape: [usize; 2],
) -> Result<CheckedMlp> {
    validate(weights, shape[1])?;
    let gate = nvfp4_linear::check(
        context,
        module,
        &input.normalized,
        &input.words,
        &weights.gate,
        shape,
    )?;
    let up = nvfp4_linear::check(
        context,
        module,
        &input.normalized,
        &input.words,
        &weights.up,
        shape,
    )?;
    let activated = mlp_activation::check(
        context,
        module,
        mlp_activation::Input {
            gate: &gate.output,
            gate_words: &gate.words,
            up: &up.output,
            up_words: &up.words,
        },
    )?;
    let down = nvfp4_linear::check(
        context,
        module,
        &activated.output,
        &activated.words,
        &weights.down,
        [shape[0], weights.gate.channels],
    )?;
    let residual = residual_add::check(
        context,
        module,
        residual_add::Input {
            residual: &input.residual,
            residual_words: &input.residual_words,
            branch: &down.output,
            branch_words: &down.words,
        },
    )?;
    let report = json!({"all_passed":true,"gate":gate.report,"up":up.report,"activation":activated.report,"down":down.report,"residual":residual.report,
        "device_intermediates_resident":true,"component_checks_only":true,
        "scope":"decoder MLP GPU component chain complete; each component uses actual preceding device outputs for its scalar comparison; independent full-layer/logit comparison is reported separately by the layer harness"});
    let stages = vec![
        Stage {
            name: "mlp_gate",
            words: gate.words,
            width: weights.gate.channels,
        },
        Stage {
            name: "mlp_up",
            words: up.words,
            width: weights.up.channels,
        },
        Stage {
            name: "mlp_activation",
            words: activated.words,
            width: weights.gate.channels,
        },
        Stage {
            name: "mlp_down",
            words: down.words,
            width: shape[1],
        },
    ];
    Ok(CheckedMlp {
        stages,
        words: residual.words,
        report,
    })
}
