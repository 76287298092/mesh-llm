//! Diagnostic-only replay of one GDN block from independent CPU hidden input.
use super::{
    driver::{Buffer, Context, Module},
    resident_gdn,
    resident_mlp::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    entry_reference::bf16_to_f32,
    kernels::{DecoderBlockKind, DecoderConfig, DecoderMlpKind},
    packages::qwen3_8_27b::model_reference::GdnDiagnostic,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn run(
    ctx: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
    config: &DecoderConfig,
    rows: usize,
    diagnostic: &GdnDiagnostic,
) -> Result<Value> {
    ensure!(
        diagnostic.layer < config.layers.len(),
        "invalid diagnostic layer"
    );
    let descriptor = &config.layers[diagnostic.layer];
    ensure!(
        matches!(descriptor.block, DecoderBlockKind::Gdn),
        "diagnostic layer is not GDN"
    );
    ensure!(
        diagnostic.hidden.len() == rows * config.hidden,
        "diagnostic hidden extent mismatch"
    );
    let quantization = match descriptor.mlp {
        DecoderMlpKind::Nvfp4 => Quantization::Nvfp4,
        DecoderMlpKind::Fp8 => Quantization::Fp8,
    };
    let layer = resident_gdn::Layer::new(
        weights,
        &descriptor.prefix,
        &descriptor.state_prefix,
        &config.gdn_shape,
        quantization,
    )?;
    let raw = diagnostic
        .hidden
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let hidden = Buffer::new(ctx, raw.len())?;
    hidden.upload(&raw)?;
    let mut state = ResidentState::new(ctx, &config.state_layout)?;
    let mut stages = Vec::new();
    let mut observer = |name: &str, buffer: &Buffer<'_>| -> Result<()> {
        let expected = diagnostic
            .stages
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing diagnostic stage {name}"))?;
        let mut bytes = vec![0; buffer.len()];
        buffer.download(&mut bytes)?;
        let actual = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect::<Vec<_>>();
        ensure!(
            actual.len() == expected.len(),
            "diagnostic stage extent mismatch: {name}"
        );
        let floats = |words: &[u16]| words.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<_>>();
        let comparison =
            crate::layer_comparison_reference::compare(&floats(&actual), &floats(expected))?;
        let differences=actual.iter().zip(expected).enumerate().filter(|(_, (a,b))|a!=b).map(|(index,(&a,&b))|json!({"index":index,"actual_bits":a,"expected_bits":b,"actual":bf16_to_f32(a),"expected":bf16_to_f32(b)})).collect::<Vec<_>>();
        stages.push(json!({"name":name,"comparison":comparison,"bf16_differences":differences.len(),"first_differences":differences.iter().take(16).collect::<Vec<_>>() }));
        Ok(())
    };
    let _output =
        layer.forward_observed(ctx, module, &hidden, &mut state, rows, Some(&mut observer))?;
    ensure!(
        stages.len() == diagnostic.stages.len(),
        "unvisited diagnostic stage"
    );
    Ok(
        json!({"kind":"isolated-gdn-boundary-diagnostic","full_model_executed":false,"layer":diagnostic.layer,"stages":stages,
        "all_passed":stages.iter().all(|s|s["bf16_differences"]==0),"scope":"Isolated GDN block supplied with independent CPU hidden input, diagnosing numerical divergence; not model execution or performance"}),
    )
}
