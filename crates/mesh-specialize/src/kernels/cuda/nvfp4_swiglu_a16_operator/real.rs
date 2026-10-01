use super::super::driver::{Context, Module};
use super::super::resident_nvfp4::Projection as ResidentNvfp4Projection;
use super::super::resident_weights::ResidentWeights;
use super::compare;
use super::types::{Case, Weight};
use crate::entry_reference::round_bf16;
use anyhow::{Context as _, Result};
use serde_json::Value;

const PREFIX: &str = "tensors/model.language_model.layers.0.mlp";

pub(super) fn run(
    context: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
) -> Result<Value> {
    let width = 5120;
    let channels = 17408;
    let gate = bind(weights, &format!("{PREFIX}.gate_proj"), width, channels)?;
    let up = bind(weights, &format!("{PREFIX}.up_proj"), width, channels)?;
    let input: Vec<u16> = (0..width)
        .map(|index| round_bf16(((index * 37 % 257) as f32 - 128.0) / 64.0))
        .collect();
    compare::run(
        context,
        module,
        Case {
            source: "verified-layer-zero-packed-weights-with-deterministic-BF16-input",
            input: &input,
            width,
            gate: Weight {
                address: gate.address,
                packed: &gate.packed,
                scales: &gate.scales,
                divisor: gate.divisor,
            },
            up: Weight {
                address: up.address,
                packed: &up.packed,
                scales: &up.scales,
                divisor: up.divisor,
            },
        },
    )
}

struct ResidentWeight {
    address: [u64; 2],
    packed: Vec<u8>,
    scales: Vec<u8>,
    divisor: f32,
}

fn bind(
    weights: &ResidentWeights<'_>,
    prefix: &str,
    width: usize,
    channels: usize,
) -> Result<ResidentWeight> {
    let projection = ResidentNvfp4Projection::new(weights, prefix, width, channels)?;
    let packed_name = format!("{prefix}.weight_packed");
    let scale_name = format!("{prefix}.weight_scale");
    let mut packed = vec![0; channels * width / 2];
    let mut scales = vec![0; channels * width / 16];
    weights.read_range(&packed_name, 0, &mut packed)?;
    weights.read_range(&scale_name, 0, &mut scales)?;
    Ok(ResidentWeight {
        address: [projection.weight_pointer, projection.scale_pointer],
        packed,
        scales,
        divisor: weights
            .positive_scalar(&format!("{prefix}.weight_global_scale"))
            .context("missing verified NVFP4 weight global scale")?,
    })
}
