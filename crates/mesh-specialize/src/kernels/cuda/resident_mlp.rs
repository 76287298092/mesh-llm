//! Feed-forward execution using quantized resident projections.

use super::{
    driver::{Buffer, Context, Module},
    resident_activation,
    resident_fp8::Output,
    resident_projection::{Projection, Quantization},
    resident_weights::ResidentWeights,
};
use anyhow::Result;

pub(super) struct Mlp<'w, 'ctx> {
    gate: Projection<'w, 'ctx>,
    up: Projection<'w, 'ctx>,
    down: Projection<'w, 'ctx>,
    channels: usize,
}

impl<'w, 'ctx> Mlp<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
        quantization: Quantization,
    ) -> Result<Self> {
        Ok(Self {
            gate: Projection::new(
                owner,
                &format!("{prefix}.gate_proj"),
                width,
                channels,
                quantization,
            )?,
            up: Projection::new(
                owner,
                &format!("{prefix}.up_proj"),
                width,
                channels,
                quantization,
            )?,
            down: Projection::new(
                owner,
                &format!("{prefix}.down_proj"),
                channels,
                width,
                quantization,
            )?,
            channels,
        })
    }

    pub(super) fn run<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<ResultBuffers<'a>> {
        let gate = self.gate.run(ctx, module, input, rows)?;
        let up = self.up.run(ctx, module, input, rows)?;
        let activation =
            resident_activation::run(ctx, module, &gate.values, &up.values, rows * self.channels)?;
        let down = self.down.run(ctx, module, &activation, rows)?;
        Ok(ResultBuffers {
            gate,
            up,
            activation,
            down,
        })
    }
}

/// Device results remain owned until the caller consumes or validates them.
pub(super) struct ResultBuffers<'ctx> {
    pub(super) gate: Output<'ctx>,
    pub(super) up: Output<'ctx>,
    pub(super) activation: Buffer<'ctx>,
    pub(super) down: Output<'ctx>,
}
