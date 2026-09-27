//! Reference-free feed-forward execution using persistent FP8 weights.

use super::{
    driver::{Buffer, Context, Module},
    resident_activation,
    resident_fp8::{Output, Projection},
    resident_weights::ResidentWeights,
};
use anyhow::Result;

#[derive(Clone, Copy)]
pub(super) enum Quantization {
    Fp8,
    Nvfp4,
}

pub(super) enum Mlp<'w, 'ctx> {
    Fp8(Fp8Mlp<'w, 'ctx>),
    Nvfp4(Nvfp4Mlp<'w, 'ctx>),
}
impl<'w, 'ctx> Mlp<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
        quantization: Quantization,
    ) -> Result<Self> {
        Ok(match quantization {
            Quantization::Fp8 => Self::Fp8(Fp8Mlp::new(owner, prefix, width, channels)?),
            Quantization::Nvfp4 => Self::Nvfp4(Nvfp4Mlp::new(owner, prefix, width, channels)?),
        })
    }
    pub(super) fn run<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<ResultBuffers<'a>> {
        match self {
            Self::Fp8(mlp) => mlp.run(ctx, module, input, rows),
            Self::Nvfp4(mlp) => mlp.run(ctx, module, input, rows),
        }
    }
}

pub(super) struct Nvfp4Mlp<'w, 'ctx> {
    gate: super::resident_nvfp4::Projection<'w, 'ctx>,
    up: super::resident_nvfp4::Projection<'w, 'ctx>,
    down: super::resident_nvfp4::Projection<'w, 'ctx>,
    channels: usize,
}
impl<'w, 'ctx> Nvfp4Mlp<'w, 'ctx> {
    fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        use super::resident_nvfp4::Projection;
        Ok(Self {
            gate: Projection::new(owner, &format!("{prefix}.gate_proj"), width, channels)?,
            up: Projection::new(owner, &format!("{prefix}.up_proj"), width, channels)?,
            down: Projection::new(owner, &format!("{prefix}.down_proj"), channels, width)?,
            channels,
        })
    }
    fn run<'a>(
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

pub(super) struct Fp8Mlp<'w, 'ctx> {
    gate: Projection<'w, 'ctx>,
    up: Projection<'w, 'ctx>,
    down: Projection<'w, 'ctx>,
    channels: usize,
}

/// Device results remain owned until the caller consumes or validates them.
pub(super) struct ResultBuffers<'ctx> {
    pub(super) gate: Output<'ctx>,
    pub(super) up: Output<'ctx>,
    pub(super) activation: Buffer<'ctx>,
    pub(super) down: Output<'ctx>,
}

impl<'w, 'ctx> Fp8Mlp<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: Projection::new(owner, &format!("{prefix}.gate_proj"), width, channels)?,
            up: Projection::new(owner, &format!("{prefix}.up_proj"), width, channels)?,
            down: Projection::new(owner, &format!("{prefix}.down_proj"), channels, width)?,
            channels,
        })
    }

    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<ResultBuffers<'a>> {
        let gate = self.gate.run(context, module, input, rows)?;
        let up = self.up.run(context, module, input, rows)?;
        let activation = resident_activation::run(
            context,
            module,
            &gate.values,
            &up.values,
            rows * self.channels,
        )?;
        let down = self.down.run(context, module, &activation, rows)?;
        Ok(ResultBuffers {
            gate,
            up,
            activation,
            down,
        })
    }
}
