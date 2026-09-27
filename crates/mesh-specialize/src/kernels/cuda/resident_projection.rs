//! Quantization-neutral owner for the resident linear projection variants.
use super::{
    driver::{Buffer, Context, Module},
    resident_bf16, resident_fp8, resident_nvfp4,
    resident_weights::ResidentWeights,
};
use anyhow::Result;

#[derive(Clone, Copy)]
pub(super) enum Quantization {
    Fp8,
    Nvfp4,
    Bf16,
}

pub(super) enum Projection<'w, 'ctx> {
    Fp8(resident_fp8::Projection<'w, 'ctx>),
    Nvfp4(resident_nvfp4::Projection<'w, 'ctx>),
    Bf16(resident_bf16::Projection<'w, 'ctx>),
}

impl<'w, 'ctx> Projection<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
        quantization: Quantization,
    ) -> Result<Self> {
        Ok(match quantization {
            Quantization::Fp8 => Self::Fp8(resident_fp8::Projection::new(
                owner, prefix, width, channels,
            )?),
            Quantization::Nvfp4 => Self::Nvfp4(resident_nvfp4::Projection::new(
                owner, prefix, width, channels,
            )?),
            Quantization::Bf16 => Self::Bf16(resident_bf16::Projection::new(
                owner,
                &format!("{prefix}.weight"),
                width,
                channels,
            )?),
        })
    }

    pub(super) fn run<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<resident_fp8::Output<'a>> {
        match self {
            Self::Fp8(projection) => projection.run(ctx, module, input, rows),
            Self::Nvfp4(projection) => projection.run(ctx, module, input, rows),
            Self::Bf16(projection) => projection.run(ctx, module, input, rows),
        }
    }
}
