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

    pub(super) fn fp8(&self) -> Option<&resident_fp8::Projection<'w, 'ctx>> {
        match self {
            Self::Fp8(projection) => Some(projection),
            Self::Nvfp4(_) | Self::Bf16(_) => None,
        }
    }

    pub(super) fn nvfp4(&self) -> Option<&resident_nvfp4::Projection<'w, 'ctx>> {
        match self {
            Self::Nvfp4(projection) => Some(projection),
            Self::Fp8(_) | Self::Bf16(_) => None,
        }
    }

    pub(super) fn shared_input<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Option<resident_fp8::QuantizedInput<'a>>> {
        match self.fp8() {
            Some(projection) => projection.shared_input(ctx, module, input, rows),
            None => Ok(None),
        }
    }

    pub(super) fn run_with_input<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
        shared: Option<&resident_fp8::QuantizedInput<'_>>,
    ) -> Result<resident_fp8::Output<'a>> {
        match self.fp8() {
            Some(projection) => projection.run_with_input(ctx, module, input, rows, shared),
            None => self.run(ctx, module, input, rows),
        }
    }
}
