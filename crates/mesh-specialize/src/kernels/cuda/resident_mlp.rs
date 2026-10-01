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
    width: usize,
    nvfp4_schedule: crate::kernels::nvfp4_mlp_schedule::Schedule,
    workspace: Option<super::model_workspace::Shared<'ctx>>,
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
            width,
            nvfp4_schedule: crate::kernels::nvfp4_mlp_schedule::current()?,
            workspace: None,
        })
    }

    pub(super) fn attach_workspace(&mut self, workspace: super::model_workspace::Shared<'ctx>) {
        self.workspace = Some(workspace);
    }
    pub(super) fn has_workspace(&self) -> bool {
        self.workspace.is_some()
    }
    pub(super) fn workspace_output<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        let binding = |p: &Projection<'w, 'ctx>| match p {
            Projection::Fp8(p) => p.workspace_binding(),
            Projection::Nvfp4(p) => Ok(p.workspace_binding()),
            Projection::Bf16(_) => anyhow::bail!("BF16 MLP workspace is unsupported"),
        };
        let projections = [
            binding(&self.gate)?,
            binding(&self.up)?,
            binding(&self.down)?,
        ];
        let shared = self
            .workspace
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing model workspace"))?;
        let mut cache = shared
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("model workspace is already borrowed"))?;
        let shape = [rows, self.width, self.channels];
        let arena = cache.prepare(shape)?;
        super::resident_mlp_workspace::enqueue_chain(
            ctx,
            module,
            arena,
            &projections,
            input,
            shape,
            false,
        )?;
        arena.copy_region(ctx, "down.values")
    }

    pub(super) fn run<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<ResultBuffers<'a>> {
        self.run_with_past(ctx, module, input, rows, 0, false)
    }

    pub(super) fn run_with_past<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
        past: usize,
        decode: bool,
    ) -> Result<ResultBuffers<'a>> {
        let (gate, up, activation) = match (self.gate.nvfp4(), self.up.nvfp4()) {
            (Some(gate), Some(up)) if self.nvfp4_schedule.fuses_decode_row(decode, rows, past) => {
                let fused = super::resident_nvfp4_swiglu_a16::run(ctx, module, input, gate, up)?;
                (fused.gate, fused.up, fused.activation)
            }
            _ => {
                let shared_input = self.gate.shared_input(ctx, module, input, rows)?;
                let shared_input_ref = shared_input.as_ref();
                let gate = self
                    .gate
                    .run_with_input(ctx, module, input, rows, shared_input_ref)?;
                let up = self
                    .up
                    .run_with_input(ctx, module, input, rows, shared_input_ref)?;
                let activation = resident_activation::run(
                    ctx,
                    module,
                    &gate.values,
                    &up.values,
                    rows * self.channels,
                )?;
                (gate, up, activation)
            }
        };
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
