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
        let shared_input = self.gate.shared_input(ctx, module, input, rows)?;
        let shared_input_ref = shared_input.as_ref();
        let gate = self
            .gate
            .run_with_input(ctx, module, input, rows, shared_input_ref)?;
        let up = self
            .up
            .run_with_input(ctx, module, input, rows, shared_input_ref)?;
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
