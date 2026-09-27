//! Experimental exact MLP chain with persistent scratch and explicit completion policy.
use super::{
    driver::{Buffer, Context, Module},
    mlp_workspace_projection::Binding,
    resident_projection::{Projection, Quantization},
    resident_weights::ResidentWeights,
    resident_workspace::ResidentWorkspace,
};
use crate::engine::workspace::WorkspaceLayout;
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct Chain<'w, 'ctx> {
    context: &'ctx Context,
    workspace: ResidentWorkspace<'ctx>,
    projections: [Binding<'w, 'ctx>; 3],
    rows: usize,
    width: usize,
    channels: usize,
}
impl<'w, 'ctx> Chain<'w, 'ctx> {
    pub(super) fn new(
        ctx: &'ctx Context,
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        shape: [usize; 3],
        kind: Quantization,
    ) -> Result<Self> {
        let [rows, width, channels] = shape;
        ensure!(
            (1..=512).contains(&rows)
                && (1..=32768).contains(&width)
                && (1..=32768).contains(&channels),
            "invalid MLP workspace shape"
        );
        ensure!(owner.belongs_to(ctx), "workspace weight context mismatch");
        ensure!(
            crate::kernels::fp8_profile::current()? == crate::kernels::fp8_profile::Profile::Exact
                && super::resident_fp8_splitk::configured_splits()?.is_none(),
            "MLP workspace experiment requires exact profile and split-K off"
        );
        let binding = |suffix: &str, k, n| -> Result<Binding<'w, 'ctx>> {
            match Projection::new(owner, &format!("{prefix}.{suffix}"), k, n, kind)? {
                Projection::Fp8(p) => p.workspace_binding(),
                Projection::Nvfp4(p) => Ok(p.workspace_binding()),
                Projection::Bf16(_) => anyhow::bail!("workspace MLP supports FP8/NVFP4 only"),
            }
        };
        let projections = [
            binding("gate_proj", width, channels)?,
            binding("up_proj", width, channels)?,
            binding("down_proj", channels, width)?,
        ];
        let mut regions = Vec::new();
        for (name, p) in ["gate", "up", "down"].into_iter().zip(&projections) {
            regions.extend(p.regions(name, rows));
        }
        for (name, bytes) in [
            ("activation.values", rows * channels * 2),
            ("activation.silu", rows * channels * 4),
            ("activation.activated", rows * channels * 2),
            ("activation.raw", rows * channels * 4),
        ] {
            regions.push((name.to_owned(), bytes));
        }
        let workspace =
            ResidentWorkspace::new(ctx, WorkspaceLayout::new(512 * 1024 * 1024, regions)?)?;
        Ok(Self {
            context: ctx,
            workspace,
            projections,
            rows,
            width,
            channels,
        })
    }
    /// Keeps arithmetic, quantization and diagnostics identical. `operator_waits`
    /// isolates persistent storage from the separate stream-ordering experiment.
    pub(super) fn run(
        &mut self,
        module: &Module<'_>,
        input: &Buffer<'_>,
        operator_waits: bool,
    ) -> Result<()> {
        ensure!(
            module.belongs_to(self.context)
                && input.belongs_to(self.context)
                && input.len() == self.rows * self.width * 2,
            "MLP workspace input/context mismatch"
        );
        let step = self.workspace.begin_step()?;
        for (name, projection) in ["gate", "up"].into_iter().zip(&self.projections) {
            // SAFETY: Checked input/context and this chain's fixed-shape regions remain
            // live under the exclusive lease through completion or error draining.
            unsafe {
                projection.enqueue(
                    self.context,
                    module,
                    &step,
                    name,
                    input.pointer(),
                    self.rows,
                )?;
            }
            if operator_waits {
                self.context.synchronize()?;
            }
        }
        let names = [
            "gate.values",
            "up.values",
            "activation.values",
            "activation.silu",
            "activation.activated",
            "activation.raw",
        ];
        let mut pointers = names
            .iter()
            .map(|name| Ok(step.region(name)?.pointer()))
            .collect::<Result<Vec<_>>>()?;
        let mut count = (self.rows * self.channels) as u32;
        let mut args = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        args.push((&mut count as *mut u32).cast());
        // SAFETY: Planned disjoint BF16/FP32 regions match the six-pointer activation ABI.
        // The same stream orders both projections before this consumer; lease drains errors.
        unsafe {
            module.function("mlp_silu_product")?.launch(
                [count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        if operator_waits {
            self.context.synchronize()?;
        }
        // SAFETY: Activation was queued on the same ordered stream into the planned
        // BF16 region; downstream scratch is disjoint and the lease remains live.
        unsafe {
            self.projections[2].enqueue(
                self.context,
                module,
                &step,
                "down",
                step.region("activation.values")?.pointer(),
                self.rows,
            )?;
        }
        step.complete()
    }
    /// Diagnostic address inspection outside timed execution.
    pub(super) fn addresses(&mut self) -> Result<Vec<u64>> {
        let names = self
            .workspace
            .layout()
            .regions()
            .iter()
            .map(|r| r.name.clone())
            .collect::<Vec<_>>();
        let step = self.workspace.begin_step()?;
        let addresses = names
            .iter()
            .map(|n| Ok(step.region(n)?.pointer()))
            .collect::<Result<Vec<_>>>()?;
        step.complete()?;
        Ok(addresses)
    }
    /// Qualification-only failure after queued work: dropping the lease must drain
    /// and poison it, so subsequent attempts cannot reuse partially written storage.
    pub(super) fn abort_after_gate(
        &mut self,
        module: &Module<'_>,
        input: &Buffer<'_>,
    ) -> Result<()> {
        ensure!(
            module.belongs_to(self.context)
                && input.belongs_to(self.context)
                && input.len() == self.rows * self.width * 2,
            "MLP abort probe input/context mismatch"
        );
        let step = self.workspace.begin_step()?;
        // SAFETY: Same validated input and planned disjoint regions as run; dropping
        // this incomplete lease drains queued work and poisons the owner.
        unsafe {
            self.projections[0].enqueue(
                self.context,
                module,
                &step,
                "gate",
                input.pointer(),
                self.rows,
            )?;
        }
        anyhow::bail!("injected MLP workspace failure after gate")
    }
    pub(super) fn snapshot(&self) -> Result<Vec<(String, Vec<u8>)>> {
        self.workspace
            .layout()
            .regions()
            .iter()
            .filter(|r| {
                r.name.ends_with("values")
                    || (r.name.ends_with("raw") && !r.name.starts_with("activation."))
            })
            .map(|r| Ok((r.name.clone(), self.workspace.read_region(&r.name)?)))
            .collect()
    }
    pub(super) fn bytes(&self) -> usize {
        self.workspace.layout().high_water_bytes()
    }
}
