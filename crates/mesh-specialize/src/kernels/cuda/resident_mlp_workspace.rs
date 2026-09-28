//! Experimental exact MLP chain with persistent scratch and explicit completion policy.
use super::{
    device_view::{ByteRange, DeviceRead, DeviceWrite, validate_launch_access},
    driver::{Buffer, Context, Module},
    mlp_workspace_projection::Binding,
    mlp_workspace_views::Views,
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
            crate::kernels::fp8_profile::current()?.exact_decoder()
                && super::resident_fp8_splitk::configured_splits()?.is_none(),
            "MLP workspace experiment requires exact decoder projections and split-K off"
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
        enqueue_chain(
            self.context,
            module,
            &mut self.workspace,
            &self.projections,
            input,
            [self.rows, self.width, self.channels],
            operator_waits,
        )
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
        let input = DeviceRead::from_buffer(input)?;
        let mut step = self.workspace.begin_step()?;
        let views = Views::new(&mut step)?;
        // SAFETY: Same validated input and planned disjoint regions as run; dropping
        // this incomplete lease drains queued work and poisons the owner.
        unsafe {
            self.projections[0].enqueue(self.context, module, &input, &views.gate, self.rows)?;
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

/// Run identical kernels using a caller-owned exclusive scratch arena.
pub(super) fn enqueue_chain(
    ctx: &Context,
    module: &Module<'_>,
    workspace: &mut ResidentWorkspace<'_>,
    projections: &[Binding<'_, '_>; 3],
    input: &Buffer<'_>,
    shape: [usize; 3],
    operator_waits: bool,
) -> Result<()> {
    let [rows, width, channels] = shape;
    ensure!(
        module.belongs_to(ctx)
            && workspace.belongs_to(ctx)
            && input.belongs_to(ctx)
            && input.len() == rows * width * 2,
        "MLP workspace input/context mismatch"
    );
    ensure!(
        (1..=512).contains(&rows)
            && (1..=32768).contains(&width)
            && (1..=32768).contains(&channels)
            && projections[0].width == width
            && projections[0].channels == channels
            && projections[1].width == width
            && projections[1].channels == channels
            && projections[2].width == channels
            && projections[2].channels == width,
        "MLP workspace projection geometry mismatch"
    );
    let input_view = DeviceRead::from_buffer(input)?;
    let input_view = input_view.subrange(ByteRange {
        offset: 0,
        bytes: input.len(),
        alignment: 2,
    })?;
    let mut step = workspace.begin_step()?;
    {
        let views = Views::new(&mut step)?;
        for (projection, outputs) in projections.iter().zip([&views.gate, &views.up]) {
            // SAFETY: Checked disjoint views and all owners remain live through lease completion/drain.
            unsafe {
                projection.enqueue(ctx, module, &input_view, outputs, rows)?;
            }
            if operator_waits {
                ctx.synchronize()?;
            }
        }
        activation(ctx, module, &views, rows * channels)?;
        if operator_waits {
            ctx.synchronize()?;
        }
        // SAFETY: Same ordered stream produces activation, all scratch is disjoint and leased.
        unsafe {
            projections[2].enqueue(
                ctx,
                module,
                &views.activation[0].as_read(),
                &views.down,
                rows,
            )?;
        }
    }
    step.complete()
}

/// Retain all intermediate BF16 boundaries and diagnostic writes in the existing ABI.
fn activation(
    ctx: &Context,
    module: &Module<'_>,
    views: &Views<'_, '_>,
    elements: usize,
) -> Result<()> {
    let gate = views.gate[3].as_read();
    let up = views.up[3].as_read();
    ensure!(
        gate.bytes() >= elements * 2 && up.bytes() >= elements * 2,
        "undersized MLP activation input"
    );
    for (output, bytes) in views.activation.iter().zip([2, 4, 2, 4]) {
        ensure!(
            output.bytes() >= elements * bytes,
            "undersized MLP activation output"
        );
    }
    validate_launch_access(ctx, &[&gate, &up], &views.activation.each_ref())?;
    let [values, silu, activated, raw] = views.activation.each_ref().map(DeviceWrite::pointer);
    let mut pointers = [gate.pointer(), up.pointer(), values, silu, activated, raw];
    let mut count = u32::try_from(elements)?;
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.push((&raw mut count).cast());
    // SAFETY: Checked disjoint views match six-pointer ABI; owners survive completion/drain.
    unsafe {
        module.function("mlp_silu_product")?.launch(
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )
    }
}
