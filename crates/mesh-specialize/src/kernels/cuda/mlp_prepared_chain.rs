//! Prepared MLP functions for explicit-stream qualification, without graph capture.
use super::{
    device_view::{DeviceRead, DeviceWrite, validate_launch_access},
    driver::{Context, Function, Module, graph::Stream},
    mlp_prepared_projection::Prepared,
    mlp_workspace_projection::Binding,
    mlp_workspace_views::Views,
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

pub(super) struct Plan<'module, 'w, 'ctx> {
    context: &'ctx Context,
    projections: [Prepared<'module, 'w, 'ctx>; 3],
    activation: Function<'module, 'ctx>,
    elements: usize,
}
impl<'module, 'w, 'ctx> Plan<'module, 'w, 'ctx> {
    pub(super) fn new(
        ctx: &'ctx Context,
        module: &'module Module<'ctx>,
        bindings: &[Binding<'w, 'ctx>; 3],
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            bindings[0].width == bindings[1].width
                && bindings[0].channels == bindings[1].channels
                && bindings[2].width == bindings[0].channels
                && bindings[2].channels == bindings[0].width,
            "prepared MLP geometry mismatch"
        );
        let elements = rows
            .checked_mul(bindings[0].channels)
            .ok_or_else(|| anyhow::anyhow!("prepared activation size overflows"))?;
        Ok(Self {
            context: ctx,
            projections: [
                Prepared::new(ctx, module, &bindings[0], rows)?,
                Prepared::new(ctx, module, &bindings[1], rows)?,
                Prepared::new(ctx, module, &bindings[2], rows)?,
            ],
            activation: module.function("mlp_silu_product")?,
            elements,
        })
    }

    /// # Safety
    /// All owners and functions survive successful stream completion or error drain.
    /// External input producers are complete. No concurrent access or capture is permitted.
    pub(super) unsafe fn enqueue(
        &self,
        stream: &Stream<'ctx>,
        input: &DeviceRead<'_, '_>,
        views: &Views<'_, '_>,
        abort_after_gate: bool,
    ) -> Result<()> {
        ensure!(
            !stream.is_capturing(),
            "prepared MLP qualification does not capture"
        );
        // SAFETY: Caller retains owners; views are a disjoint whole-chain partition.
        unsafe {
            self.projections[0].enqueue(stream, input, &views.gate)?;
        }
        ensure!(
            !abort_after_gate,
            "injected prepared MLP failure after gate"
        );
        // SAFETY: Same owner and stream completion obligations as gate.
        unsafe {
            self.projections[1].enqueue(stream, input, &views.up)?;
            self.activation(stream, views)?;
            self.projections[2].enqueue(stream, &views.activation[0].as_read(), &views.down)
        }
    }

    unsafe fn activation(&self, stream: &Stream<'ctx>, views: &Views<'_, '_>) -> Result<()> {
        let gate = views.gate[3].as_read();
        let up = views.up[3].as_read();
        ensure!(
            gate.bytes() >= self.elements * 2 && up.bytes() >= self.elements * 2,
            "undersized prepared activation input"
        );
        for (output, bytes) in views.activation.iter().zip([2, 4, 2, 4]) {
            ensure!(
                output.bytes() >= self.elements * bytes,
                "undersized prepared activation output"
            );
        }
        validate_launch_access(self.context, &[&gate, &up], &views.activation.each_ref())?;
        let [v, s, a, r] = views.activation.each_ref().map(DeviceWrite::pointer);
        let mut pointers = [gate.pointer(), up.pointer(), v, s, a, r];
        let mut count = u32::try_from(self.elements)?;
        let [p0, p1, p2, p3, p4, p5] = pointers
            .each_mut()
            .map(|p| ptr::from_mut(p).cast::<c_void>());
        let mut args = [p0, p1, p2, p3, p4, p5, ptr::from_mut(&mut count).cast()];
        // SAFETY: Validated six-pointer/count ABI; all owners survive stream completion.
        unsafe {
            self.activation.launch_on_stream(
                stream,
                [count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )
        }
    }
}
