//! Disjoint checked output views borrowed for one complete MLP workspace chain.
use super::{device_view::DeviceWrite, resident_workspace::WorkspaceStep};
use anyhow::Result;

pub(super) struct Views<'step, 'ctx> {
    pub gate: [DeviceWrite<'step, 'ctx>; 5],
    pub up: [DeviceWrite<'step, 'ctx>; 5],
    pub down: [DeviceWrite<'step, 'ctx>; 5],
    pub activation: [DeviceWrite<'step, 'ctx>; 4],
}
impl<'step, 'ctx> Views<'step, 'ctx> {
    pub(super) fn new(step: &'step mut WorkspaceStep<'_, 'ctx>) -> Result<Self> {
        let [
            g0,
            g1,
            g2,
            g3,
            g4,
            u0,
            u1,
            u2,
            u3,
            u4,
            d0,
            d1,
            d2,
            d3,
            d4,
            a0,
            a1,
            a2,
            a3,
        ] = step.write_regions([
            "gate.codes",
            "gate.scales",
            "gate.effective",
            "gate.values",
            "gate.raw",
            "up.codes",
            "up.scales",
            "up.effective",
            "up.values",
            "up.raw",
            "down.codes",
            "down.scales",
            "down.effective",
            "down.values",
            "down.raw",
            "activation.values",
            "activation.silu",
            "activation.activated",
            "activation.raw",
        ])?;
        Ok(Self {
            gate: [g0, g1, g2, g3, g4],
            up: [u0, u1, u2, u3, u4],
            down: [d0, d1, d2, d3, d4],
            activation: [a0, a1, a2, a3],
        })
    }
}
