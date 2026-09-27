//! Context-owned CUDA storage for a finalized reusable scratch layout.

use crate::engine::workspace::WorkspaceLayout;
use anyhow::{Context as _, Result, anyhow, ensure};
use std::marker::PhantomData;

use super::driver::{Buffer, Context};

/// One CUDA allocation for a fixed, checked scratch layout.
pub(super) struct ResidentWorkspace<'ctx> {
    context: &'ctx Context,
    allocation: Buffer<'ctx>,
    layout: WorkspaceLayout,
    poisoned: bool,
}

impl<'ctx> ResidentWorkspace<'ctx> {
    /// Allocate the exact high-water size of `layout` in `context`.
    pub(super) fn new(context: &'ctx Context, layout: WorkspaceLayout) -> Result<Self> {
        ensure!(
            layout.high_water_bytes() > 0 && layout.high_water_bytes() <= layout.capacity_bytes(),
            "workspace layout has invalid high-water or capacity accounting"
        );
        let allocation = Buffer::new(context, layout.high_water_bytes())?;
        ensure!(
            allocation.len() == layout.high_water_bytes(),
            "CUDA workspace allocation size differs from its finalized layout"
        );
        Ok(Self {
            context,
            allocation,
            layout,
            poisoned: false,
        })
    }

    /// Begin exclusive use of this arena for one sequence of device operations.
    pub(super) fn begin_step(&mut self) -> Result<WorkspaceStep<'_, 'ctx>> {
        ensure!(
            !self.poisoned,
            "CUDA workspace is poisoned and cannot be reused"
        );
        Ok(WorkspaceStep {
            owner: self,
            closed: false,
        })
    }

    pub(super) fn layout(&self) -> &WorkspaceLayout {
        &self.layout
    }

    /// Read completed scratch for diagnostics, never during an active mutable lease.
    pub(super) fn read_region(&self, name: &str) -> Result<Vec<u8>> {
        ensure!(!self.poisoned, "cannot read poisoned workspace");
        let region = self.layout.region(name)?;
        let mut bytes = vec![0; usize::try_from(region.length)?];
        self.allocation
            .download_at(usize::try_from(region.offset)?, &mut bytes)?;
        Ok(bytes)
    }

    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

/// Mutable-borrow lease held until all work using the workspace has completed.
pub(super) struct WorkspaceStep<'workspace, 'ctx> {
    owner: &'workspace mut ResidentWorkspace<'ctx>,
    closed: bool,
}

impl<'workspace, 'ctx> WorkspaceStep<'workspace, 'ctx> {
    /// Borrow a checked, non-owning device view for one planned region.
    pub(super) fn region(&self, name: &str) -> Result<DeviceRegion<'_, 'ctx>> {
        let region = self.owner.layout.region(name)?;
        let end = region
            .offset
            .checked_add(region.length)
            .ok_or_else(|| anyhow!("CUDA workspace region `{name}` range overflows u64"))?;
        let allocation_bytes = u64::try_from(self.owner.allocation.len())
            .context("CUDA workspace allocation size does not fit the device pointer range")?;
        ensure!(
            end <= allocation_bytes,
            "CUDA workspace region `{name}` exceeds its backing allocation"
        );
        let pointer = self
            .owner
            .allocation
            .pointer()
            .checked_add(region.offset)
            .ok_or_else(|| anyhow!("CUDA workspace device pointer plus region offset overflows"))?;
        Ok(DeviceRegion {
            pointer,
            bytes: usize::try_from(region.length)
                .context("CUDA workspace region length does not fit usize")?,
            _workspace: PhantomData,
        })
    }

    /// Synchronize the owning CUDA context before making the arena reusable.
    pub(super) fn complete(mut self) -> Result<()> {
        match self.owner.context.synchronize() {
            Ok(()) => {
                self.closed = true;
                Ok(())
            }
            Err(error) => {
                self.owner.poisoned = true;
                Err(error.context("completing CUDA workspace step"))
            }
        }
    }
}

impl Drop for WorkspaceStep<'_, '_> {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.owner.poisoned = true;
        if let Err(error) = self.owner.context.synchronize() {
            tracing::warn!(error = %error, "failed to drain incomplete CUDA workspace step");
        }
    }
}

/// Checked device address and byte length, valid only while its workspace step is borrowed.
pub(super) struct DeviceRegion<'step, 'ctx> {
    pointer: u64,
    bytes: usize,
    _workspace: PhantomData<&'step ResidentWorkspace<'ctx>>,
}

impl DeviceRegion<'_, '_> {
    /// Raw CUDA pointer for kernel argument construction; retain it only within the active lease.
    pub(super) fn pointer(&self) -> u64 {
        self.pointer
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}
