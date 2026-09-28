//! Model-owned, single-arena MLP scratch cache. Never shared across CUDA contexts.
use super::{driver::Context, resident_workspace::ResidentWorkspace};
use crate::engine::workspace::WorkspaceLayout;
use anyhow::{Result, ensure};
use std::{cell::RefCell, rc::Rc, sync::OnceLock};

pub(super) type Shared<'ctx> = Rc<RefCell<Cache<'ctx>>>;
pub(super) fn enabled() -> Result<bool> {
    static VALUE: OnceLock<Result<bool, String>> = OnceLock::new();
    match VALUE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_MLP_WORKSPACE") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "off" => Ok(false),
        Ok(v) if v == "on" => Ok(true),
        _ => Err("MESH_SPECIALIZE_MLP_WORKSPACE must be off or on".into()),
    }) {
        Ok(v) => Ok(*v),
        Err(e) => Err(anyhow::anyhow!(e.clone())),
    }
}
pub(super) fn shared(ctx: &Context) -> Result<Option<Shared<'_>>> {
    if !enabled()? {
        return Ok(None);
    }
    ensure!(
        crate::kernels::fp8_profile::current()?.exact_decoder()
            && super::resident_fp8_splitk::configured_splits()?.is_none(),
        "model MLP workspace requires exact decoder projections and split-K off"
    );
    Ok(Some(Rc::new(RefCell::new(Cache {
        ctx,
        shape: [0; 3],
        arena: None,
    }))))
}
pub(super) struct Cache<'ctx> {
    ctx: &'ctx Context,
    shape: [usize; 3],
    arena: Option<ResidentWorkspace<'ctx>>,
}
impl<'ctx> Cache<'ctx> {
    pub(super) fn prepare(&mut self, shape: [usize; 3]) -> Result<&mut ResidentWorkspace<'ctx>> {
        ensure!(
            self.arena.as_ref().is_none_or(|a| !a.is_poisoned()),
            "model MLP workspace is poisoned"
        );
        if self.shape != shape || self.arena.is_none() {
            let layout = WorkspaceLayout::mlp_chain(shape)?;
            self.arena.take();
            self.arena = Some(ResidentWorkspace::new(self.ctx, layout)?);
            self.shape = shape;
        }
        self.arena
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("missing model workspace"))
    }
}
