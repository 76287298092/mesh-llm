//! Exact greedy selection with persistent reduction scratch and a 16-byte readback.
use super::{
    driver::{Buffer, Context, Module},
    resident_workspace::ResidentWorkspace,
};
use crate::engine::workspace::WorkspaceLayout;
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct Selector<'ctx> {
    context: &'ctx Context,
    vocabulary: usize,
    workspace: ResidentWorkspace<'ctx>,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Selection {
    pub token: u32,
    pub bits: u16,
}
impl<'ctx> Selector<'ctx> {
    pub(super) fn new(context: &'ctx Context, vocabulary: usize) -> Result<Self> {
        ensure!(
            (1..=262144).contains(&vocabulary),
            "invalid greedy vocabulary"
        );
        let layout = WorkspaceLayout::new(
            8192,
            [
                ("partials".into(), vocabulary.div_ceil(1024) * 16),
                ("result".into(), 16),
            ],
        )?;
        Ok(Self {
            context,
            vocabulary,
            workspace: ResidentWorkspace::new(context, layout)?,
        })
    }
    pub(super) fn select(&mut self, module: &Module<'_>, logits: &Buffer<'_>) -> Result<Selection> {
        let result = self.inspect(module, logits)?;
        ensure!(result[1] == 0, "nonfinite logit at {}", result[2]);
        Ok(Selection {
            token: result[0],
            bits: result[3] as u16,
        })
    }
    /// Diagnostic status readback; invalid values never become selected tokens.
    pub(super) fn inspect(&mut self, module: &Module<'_>, logits: &Buffer<'_>) -> Result<[u32; 4]> {
        ensure!(
            module.belongs_to(self.context) && logits.belongs_to(self.context),
            "greedy context mismatch"
        );
        ensure!(
            logits.len() == self.vocabulary * 2,
            "greedy logit extent mismatch"
        );
        let step = self.workspace.begin_step()?;
        let mut source = logits.pointer();
        let mut partials = step.region("partials")?.pointer();
        let mut result = step.region("result")?.pointer();
        let mut vocabulary = self.vocabulary as u32;
        let mut tiles = vocabulary.div_ceil(1024);
        let mut tile_args = [
            (&mut source as *mut u64).cast::<c_void>(),
            (&mut partials as *mut u64).cast(),
            (&mut vocabulary as *mut u32).cast(),
        ];
        // SAFETY: Validated contiguous BF16 input and planned tile records remain
        // live until lease completion. Both kernels use the same ordered stream.
        unsafe {
            module.function("greedy_bf16_tiles")?.launch(
                [tiles, 1, 1],
                [128, 1, 1],
                0,
                &mut tile_args,
            )?;
        }
        let mut finish_args = [
            (&mut partials as *mut u64).cast::<c_void>(),
            (&mut result as *mut u64).cast(),
            (&mut tiles as *mut u32).cast(),
        ];
        // SAFETY: Every tile record is initialized by the preceding launch, and
        // the result region has four writable u32 words. Error drop drains work.
        unsafe {
            module.function("greedy_bf16_finish")?.launch(
                [1, 1, 1],
                [128, 1, 1],
                0,
                &mut finish_args,
            )?;
        }
        step.complete()?;
        let bytes = self.workspace.read_region("result")?;
        let words = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("four-byte word"))
        });
        ensure!(words[1] <= 1, "invalid GPU greedy status");
        if words[1] == 0 {
            ensure!(
                words[0] < vocabulary
                    && words[2] == u32::MAX
                    && words[3] <= u16::MAX as u32
                    && words[3] & 0x7f80 != 0x7f80,
                "invalid finite GPU selection"
            );
        } else {
            ensure!(words[2] < vocabulary, "invalid nonfinite GPU position");
        }
        Ok(words)
    }
}
