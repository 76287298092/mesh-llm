//! Resolved position-only kernels, used only during M=1 graph capture.

use super::ops::Args;
use crate::kernels::cuda::driver::{Function, Module};
use anyhow::Result;

pub(super) struct Position<'m, 'ctx> {
    pub(super) prepare: Function<'m, 'ctx>,
    pub(super) append: Function<'m, 'ctx>,
    pub(super) attention: Function<'m, 'ctx>,
    pub(super) past: u64,
}

impl<'m, 'ctx> Position<'m, 'ctx> {
    pub(super) fn new(module: &'m Module<'ctx>, past: u64) -> Result<Self> {
        Ok(Self {
            prepare: module.function("attention_qk_prepare_position")?,
            append: module.function("attention_kv_append_position")?,
            attention: module.function("causal_attention_bf16_position")?,
            past,
        })
    }
}

/// The exact append/attention ABI replaces the by-value u32 at its original
/// position with a pointer. Preparation instead appends a fifteenth parameter.
pub(super) fn past_argument(args: Args, position: Option<u64>, past: u32) -> Args {
    match position {
        Some(address) => args.ptr(address),
        None => args.u32(past),
    }
}
