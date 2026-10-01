mod attention_block;
mod feed_forward;
mod proposal;
mod rows;

use super::{Model, Output, Session};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    resident_state::ResidentState,
};
use anyhow::Result;

pub(super) fn forward_rows<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    session: &mut Session<'_>,
) -> Result<Output<'a>> {
    rows::forward_rows(model, context, module, tokens, target_hidden, session)
}

pub(super) fn forward_serial<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    session: &mut Session<'_>,
) -> Result<Output<'a>> {
    rows::forward_serial(model, context, module, tokens, target_hidden, session)
}

#[cfg(test)]
pub(super) use proposal::first_argmax;

fn run_rows<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    state: &mut ResidentState<'_>,
    past: usize,
) -> Result<(Buffer<'a>, Vec<u16>, usize, u32)> {
    let attention =
        attention_block::run(model, context, module, tokens, target_hidden, state, past)?;
    let hidden = feed_forward::run(model, context, module, &attention, tokens.len())?;
    proposal::run(model, context, module, hidden)
}
