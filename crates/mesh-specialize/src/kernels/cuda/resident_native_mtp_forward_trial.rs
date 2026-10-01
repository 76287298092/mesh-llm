//! Loaded-object repeat/state consistency only. This does not replace the target-batch gate.

mod capture;
mod execute;
mod report;
#[cfg(test)]
mod tests;

use super::resident_target_batch_trial::{compare, state};
use super::{
    driver::{Buffer, Context, Module},
    resident_native_mtp_forward::{Model, Session},
};
use crate::{kernels::DecoderConfig, packages::qwen3_8_27b::native_mtp_views::ProposalTokenMap};
use anyhow::{Result, ensure};
use report::{CaseReport, Report};

const HIDDEN: usize = 5_120;
const SHORTLIST: usize = 131_072;
const READBACK_BYTES: usize = 1024 * 1024;

pub(super) struct Request<'a, 'w, 'ctx> {
    pub model: &'a Model<'w, 'ctx>,
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub config: &'a DecoderConfig,
    pub base: &'a Session<'ctx>,
    pub tokens: &'a [u32; 5],
    pub normalized_target_hidden: &'a Buffer<'ctx>,
    pub continuation_token: u32,
    pub normalized_continuation_hidden: &'a Buffer<'ctx>,
    pub proposal_tokens: &'a ProposalTokenMap,
}

pub(super) fn run(request: Request<'_, '_, '_>) -> Result<serde_json::Value> {
    let mut cases = Vec::with_capacity(7);
    for rows in 1..=5 {
        cases.push(run_case(&request, rows, false));
    }
    for depth in [1, 4] {
        cases.push(run_case(&request, depth, true));
    }
    serde_json::to_value(Report::new(cases)).map_err(Into::into)
}

fn validate(request: &Request<'_, '_, '_>) -> Result<()> {
    ensure!(
        request.config.hidden == HIDDEN && request.config.vocabulary == 248_320,
        "trial requires target248320/hidden5120 geometry"
    );
    ensure!(
        request.module.belongs_to(request.context),
        "trial module context mismatch"
    );
    ensure!(
        request.base.state.belongs_to(request.context),
        "trial base context mismatch"
    );
    ensure!(
        request.base.state.layout() == &super::resident_mtp::mtp_state_layout(request.config)?,
        "trial base layout differs from config"
    );
    ensure!(
        request.base.cursor.capacity() == request.config.capacity
            && !request.base.cursor.is_poisoned(),
        "trial base cursor is invalid"
    );
    ensure!(
        request
            .base
            .cursor
            .past()
            .checked_add(6)
            .is_some_and(|end| end <= request.config.capacity),
        "trial needs room for five rows and continuation"
    );
    ensure!(
        request.proposal_tokens.len() == SHORTLIST,
        "trial signed proposal map extent mismatch"
    );
    ensure!(
        request
            .tokens
            .iter()
            .chain(std::iter::once(&request.continuation_token))
            .all(|token| *token < 248_320),
        "trial fixture token outside target vocabulary"
    );
    for (buffer, bytes) in [
        (request.normalized_target_hidden, 5 * HIDDEN * 2),
        (request.normalized_continuation_hidden, HIDDEN * 2),
    ] {
        ensure!(
            buffer.belongs_to(request.context) && buffer.len() == bytes,
            "trial normalized fixture context or extent mismatch"
        );
    }
    Ok(())
}

fn run_case(request: &Request<'_, '_, '_>, count: usize, recursive: bool) -> CaseReport {
    let mut case = CaseReport::new(count, recursive);
    match validate(request)
        .and_then(|()| execute::case(request, (count, recursive), &mut case.phases))
    {
        Ok(()) => {}
        Err(error) => case.errors.push(format!("{error:#}")),
    }
    case.passed = case.is_consistent();
    case
}
