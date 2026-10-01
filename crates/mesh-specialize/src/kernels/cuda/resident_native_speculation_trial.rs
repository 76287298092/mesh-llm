//! Correctness-only native recovery trial. Serial reconstruction is not acceleration.
mod execute;
mod fixture;
mod oracle;
mod report;
#[path = "resident_target_batch_trial/state.rs"]
mod state;
#[cfg(test)]
mod tests;

use super::{
    driver::{Context, Module},
    resident_model::Model,
    resident_native_mtp_forward::Model as NativeModel,
};
use crate::kernels::DecoderConfig;
use anyhow::{Context as _, Result, ensure};
use report::{Case, CaseReport, Report};

pub(super) struct Request<'a, 'tw, 'dw, 'ctx> {
    pub target: &'a Model<'tw, 'ctx>,
    pub draft: &'a NativeModel<'dw, 'ctx>,
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub config: &'a DecoderConfig,
    pub prompt: &'a [u32],
}

pub(super) fn run(request: Request<'_, '_, '_, '_>) -> Result<serde_json::Value> {
    validate(&request)?;
    let cases = fixture::cases().into_iter().map(|case| {
        let mut report = CaseReport::new(case);
        if let Err(error) = execute::case(&request, &mut report) {
            report.errors.push(format!("{}: {error:#}", report.stage));
        }
        report.passed = report.is_correct();
        report
    }).collect::<Vec<_>>();
    let capacity = fixture::capacity_check(request.config.capacity)?;
    serde_json::to_value(Report {
        kind: "resident-native-speculation-recovery-correctness",
        all_passed: cases.iter().all(|case| case.passed) && capacity.passed,
        native_mtp_admitted: false,
        source_arithmetic_parity: false,
        performance_claim: false,
        target_batch_gate_replaced: false,
        eos_coverage: "blocked: run_native Request has no authoritative EOS contract",
        oracle: "ordinary forward_detailed_decode plus complete serial native shifted-token teacher forcing; correctness only",
        capacity,
        cases,
    }).context("serialize native speculation trial")
}

fn validate(request: &Request<'_, '_, '_, '_>) -> Result<()> {
    ensure!((1..=512).contains(&request.prompt.len()), "prompt must contain 1..=512 rows");
    ensure!(request.config.vocabulary >= 2, "forced rejection needs two valid target tokens");
    ensure!(request.prompt.iter().all(|token| usize::try_from(*token)
        .is_ok_and(|id| id < request.config.vocabulary)), "prompt token outside target vocabulary");
    ensure!(request.target.belongs_to(request.context) && request.module.belongs_to(request.context),
        "target or module context mismatch");
    let extent = request.prompt.len().checked_add(9).context("trial cursor extent overflow")?;
    ensure!(extent <= request.config.capacity, "trial needs eight outputs and two continuation inputs");
    Ok(())
}
