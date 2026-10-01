use super::super::{schedule_reference::ScheduleProjection, validate::ValidatedView};
use super::launch::BoundOutput;
use crate::native_mtp_q4_gemv_reference::Q4ProjectionReference;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) const MAX_SCALED_ERROR: f64 = 2.0e-4;
const MAX_RECORDED_FAILURES: usize = 16;
const TARGET_VOCABULARY: u32 = 248_320;

pub(super) struct References<'a> {
    pub(super) fp64: &'a Q4ProjectionReference,
    pub(super) scheduled: &'a ScheduleProjection,
    pub(super) validated: &'a ValidatedView,
}

pub(super) struct Case<'a> {
    pub(super) name: &'a str,
    pub(super) outputs: [BoundOutput; 2],
    pub(super) references: References<'a>,
    pub(super) target_ids: &'a [u32],
}

pub(super) fn evaluate(case: Case<'_>) -> Result<Value> {
    let References {
        fp64,
        scheduled,
        validated,
    } = case.references;
    let count = usize::try_from(validated.selected_rows)?;
    ensure!(
        fp64.raw_f64.len() == count
            && fp64.logits_bf16.len() == count
            && scheduled.raw_f32.len() == count
            && scheduled.logits_bf16.len() == count
            && validated.source_rows.len() == count
            && case.target_ids.len() == count,
        "native Q4 resident reference extents differ"
    );
    ensure!(
        case.outputs.iter().all(|output| {
            output.raw_f32_bits.len() == count && output.logits_bf16.len() == count
        }),
        "native Q4 resident output extent mismatch"
    );

    let mut nonfinite_outputs = 0_usize;
    let mut schedule_raw_mismatches = 0_usize;
    let mut schedule_bf16_mismatches = 0_usize;
    let mut repeat_raw_mismatches = 0_usize;
    let mut repeat_bf16_mismatches = 0_usize;
    let mut fp64_bf16_diagnostic_mismatches = 0_usize;
    let mut max_absolute_fp64_error = 0.0_f64;
    let mut max_scaled_fp64_error = 0.0_f64;
    let mut failures = Vec::new();

    for row in 0..count {
        let expected_raw_bits = scheduled.raw_f32[row].to_bits();
        let first_raw_bits = case.outputs[0].raw_f32_bits[row];
        let second_raw_bits = case.outputs[1].raw_f32_bits[row];
        let first_raw = f32::from_bits(first_raw_bits);
        let second_raw = f32::from_bits(second_raw_bits);
        let first_bf16 = case.outputs[0].logits_bf16[row];
        let second_bf16 = case.outputs[1].logits_bf16[row];
        let finite = first_raw.is_finite() && second_raw.is_finite();
        let finite_bf16 = !is_nonfinite_bf16(first_bf16) && !is_nonfinite_bf16(second_bf16);
        nonfinite_outputs += usize::from(!first_raw.is_finite())
            + usize::from(!second_raw.is_finite())
            + usize::from(is_nonfinite_bf16(first_bf16))
            + usize::from(is_nonfinite_bf16(second_bf16));
        let raw_matches =
            first_raw_bits == expected_raw_bits && second_raw_bits == expected_raw_bits;
        let bf16_matches =
            first_bf16 == scheduled.logits_bf16[row] && second_bf16 == scheduled.logits_bf16[row];
        let repeat_raw_matches = first_raw_bits == second_raw_bits;
        let repeat_bf16_matches = first_bf16 == second_bf16;
        schedule_raw_mismatches += usize::from(first_raw_bits != expected_raw_bits)
            + usize::from(second_raw_bits != expected_raw_bits);
        schedule_bf16_mismatches += usize::from(first_bf16 != scheduled.logits_bf16[row])
            + usize::from(second_bf16 != scheduled.logits_bf16[row]);
        repeat_raw_mismatches += usize::from(!repeat_raw_matches);
        repeat_bf16_mismatches += usize::from(!repeat_bf16_matches);
        fp64_bf16_diagnostic_mismatches +=
            usize::from(scheduled.logits_bf16[row] != fp64.logits_bf16[row]);

        let (absolute_error, scaled_error) = if finite {
            let error = (f64::from(first_raw) - fp64.raw_f64[row]).abs();
            let repeated_error = (f64::from(second_raw) - fp64.raw_f64[row]).abs();
            let absolute = error.max(repeated_error);
            (absolute, absolute / fp64.raw_f64[row].abs().max(1.0))
        } else {
            (f64::MAX, f64::MAX)
        };
        max_absolute_fp64_error = max_absolute_fp64_error.max(absolute_error);
        max_scaled_fp64_error = max_scaled_fp64_error.max(scaled_error);

        if (!finite
            || !finite_bf16
            || !raw_matches
            || !bf16_matches
            || !repeat_raw_matches
            || !repeat_bf16_matches
            || scaled_error > MAX_SCALED_ERROR)
            && failures.len() < MAX_RECORDED_FAILURES
        {
            failures.push(json!({
                "proposal_row": row,
                "source_parent_row": validated.source_rows[row],
                "scheduled_raw_f32_bits": expected_raw_bits,
                "first_raw_f32_bits": first_raw_bits,
                "second_raw_f32_bits": second_raw_bits,
                "scheduled_bf16": scheduled.logits_bf16[row],
                "first_bf16": first_bf16,
                "second_bf16": second_bf16,
                "fp64_raw": fp64.raw_f64[row],
                "max_scaled_fp64_error": scaled_error,
                "finite": finite,
                "finite_bf16": finite_bf16,
                "schedule_raw_matches": raw_matches,
                "schedule_bf16_matches": bf16_matches,
                "repeat_raw_matches": repeat_raw_matches,
                "repeat_bf16_matches": repeat_bf16_matches,
            }));
        }
    }

    let scheduled_winner = first_argmax(&scheduled.logits_bf16);
    let first_winner = first_argmax(&case.outputs[0].logits_bf16);
    let second_winner = first_argmax(&case.outputs[1].logits_bf16);
    let (scheduled_target, first_target, second_target) =
        match (scheduled_winner, first_winner, second_winner) {
            (Some(expected), Some(first), Some(second)) => (
                case.target_ids.get(expected).copied(),
                case.target_ids.get(first).copied(),
                case.target_ids.get(second).copied(),
            ),
            _ => (None, None, None),
        };
    let selection_matches_schedule = scheduled_winner.is_some()
        && first_winner == scheduled_winner
        && second_winner == scheduled_winner;
    let remap_matches_schedule = scheduled_target.is_some()
        && first_target == scheduled_target
        && second_target == scheduled_target
        && scheduled_target.is_some_and(|target| target < TARGET_VOCABULARY);
    let scaled_error_passed = max_scaled_fp64_error <= MAX_SCALED_ERROR;
    let all_passed = nonfinite_outputs == 0
        && schedule_raw_mismatches == 0
        && schedule_bf16_mismatches == 0
        && repeat_raw_mismatches == 0
        && repeat_bf16_mismatches == 0
        && scaled_error_passed
        && selection_matches_schedule
        && remap_matches_schedule;
    if (!selection_matches_schedule || !remap_matches_schedule)
        && failures.len() < MAX_RECORDED_FAILURES
    {
        let expected_row = scheduled_winner.unwrap_or_default();
        let actual_row = first_winner
            .or(second_winner)
            .or(scheduled_winner)
            .unwrap_or_default();
        failures.push(json!({
            "proposal_row": actual_row,
            "expected_proposal_row": scheduled_winner,
            "actual_proposal_rows": [first_winner, second_winner],
            "expected_raw_f32_bits": scheduled.raw_f32.get(expected_row).map(|value| value.to_bits()),
            "actual_raw_f32_bits": [
                case.outputs[0].raw_f32_bits.get(actual_row),
                case.outputs[1].raw_f32_bits.get(actual_row),
            ],
            "expected_bf16": scheduled.logits_bf16.get(expected_row),
            "actual_bf16": [
                case.outputs[0].logits_bf16.get(actual_row),
                case.outputs[1].logits_bf16.get(actual_row),
            ],
            "expected_target_token": scheduled_target,
            "actual_target_tokens": [first_target, second_target],
        }));
    }

    Ok(json!({
        "case": case.name,
        "all_passed": all_passed,
        "outputs": count,
        "repeats": 2,
        "nonfinite_outputs": nonfinite_outputs,
        "schedule_raw_f32_bit_mismatches": schedule_raw_mismatches,
        "schedule_bf16_bit_mismatches": schedule_bf16_mismatches,
        "repeat_raw_bit_mismatches": repeat_raw_mismatches,
        "repeat_bf16_bit_mismatches": repeat_bf16_mismatches,
        "max_absolute_fp64_error": max_absolute_fp64_error,
        "max_scaled_fp64_error": max_scaled_fp64_error,
        "fp64_scaled_error_limit": MAX_SCALED_ERROR,
        "fp64_scaled_error_passed": scaled_error_passed,
        "serial_fp32_bf16_diagnostic_mismatches": fp64_bf16_diagnostic_mismatches,
        "serial_fp32_bf16_is_schedule_gate": false,
        "selection": {
            "expected_schedule_proposal_row": scheduled_winner,
            "first_proposal_row": first_winner,
            "second_proposal_row": second_winner,
            "expected_target_token": scheduled_target,
            "first_target_token": first_target,
            "second_target_token": second_target,
            "schedule_row_matches": selection_matches_schedule,
            "signed_map_remap_matches": remap_matches_schedule,
        },
        "failures_first_16": failures,
    }))
}

pub(super) fn first_argmax(logits: &[u16]) -> Option<usize> {
    let mut winner = None;
    let mut best = f32::NEG_INFINITY;
    for (row, &bits) in logits.iter().enumerate() {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return None;
        }
        if value > best {
            winner = Some(row);
            best = value;
        }
    }
    winner
}

fn is_nonfinite_bf16(bits: u16) -> bool {
    bits & 0x7f80 == 0x7f80
}

#[cfg(test)]
#[path = "comparison/tests.rs"]
mod tests;

pub(super) fn failed_case(name: &str, error: &anyhow::Error) -> Value {
    json!({
        "case": name,
        "all_passed": false,
        "error": format!("{error:#}"),
    })
}
