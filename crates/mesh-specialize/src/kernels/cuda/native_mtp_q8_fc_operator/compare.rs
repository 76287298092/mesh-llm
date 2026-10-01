use super::fixture::{Fixture, ROWS};
use crate::native_mtp_q8_sliced_k_fc_reference::Q8SlicedKReference;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn run(
    fixture: &Fixture,
    expected: &Q8SlicedKReference,
    outputs: &[Vec<u16>; 2],
) -> Result<Value> {
    let count = fixture.candidate.tokens() * ROWS;
    ensure!(
        outputs.iter().all(|output| output.len() == count),
        "FC output extent mismatch"
    );
    let mut nonfinite = 0;
    let mut repeat_mismatches = 0;
    let mut simple_mismatches = 0;
    let mut selected_mismatches = 0;
    let mut failures = Vec::new();
    for (index, (&first, &second)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
        nonfinite += usize::from(first & 0x7f80 == 0x7f80) + usize::from(second & 0x7f80 == 0x7f80);
        repeat_mismatches += usize::from(first != second);
        let token = index / ROWS;
        let row = index % ROWS;
        if let Some(bits) = fixture.simple_expected(token, row)? {
            simple_mismatches += usize::from(first != bits) + usize::from(second != bits);
            if (first != bits || second != bits) && failures.len() < 16 {
                failures.push(json!({"token": token, "row": row, "expected_bf16": bits, "observed_bf16": [first, second]}));
            }
        }
    }
    for token in 0..fixture.candidate.tokens() {
        for (selected, &row) in fixture.view.source_rows.iter().enumerate() {
            let oracle_index = token * fixture.view.source_rows.len() + selected;
            let bits = expected.output_bf16[oracle_index];
            let actual = [
                outputs[0][token * ROWS + row],
                outputs[1][token * ROWS + row],
            ];
            selected_mismatches += actual.iter().filter(|&&word| word != bits).count();
            if actual.iter().any(|&word| word != bits) && failures.len() < 16 {
                failures.push(json!({"token": token, "row": row, "expected_bf16": bits, "observed_bf16": actual}));
            }
        }
    }
    Ok(json!({
        "all_passed": nonfinite == 0 && repeat_mismatches == 0 && simple_mismatches == 0 && selected_mismatches == 0,
        "output_count_per_repeat": count, "repeats": 2,
        "nonfinite_outputs": nonfinite, "repeat_mismatches": repeat_mismatches,
        "all_row_simple_mismatches": simple_mismatches, "selected_oracle_mismatches": selected_mismatches,
        "selected_rows": fixture.view.source_rows,
        "all_row_exact_checked": fixture.simple_expected(0, 0)?.is_some(),
        "scheduled_f32": expected.scheduled_f32,
        "mathematical_f64": expected.mathematical_f64,
        "mathematical_error_bound": expected.mathematical_error_bound,
        "failures_first_16": failures,
        "gate": "exact BF16 on exact dyadic fixtures; independent FP64 bound retained unchanged, not a GPU tolerance",
    }))
}
