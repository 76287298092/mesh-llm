use super::{
    compare,
    fixture::{Candidate, Fixture, Kind, ROWS},
};

#[test]
fn cancellation_matches_simple_oracle_when_five_columns_are_distinct() {
    let fixture = Fixture::new(Candidate::C8, Kind::Cancellation).expect("fixture");
    let expected = crate::native_mtp_q8_sliced_k_fc_reference::run(
        &fixture.object,
        &fixture.view,
        &fixture.input,
    )
    .expect("oracle");
    let actual: Vec<u16> = (0..5 * ROWS)
        .map(|index| {
            fixture
                .simple_expected(index / ROWS, index % ROWS)
                .expect("simple oracle")
                .expect("simple case")
        })
        .collect();

    let report = compare::run(&fixture, &expected, &[actual.clone(), actual]).expect("comparison");

    assert_eq!(report["all_passed"], true);
    let first_row: Vec<u16> = expected
        .output_bf16
        .chunks(fixture.view.source_rows.len())
        .map(|column| column[0])
        .collect();
    assert_eq!(
        first_row
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5
    );
}

#[test]
fn comparison_fails_when_last_row_is_poisoned_or_repeat_changes() {
    let fixture = Fixture::new(Candidate::C4, Kind::LastK).expect("fixture");
    let expected = crate::native_mtp_q8_sliced_k_fc_reference::run(
        &fixture.object,
        &fixture.view,
        &fixture.input,
    )
    .expect("oracle");
    let actual: Vec<u16> = (0..ROWS)
        .map(|row| {
            fixture
                .simple_expected(0, row)
                .expect("simple oracle")
                .expect("simple case")
        })
        .collect();
    let mut changed = actual.clone();
    changed[ROWS - 1] = 0x7fc1;

    let report = compare::run(&fixture, &expected, &[actual, changed]).expect("comparison");

    assert_eq!(report["all_passed"], false);
    assert_eq!(report["nonfinite_outputs"], 1);
    assert_eq!(report["repeat_mismatches"], 1);
    assert_eq!(report["selected_oracle_mismatches"], 1);
}
