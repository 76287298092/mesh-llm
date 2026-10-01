use super::{comparison::compare, fixtures::{self, Fixture}};

#[test]
fn gate_rejects_corruption_when_interior_row_changes() {
    let expected = fixtures::input(Fixture::DenseSigned, 5, 5120).unwrap();
    let mut actual = expected.clone();
    actual[2 * 5120 + 123] ^= 1;

    let report = compare(&expected, &actual);

    assert!(!report.passed());
    assert_eq!(report.mismatches, 1);
    let serialized = serde_json::to_value(report).unwrap();
    assert_eq!(serialized["first_16"][0]["index"], 10363);
}

#[test]
fn identical_wrong_repeats_do_not_qualify_against_oracle() {
    let expected = vec![0x3f00; 256];
    let first = vec![0; 256];
    let second = first.clone();

    let oracle_first = compare(&expected, &first);
    let oracle_second = compare(&expected, &second);
    let repeat = compare(&first, &second);

    assert!(repeat.passed());
    assert!(!oracle_first.passed() && !oracle_second.passed());
    assert_eq!(oracle_first.mismatches, 256);
}

#[test]
fn gate_retains_counts_when_many_words_or_nonfinite_words_fail() {
    let expected = vec![0x3f80; 32];
    let actual = vec![0x7f80; 32];

    let report = compare(&expected, &actual);

    assert!(!report.passed());
    let serialized = serde_json::to_value(report).unwrap();
    assert_eq!(serialized["mismatches"], 32);
    assert_eq!(serialized["actual_nonfinite"], 32);
    assert_eq!(serialized["first_16"].as_array().unwrap().len(), 16);
}
