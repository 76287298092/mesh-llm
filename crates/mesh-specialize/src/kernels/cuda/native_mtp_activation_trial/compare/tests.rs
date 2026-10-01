use super::super::oracle::{self, Input, Operation};
use super::{compare, words};

#[test]
fn rejects_interior_corruption_when_boundary_words_match() {
    let expected = vec![0x3f80; 513];
    let mut actual = expected.clone();
    actual[257] ^= 1;
    let check = compare(&expected, &actual);
    assert!(!check.passed());
    assert_eq!(check.exact_mismatches, 1);
    assert_eq!(check.failures[0].index, 257);
}

#[test]
fn rejects_both_repeats_when_identically_wrong() {
    let expected = [0x3f80, 0x3f81, 0x3f82];
    let first = [0x3f80, 0x3f80, 0x3f82];
    let second = first;
    let passed = compare(&expected, &first).passed()
        && compare(&expected, &second).passed()
        && compare(&first, &second).passed();
    assert!(!passed);
}

#[test]
fn rejects_signed_zero_loss_even_when_numerically_equal() {
    let expected = [Operation::SiluMul.expected(Input {
        gate: 0x8000,
        factor: 0x3f80,
    })];
    let check = compare(&expected, &[0]);
    assert!(!check.passed());
    assert_eq!(check.exact_mismatches, 1);
}

#[test]
fn rejects_nonfinite_corruption_and_unwritten_poisons() {
    for word in [0x7f80, 0xff80, 0x7fc1, 0xffc2] {
        let check = compare(&[0x3f80; 3], &[0x3f80, word, 0x3f80]);
        assert!(!check.passed());
        assert_eq!(check.finite, 2);
        assert_eq!(check.nonfinite, 1);
    }
}

#[test]
fn rejects_premature_bf16_activation_rounding_for_each_kernel() {
    let inputs = oracle::inputs();
    for operation in [Operation::AttentionGate, Operation::SiluMul] {
        let expected: Vec<_> = inputs
            .iter()
            .copied()
            .map(|input| operation.expected(input))
            .collect();
        let premature: Vec<_> = inputs
            .iter()
            .map(|input| {
                oracle::round(
                    oracle::decode(oracle::round(operation.activation(input.gate)))
                        * oracle::decode(input.factor),
                )
            })
            .collect();
        let check = compare(&expected, &premature);
        assert!(!check.passed());
        assert!(check.exact_mismatches > 0);
    }
}

#[test]
fn parses_every_output_word_and_rejects_truncation() {
    let raw: Vec<_> = [0x8000_u16, 0x3f81, 0xc123]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let actual = words(&raw).unwrap();
    assert!(compare(&[0x8000, 0x3f81, 0xc123], &actual).passed());
    assert!(words(&raw[..5]).is_none());
    assert!(!compare(&actual, &actual[..2]).passed());
    assert!(!compare(&actual[..2], &actual).passed());
}

#[test]
fn counts_all_failures_while_retaining_only_first_sixteen() {
    let check = compare(&[0x3f80; 33], &[0x3f81; 33]);
    assert_eq!(check.exact_mismatches, 33);
    assert_eq!(check.failures.len(), 16);
    assert_eq!(check.failures[15].index, 15);
}

#[test]
fn repeat_comparison_rejects_only_second_repeat_corruption() {
    let expected = [0x3f80; 5];
    let mut second = expected;
    second[2] = 0xbf80;
    assert!(!compare(&expected, &second).passed());
    assert_eq!(compare(&expected, &second).exact_mismatches, 1);
}
