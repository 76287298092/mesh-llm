use super::reference::decode;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Difference {
    index: usize,
    expected_word: Option<u16>,
    actual_word: Option<u16>,
}

#[derive(Serialize)]
pub(super) struct Comparison {
    pub mismatches: usize,
    expected_nonfinite: usize,
    actual_nonfinite: usize,
    extent_matches: bool,
    first_16: Vec<Difference>,
}

impl Comparison {
    pub(super) const fn passed(&self) -> bool {
        self.extent_matches && self.mismatches == 0
            && self.expected_nonfinite == 0 && self.actual_nonfinite == 0
    }
}

pub(super) fn compare(expected: &[u16], actual: &[u16]) -> Comparison {
    let mut result = Comparison {
        mismatches: 0,
        expected_nonfinite: expected.iter().filter(|&&word| !decode(word).is_finite()).count(),
        actual_nonfinite: actual.iter().filter(|&&word| !decode(word).is_finite()).count(),
        extent_matches: expected.len() == actual.len(),
        first_16: Vec::with_capacity(16),
    };
    for index in 0..expected.len().max(actual.len()) {
        let expected_word = expected.get(index).copied();
        let actual_word = actual.get(index).copied();
        if expected_word != actual_word {
            result.mismatches += 1;
            if result.first_16.len() < 16 {
                result.first_16.push(Difference { index, expected_word, actual_word });
            }
        }
    }
    result
}

#[derive(Serialize)]
pub(super) struct Diagnostic {
    max_absolute_error: f64,
    squared_error_sum: f64,
    ideal_squared_sum: f64,
    nonfinite_pairs: usize,
}

pub(super) fn diagnostic(ideal: &[f64], actual: &[u16]) -> Diagnostic {
    let mut result = Diagnostic { max_absolute_error: 0.0, squared_error_sum: 0.0,
        ideal_squared_sum: 0.0, nonfinite_pairs: 0 };
    for (&expected, &word) in ideal.iter().zip(actual) {
        let observed = f64::from(decode(word));
        if expected.is_finite() && observed.is_finite() {
            let error = observed - expected;
            result.max_absolute_error = result.max_absolute_error.max(error.abs());
            result.squared_error_sum += error * error;
            result.ideal_squared_sum += expected * expected;
        } else {
            result.nonfinite_pairs += 1;
        }
    }
    result
}
