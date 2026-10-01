use super::oracle::finite;

#[derive(Debug)]
pub(super) struct Failure {
    pub index: usize,
    pub expected: Option<u16>,
    pub actual: Option<u16>,
}

#[derive(Debug)]
pub(super) struct Comparison {
    pub expected_elements: usize,
    pub actual_elements: usize,
    pub compared_elements: usize,
    pub finite: usize,
    pub nonfinite: usize,
    pub exact_mismatches: usize,
    pub failures: Vec<Failure>,
}

impl Comparison {
    pub(super) const fn passed(&self) -> bool {
        self.expected_elements == self.actual_elements
            && self.exact_mismatches == 0
            && self.nonfinite == 0
    }
}

pub(super) fn words(raw: &[u8]) -> Option<Vec<u16>> {
    if !raw.len().is_multiple_of(2) {
        return None;
    }
    Some(
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect(),
    )
}

pub(super) fn compare(expected: &[u16], actual: &[u16]) -> Comparison {
    let mut result = Comparison {
        expected_elements: expected.len(),
        actual_elements: actual.len(),
        compared_elements: expected.len().min(actual.len()),
        finite: actual.iter().filter(|&&word| finite(word)).count(),
        nonfinite: actual.iter().filter(|&&word| !finite(word)).count(),
        exact_mismatches: 0,
        failures: Vec::with_capacity(16),
    };
    for index in 0..expected.len().max(actual.len()) {
        let expected = expected.get(index).copied();
        let actual = actual.get(index).copied();
        let mismatch = expected != actual;
        result.exact_mismatches += usize::from(mismatch);
        if (mismatch || actual.is_some_and(|word| !finite(word))) && result.failures.len() < 16 {
            result.failures.push(Failure {
                index,
                expected,
                actual,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests;
