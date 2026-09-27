//! Logical row-wise RMSNorm fixtures with an independent host oracle.

const EPSILON: f32 = 1e-5;
const SHAPES: [(usize, usize); 8] = [
    (1, 1),
    (3, 7),
    (3, 255),
    (3, 256),
    (3, 257),
    (1, 5120),
    (17, 5120),
    (128, 5120),
];

pub(super) struct Fixture {
    pub name: String,
    pub rows: usize,
    pub width: usize,
    pub epsilon: f32,
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub expected: Vec<f32>,
}

pub(super) fn fixtures() -> Result<Vec<Fixture>, String> {
    let mut result = Vec::with_capacity(SHAPES.len() + 1);
    for (rows, width) in SHAPES {
        result.push(make_fixture(rows, width, false)?);
    }
    result.push(make_fixture(3, 257, true)?);
    Ok(result)
}

fn make_fixture(rows: usize, width: usize, zero_input: bool) -> Result<Fixture, String> {
    let name = if zero_input {
        "zero-3x257".to_string()
    } else {
        format!("rows-{rows}-width-{width}")
    };
    let input = make_input(rows * width, zero_input)?;
    let weight = make_weight(width)?;
    let expected = rowwise_reference(&input, &weight, rows, width)?;
    Ok(Fixture {
        name,
        rows,
        width,
        epsilon: EPSILON,
        input,
        weight,
        expected,
    })
}

fn make_input(length: usize, zero_input: bool) -> Result<Vec<f32>, String> {
    if zero_input {
        return Ok(vec![0.0; length]);
    }
    let mut input = Vec::with_capacity(length);
    for index in 0..length {
        let residue = u8::try_from((index * 17 + 13) % 101)
            .map_err(|_| "RMSNorm input fixture value is out of range".to_string())?;
        input.push((f32::from(residue) - 50.0) / 16.0);
    }
    Ok(input)
}

fn make_weight(width: usize) -> Result<Vec<f32>, String> {
    let mut weight = Vec::with_capacity(width);
    for column in 0..width {
        let residue = u8::try_from((column * 7 + 3) % 29)
            .map_err(|_| "RMSNorm weight fixture value is out of range".to_string())?;
        weight.push((f32::from(residue) - 14.0) / 8.0);
    }
    Ok(weight)
}

fn rowwise_reference(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let mut expected = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width).take(rows) {
        expected.extend(crate::reference::rms_norm(row, weight, EPSILON)?);
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::{EPSILON, SHAPES, fixtures};

    #[test]
    fn fixtures_cover_width_boundaries_and_finite_signed_values() {
        let cases = fixtures().unwrap();
        assert_eq!(cases.len(), 9);
        for ((expected_rows, expected_width), case) in SHAPES.iter().zip(&cases[..8]) {
            assert_eq!((case.rows, case.width), (*expected_rows, *expected_width));
            assert_eq!(case.epsilon, EPSILON);
            assert_eq!(case.input.len(), case.rows * case.width);
            assert_eq!(case.weight.len(), case.width);
            assert_eq!(case.expected.len(), case.rows * case.width);
            assert!(case.input.iter().all(|value| value.is_finite()));
            assert!(case.weight.iter().all(|value| value.is_finite()));
            assert!(case.expected.iter().all(|value| value.is_finite()));
        }
        assert!(cases.iter().any(|case| case.width == 255));
        assert!(cases.iter().any(|case| case.width == 256));
        assert!(cases.iter().any(|case| case.width == 257));
        assert!(cases.iter().any(|case| case.width == 5120));
        assert!(
            cases
                .iter()
                .any(|case| case.expected.iter().any(|value| *value < 0.0))
        );
        assert!(
            cases
                .iter()
                .any(|case| case.expected.iter().any(|value| *value > 0.0))
        );
    }

    #[test]
    fn zero_input_has_exactly_zero_output() {
        let cases = fixtures().unwrap();
        let case = cases.iter().find(|case| case.name == "zero-3x257").unwrap();
        assert_eq!((case.rows, case.width), (3, 257));
        assert!(case.input.iter().all(|value| *value == 0.0));
        assert_eq!(case.expected, vec![0.0; 3 * 257]);
    }

    #[test]
    fn width_one_matches_the_hand_worked_rms_norm() {
        let cases = fixtures().unwrap();
        let case = &cases[0];
        let input = f64::from(case.input[0]);
        let weight = f64::from(case.weight[0]);
        let expected = (input / (input * input + f64::from(EPSILON)).sqrt() * weight) as f32;
        assert!((case.expected[0] - expected).abs() < 1e-7);
    }
}
