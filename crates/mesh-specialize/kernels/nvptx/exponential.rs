//! no_std-compatible FP64 exponential for nonpositive attention weights.

use core::f64::consts::{LN_2, LOG2_E};

const EXP_TAYLOR_COEFFICIENTS: [f64; 17] = [
    1.0 / 20_922_789_888_000.0,
    1.0 / 1_307_674_368_000.0,
    1.0 / 87_178_291_200.0,
    1.0 / 6_227_020_800.0,
    1.0 / 479_001_600.0,
    1.0 / 39_916_800.0,
    1.0 / 3_628_800.0,
    1.0 / 362_880.0,
    1.0 / 40_320.0,
    1.0 / 5_040.0,
    1.0 / 720.0,
    1.0 / 120.0,
    1.0 / 24.0,
    1.0 / 6.0,
    1.0 / 2.0,
    1.0,
    1.0,
];

/// Evaluate `exp(x)` for nonpositive FP64 `x` using range reduction and degree-16 Taylor.
///
/// The attention online-softmax path calls this only with `x <= 0`. Positive inputs
/// are outside the contract and return one, matching the zero endpoint behavior.
#[inline(always)]
pub fn exp_nonpositive(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    if value >= 0.0 {
        return 1.0;
    }
    if value < -745.0 {
        return 0.0;
    }

    let exponent = (value * LOG2_E - 0.5) as i32;
    let reduced = value - f64::from(exponent) * LN_2;
    let mut polynomial = 0.0_f64;
    for coefficient in EXP_TAYLOR_COEFFICIENTS {
        polynomial = coefficient + reduced * polynomial;
    }

    let first_exponent = exponent / 2;
    let second_exponent = exponent - first_exponent;
    let first_factor = normal_power_of_two(first_exponent);
    let second_factor = normal_power_of_two(second_exponent);
    (polynomial * first_factor) * second_factor
}

#[inline(always)]
fn normal_power_of_two(exponent: i32) -> f64 {
    let biased_exponent = (1023 + exponent) as u64;
    f64::from_bits(biased_exponent << 52)
}

#[cfg(test)]
mod tests {
    use super::exp_nonpositive;

    fn assert_relative_error(value: f64) {
        let actual = exp_nonpositive(value);
        let expected = value.exp();
        let error = (actual - expected).abs();
        let tolerance = if expected < f64::MIN_POSITIVE {
            f64::from_bits(1)
        } else {
            5.0e-14 * expected.abs()
        };
        assert!(
            error <= tolerance,
            "exp({value}) actual={actual:e}, expected={expected:e}, error={error:e}, tolerance={tolerance:e}"
        );
    }

    #[test]
    fn known_nonpositive_points_match_f64_exp() {
        for value in [0.0, -1.0, -0.01, -128.0, -700.0, -744.0] {
            assert_relative_error(value);
        }
        assert_relative_error(-745.0);
        assert_eq!(exp_nonpositive(-745.000_001), 0.0);
        assert_eq!(exp_nonpositive(f64::NEG_INFINITY), 0.0);
        assert!(exp_nonpositive(f64::NAN).is_nan());
    }

    #[test]
    fn dense_nonpositive_grid_matches_f64_exp() {
        for step in 0..=12_800 {
            let value = -128.0 + f64::from(step) * 0.01;
            assert_relative_error(value);
        }
    }
}
