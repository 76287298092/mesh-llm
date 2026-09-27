//! no_std-compatible stable FP32 SiLU arithmetic.

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

/// Evaluate SiLU with stable FP64 logistic arithmetic and a bounded exp polynomial.
#[inline(always)]
pub fn silu(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    if value == f32::INFINITY {
        return value;
    }
    if value == f32::NEG_INFINITY {
        return -0.0;
    }

    let wide = f64::from(value);
    let magnitude = wide.abs();
    if magnitude >= 128.0 {
        return if wide.is_sign_negative() { -0.0 } else { value };
    }

    let exponential = exp_negative_magnitude(magnitude);
    let sigmoid = if wide >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    (wide * sigmoid) as f32
}

#[inline(always)]
fn exp_negative_magnitude(magnitude: f64) -> f64 {
    let value = -magnitude;
    let exponent = (value * LOG2_E - 0.5) as i32;
    let reduced = value - f64::from(exponent) * LN_2;
    let mut polynomial = 0.0_f64;
    for coefficient in EXP_TAYLOR_COEFFICIENTS {
        polynomial = coefficient + reduced * polynomial;
    }
    let power_exponent = (1023 + exponent) as u64;
    let power_of_two = f64::from_bits(power_exponent << 52);
    polynomial * power_of_two
}

#[cfg(test)]
mod tests {
    use super::silu;
    use crate::{causal_conv4_reference, entry_reference::round_bf16};

    #[test]
    fn all_finite_bf16_inputs_match_independent_silu() {
        let minimum_subnormal = f64::from(f32::from_bits(1));
        for bits in 0_u32..=u16::MAX as u32 {
            let input = f32::from_bits(bits << 16);
            if !input.is_finite() {
                continue;
            }
            let actual = silu(input);
            let expected = causal_conv4_reference::silu(input);
            assert_eq!(
                round_bf16(actual),
                round_bf16(expected),
                "BF16 rounding differs for input bits {bits:#06x}"
            );
            let error = (f64::from(actual) - f64::from(expected)).abs();
            let tolerance = minimum_subnormal + 2.0e-7 * f64::from(expected.abs());
            assert!(
                error <= tolerance,
                "SiLU differs for input bits {bits:#06x}: actual={actual:?}, expected={expected:?}, error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn preserves_signed_zero_and_matches_negative_one_over_256_rounding() {
        assert_eq!(silu(0.0).to_bits(), 0);
        assert_eq!(silu(-0.0).to_bits(), 0x8000_0000);
        let input = f32::from_bits(0xbb80_0000);
        assert_eq!(input, -1.0 / 256.0);
        assert_eq!(
            round_bf16(silu(input)),
            round_bf16(causal_conv4_reference::silu(input))
        );
    }

    #[test]
    fn handles_nonfinite_values_explicitly() {
        assert_eq!(silu(f32::INFINITY), f32::INFINITY);
        assert_eq!(silu(f32::NEG_INFINITY).to_bits(), 0x8000_0000);
        assert!(silu(f32::NAN).is_nan());
    }
}
