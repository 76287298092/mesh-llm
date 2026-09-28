//! no_std FP32 exponential and FP64 logarithm for row log-softmax scoring.
//!
//! Pure Rust, no inline assembly. The device compiler may contract the Horner
//! steps into fused multiply-adds; the bounds below hold either way.

use core::f32::consts::LOG2_E;
use core::f64::consts::{LN_2, SQRT_2};

/// High part of ln(2) with 16 significant bits, so `n * LN2_HI` is exact for |n| <= 255.
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
/// Degree-7 Taylor coefficients of exp, highest degree first.
const EXP_COEFFICIENTS: [f32; 8] = [
    1.0 / 5040.0,
    1.0 / 720.0,
    1.0 / 120.0,
    1.0 / 24.0,
    1.0 / 6.0,
    0.5,
    1.0,
    1.0,
];

/// Evaluate `exp(x)` in FP32 for `x <= 0`.
///
/// Relative error is a few FP32 ulps for `-87 <= x <= 0`. Inputs below -87
/// return zero: they contribute less than 2^-125 to a softmax denominator that
/// always includes the maximum term 1. Positive inputs are outside the contract
/// and return one; NaN propagates.
#[inline(always)]
pub fn exp_nonpositive_f32(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    if value >= 0.0 {
        return 1.0;
    }
    if value < -87.0 {
        return 0.0;
    }
    // Truncation of (t - 0.5) rounds a negative t to nearest.
    let exponent = (value * LOG2_E - 0.5) as i32;
    let scale = exponent as f32;
    let reduced = (value - scale * LN2_HI) - scale * LN2_LO;
    let mut polynomial = 0.0_f32;
    for coefficient in EXP_COEFFICIENTS {
        polynomial = polynomial * reduced + coefficient;
    }
    // exponent is in -126..=0, so the power of two is a normal FP32 value.
    let power = f32::from_bits(((127 + exponent) as u32) << 23);
    polynomial * power
}

/// Natural logarithm of a positive, finite, normal FP64 value.
///
/// Uses `x = m * 2^e` with `m` in `[sqrt(1/2), sqrt(2)]` and the odd
/// `2 * atanh((m - 1) / (m + 1))` series to degree 27. Absolute error is a few
/// FP64 ulps of `e * ln(2)`. Zero, negative, subnormal and nonfinite inputs are
/// outside the contract and return NaN.
#[inline(always)]
pub fn ln_positive_f64(value: f64) -> f64 {
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if value.is_nan() || value <= 0.0 || biased == 0 || biased == 0x7ff {
        return f64::NAN;
    }
    let mut exponent = biased - 1023;
    let mut mantissa = f64::from_bits((bits & ((1_u64 << 52) - 1)) | (1023_u64 << 52));
    if mantissa > SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let ratio = (mantissa - 1.0) / (mantissa + 1.0);
    let square = ratio * ratio;
    let mut series = 0.0_f64;
    for odd in (1..=27_u32).rev().step_by(2) {
        series = series * square + 1.0 / f64::from(odd);
    }
    f64::from(exponent) * LN_2 + 2.0 * ratio * series
}

#[cfg(test)]
mod tests {
    use super::{exp_nonpositive_f32, ln_positive_f64};

    #[test]
    fn exponential_matches_host_within_eight_ulps() {
        let mut value = 0.0_f32;
        while value > -87.0 {
            let actual = f64::from(exp_nonpositive_f32(value));
            let expected = f64::from(value).exp();
            let ulp = expected * f64::from(f32::EPSILON);
            assert!(
                (actual - expected).abs() <= 8.0 * ulp,
                "exp({value}) = {actual:e}, expected {expected:e}"
            );
            value -= 0.013_7;
        }
        assert_eq!(exp_nonpositive_f32(0.0), 1.0);
        assert_eq!(exp_nonpositive_f32(-0.0), 1.0);
        assert_eq!(exp_nonpositive_f32(-87.5), 0.0);
        assert_eq!(exp_nonpositive_f32(f32::NEG_INFINITY), 0.0);
        assert!(exp_nonpositive_f32(f32::NAN).is_nan());
    }

    #[test]
    fn logarithm_matches_host() {
        let mut value = 1.0_f64;
        while value < 1.0e7 {
            let actual = ln_positive_f64(value);
            let expected = value.ln();
            assert!(
                (actual - expected).abs() <= 4.0e-15 * expected.abs().max(1.0),
                "ln({value}) = {actual:e}, expected {expected:e}"
            );
            value *= 1.000_731;
        }
        for value in [0.5, 0.707, 1.414_3, 2.0, 1.0e-300, 1.0e300] {
            let expected = f64::ln(value);
            assert!((ln_positive_f64(value) - expected).abs() <= 1.0e-13 * expected.abs().max(1.0));
        }
        assert_eq!(ln_positive_f64(1.0), 0.0);
        for value in [0.0, -1.0, f64::INFINITY, f64::NAN, f64::from_bits(1)] {
            assert!(ln_positive_f64(value).is_nan());
        }
    }
}
