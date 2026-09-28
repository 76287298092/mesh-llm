//! Isolated exact-order variant of exponential.rs without an iterated coefficient array.
//! The original helper remains the control. Device equivalence requires raw-bit proof.
use core::f64::consts::{LN_2, LOG2_E};

#[cfg(target_arch = "nvptx64")]
use super::causal_attention::{add_rn, multiply_rn, subtract_rn};

// Host-only operation-order checks use ordinary strict FP64 arithmetic, not a GPU oracle.
#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
fn add_rn(left: f64, right: f64) -> f64 {
    left + right
}
#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
fn multiply_rn(left: f64, right: f64) -> f64 {
    left * right
}
#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
fn subtract_rn(left: f64, right: f64) -> f64 {
    left - right
}

/// Same branch order, range reduction, coefficient bits and rounding stages as control.
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

    let exponent = subtract_rn(multiply_rn(value, LOG2_E), 0.5) as i32;
    let reduced = subtract_rn(value, multiply_rn(f64::from(exponent), LN_2));
    let polynomial = horner(reduced);
    let first_exponent = exponent / 2;
    let second_exponent = exponent - first_exponent;
    let first_factor = normal_power_of_two(first_exponent);
    let second_factor = normal_power_of_two(second_exponent);
    multiply_rn(multiply_rn(polynomial, first_factor), second_factor)
}

#[inline(always)]
fn horner(reduced: f64) -> f64 {
    // Exact binary64 encodings of the control's 1/16! through 1/0! constants.
    // Keep the initial +0 multiplication and all 17 separate multiply/add stages.
    let mut p = 0.0_f64;
    p = add_rn(
        f64::from_bits(0x3d2a_e7f3_e733_b81f),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3d6a_e7f3_e733_b81f),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3da9_3974_a8c0_7c9d),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3de6_1246_13a8_6d09),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3e21_eed8_eff8_d898),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3e5a_e645_67f5_44e4),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3e92_7e4f_b778_9f5c),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3ec7_1de3_a556_c734),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3efa_01a0_1a01_a01a),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3f2a_01a0_1a01_a01a),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3f56_c16c_16c1_6c17),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3f81_1111_1111_1111),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3fa5_5555_5555_5555),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3fc5_5555_5555_5555),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3fe0_0000_0000_0000),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3ff0_0000_0000_0000),
        multiply_rn(reduced, p),
    );
    p = add_rn(
        f64::from_bits(0x3ff0_0000_0000_0000),
        multiply_rn(reduced, p),
    );
    p
}

#[inline(always)]
fn normal_power_of_two(exponent: i32) -> f64 {
    let biased_exponent = (1023 + exponent) as u64;
    f64::from_bits(biased_exponent << 52)
}
