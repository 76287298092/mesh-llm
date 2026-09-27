//! Independent logical reference for packed NVFP4 matrix multiplication.

use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    projection_reference,
};

const GROUP_WIDTH: usize = 16;
const GROUP_BYTES: usize = GROUP_WIDTH / 2;
const E2M1_LEVELS: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u8],
    pub rows: usize,
    pub global: f32,
}

/// Multiply logical E2M1 matrices, then apply the reciprocal global-scale product.
pub fn run(
    input: Matrix<'_>,
    weights: Matrix<'_>,
    width: usize,
) -> Result<projection_reference::LinearReference> {
    ensure!(
        (GROUP_WIDTH..=32768).contains(&width) && width.is_multiple_of(GROUP_WIDTH),
        "NVFP4 linear width must be a multiple of 16 in 16..=32768"
    );
    validate_matrix(&input, width, 1, 2048, "input")?;
    validate_matrix(&weights, width, 1, 32768, "weight")?;

    let global_product = input.global * weights.global;
    ensure!(
        global_product.is_finite() && global_product > 0.0,
        "NVFP4 global-scale product must be positive and finite"
    );
    let global_factor = 1.0_f32 / global_product;
    ensure!(
        global_factor.is_finite() && global_factor > 0.0,
        "NVFP4 global-scale factor must be positive and finite"
    );

    let count = input
        .rows
        .checked_mul(weights.rows)
        .context("NVFP4 linear output extent overflows usize")?;
    let mut output = projection_reference::LinearReference {
        unrounded: Vec::new(),
        normalized: Vec::new(),
        absolute_sums: Vec::new(),
    };
    output
        .unrounded
        .try_reserve_exact(count)
        .context("cannot reserve NVFP4 linear outputs")?;
    output
        .normalized
        .try_reserve_exact(count)
        .context("cannot reserve NVFP4 linear BF16 outputs")?;
    output
        .absolute_sums
        .try_reserve_exact(count)
        .context("cannot reserve NVFP4 linear absolute sums")?;

    accumulate_rows(input, weights, width, global_factor, &mut output)?;
    Ok(output)
}

fn validate_matrix(
    matrix: &Matrix<'_>,
    width: usize,
    min_rows: usize,
    max_rows: usize,
    name: &str,
) -> Result<()> {
    ensure!(
        (min_rows..=max_rows).contains(&matrix.rows),
        "invalid NVFP4 {name} row count"
    );
    ensure!(
        matrix.global.is_finite() && matrix.global > 0.0,
        "NVFP4 {name} global scale must be positive and finite"
    );
    let packed_len = matrix
        .rows
        .checked_mul(width / 2)
        .context("NVFP4 packed matrix extent overflows usize")?;
    let scale_len = matrix
        .rows
        .checked_mul(width / GROUP_WIDTH)
        .context("NVFP4 scale matrix extent overflows usize")?;
    ensure!(
        matrix.packed.len() == packed_len,
        "NVFP4 {name} packed extent mismatch"
    );
    ensure!(
        matrix.scales.len() == scale_len,
        "NVFP4 {name} scale extent mismatch"
    );
    ensure!(
        matrix.scales.iter().all(|&code| code <= 126),
        "NVFP4 {name} scales must be finite nonnegative E4M3FN codes"
    );
    Ok(())
}

fn accumulate_rows(
    input: Matrix<'_>,
    weights: Matrix<'_>,
    width: usize,
    global_factor: f32,
    output: &mut projection_reference::LinearReference,
) -> Result<()> {
    let groups = width / GROUP_WIDTH;
    let packed_row_len = width / 2;
    for (input_packed, input_scales) in input
        .packed
        .chunks_exact(packed_row_len)
        .zip(input.scales.chunks_exact(groups))
    {
        for (weight_packed, weight_scales) in weights
            .packed
            .chunks_exact(packed_row_len)
            .zip(weights.scales.chunks_exact(groups))
        {
            let (raw_sum, absolute_sum) =
                dot(input_packed, input_scales, weight_packed, weight_scales);
            let raw_fp32 = raw_sum as f32;
            ensure!(raw_fp32.is_finite(), "NVFP4 raw dot overflows FP32");
            let scaled = raw_fp32 * global_factor;
            ensure!(scaled.is_finite(), "NVFP4 scaled dot overflows FP32");
            let normalized = round_bf16(scaled);
            ensure!(
                bf16_to_f32(normalized).is_finite(),
                "NVFP4 linear output overflows BF16"
            );
            let absolute_scaled = absolute_sum * f64::from(global_factor);
            ensure!(
                absolute_scaled.is_finite(),
                "NVFP4 absolute sum overflows f64"
            );
            output.unrounded.push(scaled);
            output.normalized.push(normalized);
            output.absolute_sums.push(absolute_scaled);
        }
    }
    Ok(())
}

fn dot(
    input_packed: &[u8],
    input_scales: &[u8],
    weight_packed: &[u8],
    weight_scales: &[u8],
) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for (group, (&input_code, &weight_code)) in input_scales.iter().zip(weight_scales).enumerate() {
        let input_scale = f64::from(projection_reference::decode(input_code));
        let weight_scale = f64::from(projection_reference::decode(weight_code));
        let byte_start = group * GROUP_BYTES;
        let input_group = &input_packed[byte_start..byte_start + GROUP_BYTES];
        let weight_group = &weight_packed[byte_start..byte_start + GROUP_BYTES];
        for (&input_byte, &weight_byte) in input_group.iter().zip(weight_group) {
            for shift in [0, 4] {
                let input_code = (input_byte >> shift) & 0x0f;
                let weight_code = (weight_byte >> shift) & 0x0f;
                let term =
                    decode_e2m1(input_code) * input_scale * decode_e2m1(weight_code) * weight_scale;
                sum += term;
                absolute_sum += term.abs();
            }
        }
    }
    (sum, absolute_sum)
}

fn decode_e2m1(code: u8) -> f64 {
    let magnitude = E2M1_LEVELS[usize::from(code & 0x07)];
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(codes: &[u8]) -> Vec<u8> {
        codes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| pair[0] | (pair[1] << 4))
            .collect()
    }

    fn matrix<'a>(packed: &'a [u8], scales: &'a [u8], rows: usize, global: f32) -> Matrix<'a> {
        Matrix {
            packed,
            scales,
            rows,
            global,
        }
    }

    #[test]
    fn signed_packing_and_nonuniform_group_scales_match_hand_computed_dots() {
        let input_packed = [0x21, 0xfa, 0, 0, 0, 0, 0, 0, 0x31, 0, 0, 0, 0, 0, 0, 0];
        let weight_packed = [
            0x12, 0x12, 0, 0, 0, 0, 0, 0, 0x21, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x02, 0, 0, 0, 0, 0, 0,
            0x12, 0, 0, 0, 0, 0, 0, 0,
        ];
        let input_scales = [0x38, 0x40];
        let weight_scales = [0x38, 0x48, 0x40, 0x30];
        let result = run(
            matrix(&input_packed, &input_scales, 1, 1.0),
            matrix(&weight_packed, &weight_scales, 2, 1.0),
            32,
        )
        .unwrap();
        assert_eq!(result.unrounded, [11.0, -0.25]);
        assert_eq!(result.absolute_sums, [19.0, 3.75]);
        assert_eq!(result.normalized, [round_bf16(11.0), round_bf16(-0.25)]);
    }

    #[test]
    fn reciprocal_global_product_scales_the_dot_in_the_declared_direction() {
        let packed = [0x02, 0, 0, 0, 0, 0, 0, 0];
        let scales = [0x38];
        let result = run(
            matrix(&packed, &scales, 1, 2.0),
            matrix(&packed, &scales, 1, 4.0),
            16,
        )
        .unwrap();
        assert_eq!(result.unrounded, [0.125]);
        assert_eq!(result.absolute_sums, [0.125]);
        assert_eq!(result.normalized, [round_bf16(0.125)]);
    }

    #[test]
    fn rectangular_m_n_tails_and_k80_keep_logical_row_order() {
        let input_codes = [2_u8; 3 * 80];
        let weight_codes = [1_u8; 5 * 80];
        let input_packed = packed(&input_codes);
        let weight_packed = packed(&weight_codes);
        let input_scales = [0x38; 3 * 5];
        let weight_scales = [0x38; 5 * 5];
        let result = run(
            matrix(&input_packed, &input_scales, 3, 1.0),
            matrix(&weight_packed, &weight_scales, 5, 1.0),
            80,
        )
        .unwrap();
        assert_eq!(result.unrounded, [40.0; 15]);
        assert_eq!(result.absolute_sums, [40.0; 15]);
        assert_eq!(result.normalized, [round_bf16(40.0); 15]);
    }

    #[test]
    fn zero_scales_and_negative_zero_payloads_are_valid() {
        let negative_zero = [0x08, 0, 0, 0, 0, 0, 0, 0];
        let positive_one = [0x02, 0, 0, 0, 0, 0, 0, 0];
        let zero_scale = [0];
        let unit_scale = [0x38];
        let first = run(
            matrix(&negative_zero, &unit_scale, 1, 1.0),
            matrix(&positive_one, &zero_scale, 1, 1.0),
            16,
        )
        .unwrap();
        let second = run(
            matrix(&positive_one, &zero_scale, 1, 1.0),
            matrix(&positive_one, &unit_scale, 1, 1.0),
            16,
        )
        .unwrap();
        assert_eq!(first.unrounded, [0.0]);
        assert_eq!(first.absolute_sums, [0.0]);
        assert_eq!(second.unrounded, [0.0]);
        assert_eq!(second.absolute_sums, [0.0]);
    }

    #[test]
    fn rejects_bad_dimensions_extents_and_scale_codes() {
        let packed = [0x02, 0, 0, 0, 0, 0, 0, 0];
        let scale = [0x38];
        let valid = matrix(&packed, &scale, 1, 1.0);
        assert!(run(valid, valid, 15).is_err());
        assert!(run(valid, valid, 32784).is_err());
        assert!(run(matrix(&packed, &scale, 0, 1.0), valid, 16).is_err());
        assert!(run(matrix(&packed, &scale, 2049, 1.0), valid, 16).is_err());
        assert!(run(valid, matrix(&packed, &scale, 0, 1.0), 16).is_err());
        assert!(run(valid, matrix(&packed, &scale, 32769, 1.0), 16).is_err());
        assert!(run(matrix(&packed[..7], &scale, 1, 1.0), valid, 16).is_err());
        assert!(run(valid, matrix(&packed[..7], &scale, 1, 1.0), 16).is_err());
        assert!(run(matrix(&packed, &[], 1, 1.0), valid, 16).is_err());
        assert!(run(valid, matrix(&packed, &[], 1, 1.0), 16).is_err());
        for invalid_scale in [0x7f, 0x80, 0xff] {
            assert!(run(matrix(&packed, &[invalid_scale], 1, 1.0), valid, 16).is_err());
            assert!(run(valid, matrix(&packed, &[invalid_scale], 1, 1.0), 16).is_err());
        }
    }

    #[test]
    fn rejects_invalid_global_scales_and_output_overflow() {
        let packed = [0x02, 0, 0, 0, 0, 0, 0, 0];
        let scale = [0x38];
        let valid = matrix(&packed, &scale, 1, 1.0);
        for invalid_global in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(run(matrix(&packed, &scale, 1, invalid_global), valid, 16).is_err());
            assert!(run(valid, matrix(&packed, &scale, 1, invalid_global), 16).is_err());
        }
        assert!(
            run(
                matrix(&packed, &scale, 1, f32::MAX),
                matrix(&packed, &scale, 1, 2.0),
                16
            )
            .is_err()
        );
        assert!(run(matrix(&packed, &scale, 1, f32::from_bits(1)), valid, 16).is_err());

        let largest_codes = [0x07, 0, 0, 0, 0, 0, 0, 0];
        assert!(
            run(
                matrix(&largest_codes, &scale, 1, 1.0),
                matrix(&largest_codes, &scale, 1, 5.0e-39),
                16,
            )
            .is_err()
        );

        let bf16_overflow_global = 1.058e-37;
        assert!(
            run(
                matrix(&largest_codes, &scale, 1, 1.0),
                matrix(&largest_codes, &scale, 1, bf16_overflow_global),
                16,
            )
            .is_err()
        );
    }
}
