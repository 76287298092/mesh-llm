//! Independent reference for the FP32-local-scale NVFP4 quantization profile.
//!
//! This models per-16-value E4M3FN scales and E2M1 payloads. It does not claim
//! bit parity with BF16 fake-quantization helpers.

use anyhow::{Context, Result, ensure};

use crate::{entry_reference::bf16_to_f32, projection_reference};

const GROUP_WIDTH: usize = 16;
const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, PartialEq)]
pub struct Quantized {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub effective: Vec<f32>,
}

/// Quantize BF16 activations with one encoded E4M3FN scale per 16 values.
///
/// Local scales are computed as FP32 `(amax / 6) * global_scale`; decoded
/// effective scales divide the stored E4M3FN scale by `global_scale`.
pub fn run(input: &[u16], rows: usize, width: usize, global_scale: f32) -> Result<Quantized> {
    let (input_len, scale_len) = validate(input, rows, width, global_scale)?;
    let mut result = Quantized {
        packed: Vec::with_capacity(input_len / 2),
        scales: Vec::with_capacity(scale_len),
        effective: Vec::with_capacity(scale_len),
    };
    let mut codes = Vec::with_capacity(input_len);

    for row in 0..rows {
        let row_start = row * width;
        for group in 0..width / GROUP_WIDTH {
            let start = row_start + group * GROUP_WIDTH;
            let values = &input[start..start + GROUP_WIDTH];
            let amax = values
                .iter()
                .map(|&bits| bf16_to_f32(bits).abs())
                .fold(0.0_f32, f32::max);
            let local_scale = (amax / 6.0_f32) * global_scale;
            ensure!(local_scale.is_finite(), "NVFP4 local scale overflows FP32");
            let encoded = projection_reference::encode(local_scale)?;
            let scale = if encoded == 0 { 0x20 } else { encoded };
            let effective = projection_reference::decode(scale) / global_scale;
            ensure!(
                effective.is_finite() && effective > 0.0,
                "NVFP4 effective scale is not positive and finite"
            );
            result.scales.push(scale);
            result.effective.push(effective);
            for &bits in values {
                let scaled = bf16_to_f32(bits) / effective;
                ensure!(!scaled.is_nan(), "NVFP4 scaled input is NaN");
                codes.push(encode_e2m1(scaled));
            }
        }
    }
    for pair in codes.as_chunks::<2>().0 {
        result.packed.push(pair[0] | (pair[1] << 4));
    }
    Ok(result)
}

fn validate(input: &[u16], rows: usize, width: usize, global_scale: f32) -> Result<(usize, usize)> {
    ensure!((1..=2048).contains(&rows), "invalid NVFP4 row count");
    ensure!(
        (16..=32768).contains(&width) && width.is_multiple_of(GROUP_WIDTH),
        "NVFP4 width must be a multiple of 16 in 16..=32768"
    );
    ensure!(
        global_scale.is_finite() && global_scale > 0.0,
        "NVFP4 global scale must be positive and finite"
    );
    let input_len = rows
        .checked_mul(width)
        .context("NVFP4 input extent overflows usize")?;
    let scale_len = rows
        .checked_mul(width / GROUP_WIDTH)
        .context("NVFP4 scale extent overflows usize")?;
    ensure!(input.len() == input_len, "NVFP4 input extent mismatch");
    ensure!(
        input.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "NVFP4 input must contain finite BF16 values"
    );
    Ok((input_len, scale_len))
}

fn encode_e2m1(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x08 } else { 0 };
    let magnitude = value.abs().min(6.0);
    let mut best = 0_u8;
    let mut error = f32::INFINITY;
    for (code, level) in E2M1_LEVELS.iter().copied().enumerate() {
        let distance = (magnitude - level).abs();
        if distance < error || (distance == error && code & 1 == 0) {
            best = code as u8;
            error = distance;
        }
    }
    sign | best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry_reference::round_bf16;

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    #[test]
    fn hand_computed_fp4_midpoints_anchor_scale_and_preserve_negative_zero() {
        let input = [
            bf16(0.25),
            bf16(0.75),
            bf16(1.25),
            bf16(1.75),
            bf16(2.5),
            bf16(3.5),
            bf16(5.0),
            0x8000,
            bf16(-0.25),
            bf16(-0.75),
            bf16(-1.25),
            bf16(-1.75),
            bf16(-2.5),
            bf16(-3.5),
            bf16(-5.0),
            bf16(6.0),
        ];
        let result = run(&input, 1, 16, 1.0).unwrap();
        assert_eq!(result.scales, [0x38]);
        assert_eq!(result.effective, [1.0]);
        assert_eq!(
            result.packed,
            [0x20, 0x42, 0x64, 0x86, 0xa8, 0xca, 0xec, 0x7e]
        );
    }

    #[test]
    fn e4m3_scale_encoder_uses_even_code_at_fp8_midpoints() {
        let mut input = [bf16(0.0); 16];
        input[0] = bf16(6.0);
        let first = run(&input, 1, 16, 1.0625).unwrap();
        let second = run(&input, 1, 16, 1.1875).unwrap();
        assert_eq!(first.scales, [0x38]);
        assert_eq!(second.scales, [0x3a]);
        assert_eq!(
            first.effective,
            [projection_reference::decode(0x38) / 1.0625]
        );
        assert_eq!(
            second.effective,
            [projection_reference::decode(0x3a) / 1.1875]
        );
    }

    #[test]
    fn zero_groups_use_fallback_and_rows_keep_group_order() {
        let mut input = vec![bf16(0.0); 64];
        input[0] = bf16(6.0);
        input[16] = bf16(3.0);
        input[48] = bf16(1.5);
        let result = run(&input, 2, 32, 1.0).unwrap();
        assert_eq!(result.scales, [0x38, 0x30, 0x20, 0x28]);
        assert_eq!(result.effective, [1.0, 0.5, 0.125, 0.25]);
        assert_eq!(result.packed.len(), 32);
    }

    #[test]
    fn scale_saturation_clamps_to_fp4_six_and_local_overflow_is_rejected() {
        let mut input = [bf16(0.0); 16];
        input[0] = bf16(6.0);
        let saturated = run(&input, 1, 16, 1000.0).unwrap();
        assert_eq!(saturated.scales, [0x7e]);
        assert_eq!(saturated.packed[0] & 0x0f, 0x07);

        let finite_max = [0x7f7f; 16];
        assert!(run(&finite_max, 1, 16, 40.0).is_err());
    }

    #[test]
    fn rejects_bad_dimensions_extent_values_scales_and_small_global_overflow() {
        let input = [bf16(1.0); 16];
        assert!(run(&input, 0, 16, 1.0).is_err());
        assert!(run(&input, 1, 15, 1.0).is_err());
        assert!(run(&input, 1, 16, 0.0).is_err());
        assert!(run(&input, 1, 16, f32::NAN).is_err());
        assert!(run(&input[..15], 1, 16, 1.0).is_err());
        let mut nonfinite = input;
        nonfinite[0] = 0x7f80;
        assert!(run(&nonfinite, 1, 16, 1.0).is_err());
        let mut small_global_input = [bf16(0.0); 16];
        small_global_input[0] = bf16(6.0);
        assert!(run(&small_global_input, 1, 16, 1e-40).is_err());
    }
}
