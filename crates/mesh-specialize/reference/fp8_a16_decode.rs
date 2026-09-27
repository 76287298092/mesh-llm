use anyhow::{Result, ensure};

/// Results from the independent FP64-dot A16 decode reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp8A16DecodeResult {
    /// Scaled values rounded once from the FP64 oracle to FP32.
    pub unrounded_fp32: Vec<f32>,
    /// The same FP32 values rounded to BF16 with round-to-nearest-even.
    pub output_bf16: Vec<u16>,
}

/// Evaluate one BF16 activation row against row-major E4M3FN weights.
///
/// This CPU oracle decodes each finite E4M3FN value from its logical sign,
/// exponent, and fraction fields. It accumulates BF16-activation products in
/// FP64, applies each represented BF16 output scale after the dot, then rounds
/// once to FP32 and once to BF16. The GPU candidate uses four FP32 FMA chains
/// per lane and a warp reduction, so results can differ, especially under
/// cancellation and near BF16 rounding boundaries.
pub fn run(
    input_bf16: &[u16],
    weight_e4m3: &[u8],
    weight_scale_bf16: &[u16],
    n: usize,
    k: usize,
) -> Result<Fp8A16DecodeResult> {
    ensure!(n > 0, "FP8 A16 decode requires at least one output row");
    ensure!(k > 0, "FP8 A16 decode requires a nonempty activation row");
    ensure!(
        input_bf16.len() == k,
        "FP8 A16 activation length differs from K"
    );
    ensure!(
        weight_scale_bf16.len() == n,
        "FP8 A16 scale count differs from N"
    );
    let weight_len = n
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("FP8 A16 weight size overflows usize"))?;
    ensure!(
        weight_e4m3.len() == weight_len,
        "FP8 A16 weight length differs from N*K"
    );
    ensure!(
        input_bf16.iter().all(|&bits| decode_bf16(bits).is_finite()),
        "FP8 A16 activations must be finite BF16 values"
    );
    ensure!(
        weight_scale_bf16
            .iter()
            .all(|&bits| decode_bf16(bits).is_finite()),
        "FP8 A16 scales must be finite BF16 values"
    );
    ensure!(
        weight_e4m3
            .iter()
            .all(|&code| decode_e4m3fn(code).is_some()),
        "FP8 A16 weights must be finite E4M3FN values"
    );

    let input: Vec<f64> = input_bf16
        .iter()
        .map(|&bits| f64::from(decode_bf16(bits)))
        .collect();
    let mut unrounded_fp32 = Vec::with_capacity(n);
    let mut output_bf16 = Vec::with_capacity(n);
    for (row, &scale_bits) in weight_scale_bf16.iter().enumerate() {
        let mut dot = 0.0_f64;
        let weight_start = row * k;
        for (&activation, &weight_code) in input
            .iter()
            .zip(&weight_e4m3[weight_start..weight_start + k])
        {
            let weight = decode_e4m3fn(weight_code).expect("all E4M3FN codes were validated above");
            dot += activation * weight;
        }
        let scale = f64::from(decode_bf16(scale_bits));
        let scaled = (dot * scale) as f32;
        ensure!(
            scaled.is_finite(),
            "FP8 A16 scaled output at row {row} overflows FP32"
        );
        unrounded_fp32.push(scaled);
        output_bf16.push(round_bf16_rne(scaled));
    }

    Ok(Fp8A16DecodeResult {
        unrounded_fp32,
        output_bf16,
    })
}

#[inline]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn decode_e4m3fn(code: u8) -> Option<f64> {
    let magnitude = code & 0x7f;
    if magnitude == 0x7f {
        return None;
    }
    let exponent = i32::from(magnitude >> 3);
    let fraction = i32::from(magnitude & 7);
    let value = if exponent == 0 {
        f64::from(fraction) / 512.0
    } else {
        f64::from(8 + fraction) * 2.0_f64.powi(exponent - 10)
    };
    Some(if code & 0x80 == 0 { value } else { -value })
}

#[inline]
fn round_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{decode_bf16, decode_e4m3fn, round_bf16_rne, run};

    #[test]
    fn zero_and_signed_scales_preserve_their_ieee_results() {
        let positive_zero = run(&[0x4040], &[0x38], &[0x0000], 1, 1).unwrap();
        assert_eq!(positive_zero.unrounded_fp32, [0.0]);
        assert_eq!(positive_zero.output_bf16, [0x0000]);

        let negative_zero = run(&[0x4040], &[0x38], &[0x8000], 1, 1).unwrap();
        assert_eq!(
            negative_zero.unrounded_fp32[0].to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(negative_zero.output_bf16, [0x8000]);

        let signed = run(&[0xc000, 0x3f80], &[0x38, 0x38], &[0xbf00], 1, 2).unwrap();
        assert_eq!(signed.unrounded_fp32, [0.5]);
        assert_eq!(signed.output_bf16, [0x3f00]);
    }

    #[test]
    fn decodes_subnormals_and_saturation_free_finite_extremes() {
        assert_eq!(decode_e4m3fn(0x01), Some(1.0 / 512.0));
        assert_eq!(decode_e4m3fn(0x81), Some(-1.0 / 512.0));
        assert_eq!(decode_e4m3fn(0x7e), Some(448.0));
        assert_eq!(decode_e4m3fn(0xff), None);

        let result = run(
            &[0x7f7f, 0x0001],
            &[0x01, 0x00, 0x00, 0x7e],
            &[0x3f80; 2],
            2,
            2,
        )
        .unwrap();
        assert!(result.unrounded_fp32.iter().all(|value| value.is_finite()));
        assert!(
            result
                .output_bf16
                .iter()
                .all(|&bits| decode_bf16(bits).is_finite())
        );
        assert!(result.unrounded_fp32[0] > 6.0e35);
        assert!(result.unrounded_fp32[1] > 4.0e-38);
    }

    #[test]
    fn includes_odd_k_tails_for_each_output_row() {
        let input = [0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0];
        let weights = [
            0x38, 0x38, 0x38, 0x38, 0x38, // 1, 1, 1, 1, 1
            0xb8, 0x30, 0x38, 0xb0, 0x40, // -1, 0.5, 1, -0.5, 2
        ];
        let result = run(&input, &weights, &[0x3f80; 2], 2, 5).unwrap();
        assert_eq!(result.unrounded_fp32, [15.0, 11.0]);
        assert_eq!(result.output_bf16, [0x4170, 0x4130]);
    }

    #[test]
    fn fp64_reference_retains_cancellation_terms() {
        let input = [0x4b80, 0x3f80, 0xcb80, 0x3f80]; // 2^24, 1, -2^24, 1
        let result = run(&input, &[0x38; 4], &[0x3f80], 1, 4).unwrap();
        assert_eq!(result.unrounded_fp32, [2.0]);
        assert_eq!(result.output_bf16, [0x4000]);
    }

    #[test]
    fn bf16_output_uses_round_to_nearest_even() {
        assert_eq!(round_bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(round_bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn rejects_nonfinite_fp8_and_shape_mismatches() {
        assert!(run(&[0x3f80], &[0x7f], &[0x3f80], 1, 1).is_err());
        assert!(run(&[0x3f80], &[0x38], &[0x3f80], 2, 1).is_err());
    }
}
