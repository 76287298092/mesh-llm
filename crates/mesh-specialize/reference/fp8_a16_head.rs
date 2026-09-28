//! Logical FP64 oracle for the separate small-batch A16 schedule.
use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct Fp8A16HeadResult {
    pub unrounded_fp32: Vec<f32>,
    pub output_bf16: Vec<u16>,
}

fn bf16(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn weight(code: u8) -> f64 {
    let exponent = i32::from((code >> 3) & 15);
    let fraction = f64::from(code & 7);
    let magnitude = if exponent == 0 {
        fraction / 512.0
    } else {
        (1.0 + fraction / 8.0) * 2.0_f64.powi(exponent - 7)
    };
    if code & 128 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn validate(
    input: &[u16],
    weights: &[u8],
    scales: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!((1..=8).contains(&m), "M must be in 1..=8");
    ensure!(
        (8..=262_144).contains(&n) && n.is_multiple_of(8),
        "N must be a multiple of 8 in 8..=262144"
    );
    ensure!(
        (16..=32_768).contains(&k) && k.is_multiple_of(16),
        "K must be a multiple of 16 in 16..=32768"
    );
    // M*K and M*N fit 32-bit usize; check the larger N*K product explicitly.
    let weight_len = n
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("N*K overflows usize"))?;
    ensure!(input.len() == m * k, "input extent differs from M*K");
    ensure!(
        weights.len() == weight_len,
        "weight extent differs from N*K"
    );
    ensure!(scales.len() == n, "scale extent differs from N");
    ensure!(
        input.iter().all(|&x| bf16(x).is_finite()),
        "nonfinite BF16 input"
    );
    ensure!(
        scales.iter().all(|&x| bf16(x).is_finite()),
        "nonfinite BF16 scale"
    );
    ensure!(
        weights.iter().all(|&x| x & 127 != 127),
        "nonfinite E4M3FN weight"
    );
    Ok(())
}

/// Accumulate logical represented products in FP64 without replaying GPU fragments
/// or slice reduction. Scale the FP64 dot in FP64, round once to FP32, then BF16
/// RNE. The GPU instead accumulates MMA FP32 partials, adds warp partials in order,
/// and multiplies by the represented scale with FP32 RN. This oracle is a
/// numerical reference, not a promise of bit equality with that association.
/// Negative scales and either signed zero are accepted. Reject nonfinite inputs,
/// FP32 output overflow and BF16 output overflow. No quantization of activations.
pub fn run(
    input: &[u16],
    weights: &[u8],
    scales: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Fp8A16HeadResult> {
    validate(input, weights, scales, m, n, k)?;
    let mut unrounded_fp32 = Vec::with_capacity(m * n);
    let mut output_bf16 = Vec::with_capacity(m * n);
    for token in 0..m {
        for column in 0..n {
            let mut dot = 0.0_f64;
            for kk in 0..k {
                dot += bf16(input[token * k + kk]) * weight(weights[column * k + kk]);
            }
            let scaled = (dot * bf16(scales[column])) as f32;
            ensure!(
                scaled.is_finite(),
                "FP32 overflow at token {token}, column {column}"
            );
            let rounded = round_bf16(scaled);
            ensure!(
                bf16(rounded).is_finite(),
                "BF16 overflow at token {token}, column {column}"
            );
            unrounded_fp32.push(scaled);
            output_bf16.push(rounded);
        }
    }
    Ok(Fp8A16HeadResult {
        unrounded_fp32,
        output_bf16,
    })
}

#[cfg(test)]
mod tests {
    use super::{round_bf16, run, weight};

    #[test]
    fn logical_rows_and_signed_scales() {
        for m in [1, 5, 8] {
            let mut input = vec![0x3f80; m * 16];
            for row in 0..m {
                input[row * 16] = 0x4000;
            }
            let result = run(
                &input,
                &[0x38; 128],
                &[0x3f80, 0xbf80, 0, 0x8000, 0x3f00, 0x4000, 0x3f80, 0x3f80],
                m,
                8,
                16,
            )
            .unwrap();
            for row in result.unrounded_fp32.as_chunks::<8>().0 {
                assert_eq!(*row, [17.0, -17.0, 0.0, -0.0, 8.5, 34.0, 17.0, 17.0]);
                assert_eq!(row[3].to_bits(), (-0.0_f32).to_bits());
            }
        }
    }

    #[test]
    fn zeros_and_cancellation_are_logical() {
        assert_eq!(
            run(&[0; 16], &[0x7e; 128], &[0x3f80; 8], 1, 8, 16)
                .unwrap()
                .output_bf16,
            [0; 8]
        );
        let mut input = [0; 16];
        input[..4].copy_from_slice(&[0x4b80, 0x3f80, 0xcb80, 0x3f80]);
        assert_eq!(
            run(&input, &[0x38; 128], &[0x3f80; 8], 1, 8, 16)
                .unwrap()
                .unrounded_fp32,
            [2.0; 8]
        );
    }

    #[test]
    fn finite_code_extremes_and_rounding() {
        assert_eq!(weight(1), 1.0 / 512.0);
        assert_eq!(weight(0x81), -1.0 / 512.0);
        assert_eq!(weight(0x7e), 448.0);
        assert_eq!(weight(0xfe), -448.0);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(round_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn rejects_extents_shapes_and_nonfinite_values() {
        for (m, n, k) in [
            (0, 8, 16),
            (9, 8, 16),
            (1, 7, 16),
            (1, 262152, 16),
            (1, 8, 15),
            (1, 8, 32784),
        ] {
            assert!(run(&[], &[], &[], m, n, k).is_err());
        }
        assert!(run(&[0; 15], &[0; 128], &[0; 8], 1, 8, 16).is_err());
        assert!(run(&[0; 16], &[0; 127], &[0; 8], 1, 8, 16).is_err());
        assert!(run(&[0; 16], &[0; 128], &[0; 7], 1, 8, 16).is_err());
        for bad in [0x7f80, 0xff80, 0x7fc0] {
            assert!(run(&[bad; 16], &[0; 128], &[0; 8], 1, 8, 16).is_err());
            assert!(run(&[0; 16], &[0; 128], &[bad; 8], 1, 8, 16).is_err());
        }
        for bad in [0x7f, 0xff] {
            assert!(run(&[0; 16], &[bad; 128], &[0; 8], 1, 8, 16).is_err());
        }
        assert!(run(&[0x7f7f; 16], &[0x7e; 128], &[0x3f80; 8], 1, 8, 16).is_err());
    }
}
