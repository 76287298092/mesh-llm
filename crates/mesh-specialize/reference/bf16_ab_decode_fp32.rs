//! Independent sequential FP64 oracle for the experimental BF16 A/B projection.
use crate::entry_reference::{bf16_to_f32, round_bf16};
use anyhow::{Result, ensure};

#[derive(Debug)]
pub struct Projection {
    pub raw: Vec<f32>,
    pub bf16: Vec<u16>,
    pub absolute_products: Vec<f64>,
}

/// Logical row-major dot products. No device tile, vector, or reduction simulation.
pub fn run(input: &[u16], weights: &[u16], n: usize, k: usize) -> Result<Projection> {
    ensure!((1..=256).contains(&n), "A/B N must be in 1..=256");
    ensure!(
        (8..=32768).contains(&k) && k.is_multiple_of(8),
        "A/B K must be a multiple of eight in 8..=32768"
    );
    ensure!(
        input.len() == k && weights.len() == n * k,
        "A/B extent mismatch"
    );
    ensure!(
        input
            .iter()
            .chain(weights)
            .all(|&v| bf16_to_f32(v).is_finite()),
        "A/B inputs must be finite"
    );
    let mut result = Projection {
        raw: Vec::with_capacity(n),
        bf16: Vec::with_capacity(n),
        absolute_products: Vec::with_capacity(n),
    };
    for row in weights.chunks_exact(k) {
        let mut sum = 0.0_f64;
        let mut absolute = 0.0_f64;
        for (&x, &w) in input.iter().zip(row) {
            let product = f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w));
            sum += product;
            absolute += product.abs();
        }
        let raw = sum as f32;
        let bf16 = round_bf16(raw);
        ensure!(
            raw.is_finite() && bf16_to_f32(bf16).is_finite(),
            "A/B oracle output overflow"
        );
        result.raw.push(raw);
        result.bf16.push(bf16);
        result.absolute_products.push(absolute);
    }
    Ok(result)
}

/// Conservative forward-error envelope for normal, non-overflowing FP32 dots.
/// Four chains: <=2*ceil(K/1024) FMAs, two chain adds, five warp adds, five final
/// warp adds (including zero partners). Two extra roundings cover oracle cast
/// and FP64 reference rounding. Absolute floor covers gradual underflow for
/// this bounded K. This is an operator gate, not a model-quality allowance.
pub fn raw_budget(k: usize, absolute_products: f64) -> f64 {
    let steps = (2 * k.div_ceil(1024) + 14) as f64;
    let u = 2.0_f64.powi(-24);
    steps * u / (1.0 - steps * u) * absolute_products + 1e-37
}

/// One BF16 spacing at |value|, with the minimum subnormal spacing at zero.
pub fn bf16_spacing(value: f32) -> f64 {
    let exponent = ((value.to_bits() >> 23) & 255) as i32;
    if exponent == 0 {
        2.0_f64.powi(-133)
    } else {
        2.0_f64.powi(exponent - 127 - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf(values: &[f32]) -> Vec<u16> {
        values.iter().copied().map(round_bf16).collect()
    }

    #[test]
    fn hand_computed_signed_rows() {
        let x = bf(&[1.0, -2.0, 0.5, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let weights = bf(&[
            2.0, 3.0, 4.0, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.5, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let result = run(&x, &weights, 2, 8).unwrap();
        assert_eq!(result.raw, [-4.0, 1.0]);
        assert_eq!(result.bf16, bf(&[-4.0, 1.0]));
        assert_eq!(result.absolute_products, [12.0, 7.0]);
    }

    #[test]
    fn cancellation_and_rne_midpoints() {
        let x = bf(&[1.0; 8]);
        let weights = bf(&[
            256.0,
            -256.0,
            1.0,
            1.0 / 256.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            3.0 / 256.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        let result = run(&x, &weights, 2, 8).unwrap();
        assert_eq!(result.raw, [1.0 + 1.0 / 256.0, 1.0 + 3.0 / 256.0]);
        assert_eq!(result.bf16, bf(&[1.0, 1.015625]));
    }

    #[test]
    fn validates_inputs_and_error_envelope() {
        assert!(run(&[0; 8], &[0; 8], 0, 8).is_err());
        assert!(run(&[0; 7], &[0; 7], 1, 7).is_err());
        assert!(run(&[0; 8], &[0; 7], 1, 8).is_err());
        assert!(run(&[0x7fc0; 8], &[0; 8], 1, 8).is_err());
        assert!(run(&[0x7f7f; 8], &[0x7f7f; 8], 1, 8).is_err());
        assert_eq!(run(&[0; 8], &[0x3f80; 8], 1, 8).unwrap().raw, [0.0]);
        assert!(raw_budget(5120, 1.0) < 2e-6);
        assert_eq!(bf16_spacing(1.0), 1.0 / 128.0);
    }
}
