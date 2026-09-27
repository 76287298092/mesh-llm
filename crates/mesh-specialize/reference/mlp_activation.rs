//! Independent BF16 SiLU and product reference for MLP activations.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

const MAX_ELEMENTS: usize = 67_108_864;

#[derive(Debug, PartialEq)]
pub struct Activation {
    pub output: Vec<u16>,
    pub silu: Vec<f32>,
    pub activated: Vec<u16>,
    pub unrounded: Vec<f32>,
}

/// Apply stable SiLU to BF16 gate values, round, then multiply by BF16 up values.
pub fn run(gate: &[u16], up: &[u16]) -> Result<Activation> {
    let count = validate(gate, up)?;
    let mut result = Activation {
        output: Vec::new(),
        silu: Vec::new(),
        activated: Vec::new(),
        unrounded: Vec::new(),
    };
    result
        .output
        .try_reserve_exact(count)
        .context("cannot reserve MLP activation BF16 outputs")?;
    result
        .silu
        .try_reserve_exact(count)
        .context("cannot reserve MLP SiLU outputs")?;
    result
        .activated
        .try_reserve_exact(count)
        .context("cannot reserve MLP activated BF16 values")?;
    result
        .unrounded
        .try_reserve_exact(count)
        .context("cannot reserve MLP unrounded outputs")?;

    for (&gate_bits, &up_bits) in gate.iter().zip(up) {
        let activated_value = silu(bf16_to_f32(gate_bits))?;
        let activated_bits = round_bf16(activated_value);
        let activated_bf16 = bf16_to_f32(activated_bits);
        ensure!(
            activated_bf16.is_finite(),
            "MLP SiLU activation overflows BF16"
        );
        let product = activated_bf16 * bf16_to_f32(up_bits);
        ensure!(product.is_finite(), "MLP activation product overflows FP32");
        let output_bits = round_bf16(product);
        ensure!(
            bf16_to_f32(output_bits).is_finite(),
            "MLP activation product overflows BF16"
        );
        result.output.push(output_bits);
        result.silu.push(activated_value);
        result.activated.push(activated_bits);
        result.unrounded.push(product);
    }
    Ok(result)
}

fn validate(gate: &[u16], up: &[u16]) -> Result<usize> {
    ensure!(
        (1..=MAX_ELEMENTS).contains(&gate.len()),
        "invalid MLP activation element count"
    );
    ensure!(
        up.len() == gate.len(),
        "MLP activation input extent mismatch"
    );
    ensure!(
        gate.iter()
            .chain(up)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "MLP activation inputs must be finite BF16 values"
    );
    Ok(gate.len())
}

fn silu(value: f32) -> Result<f32> {
    ensure!(value.is_finite(), "SiLU input must be finite");
    let wide = f64::from(value);
    let activated = if value >= 0.0 {
        wide / (1.0 + (-wide).exp())
    } else {
        let exponential = wide.exp();
        wide * exponential / (1.0 + exponential)
    };
    let result = activated as f32;
    ensure!(result.is_finite(), "SiLU output overflows FP32");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    #[test]
    fn zero_signed_zero_and_signed_up_values_keep_ieee_signs() {
        let gate = [bf16(0.0), bf16(-0.0), bf16(0.0), bf16(-0.0)];
        let up = [bf16(2.0), bf16(2.0), bf16(-2.0), bf16(-2.0)];
        let result = run(&gate, &up).unwrap();
        assert_eq!(
            result
                .silu
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0, 0x8000_0000, 0, 0x8000_0000]
        );
        assert_eq!(result.activated, [0x0000, 0x8000, 0x0000, 0x8000]);
        assert_eq!(
            result
                .unrounded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0, 0x8000_0000, 0x8000_0000, 0]
        );
        assert_eq!(result.output, [0x0000, 0x8000, 0x8000, 0x0000]);
    }

    #[test]
    fn positive_and_negative_gates_round_silu_before_signed_up_multiply() {
        let result = run(&[bf16(1.0), bf16(-1.0)], &[bf16(-1.0), bf16(-2.0)]).unwrap();
        let positive_silu = 1.0_f64 / (1.0 + (-1.0_f64).exp());
        let negative_silu = -(-1.0_f64).exp() / (1.0 + (-1.0_f64).exp());
        assert_eq!(result.silu, [positive_silu as f32, negative_silu as f32]);
        assert_eq!(result.activated, [bf16(0.73046875), bf16(-0.26953125)]);
        assert_eq!(result.unrounded, [-0.73046875, 0.5390625]);
        assert_eq!(result.output, [bf16(-0.73046875), bf16(0.5390625)]);
    }

    #[test]
    fn boundary_and_extreme_gates_keep_stable_fp64_silu_results() {
        let gate = [bf16(1.0e20), bf16(-1.0e20), bf16(-90.0), bf16(1.0)];
        let up = [bf16(1.0); 4];
        let result = run(&gate, &up).unwrap();
        assert_eq!(result.silu[0].to_bits(), bf16_to_f32(gate[0]).to_bits());
        assert_eq!(result.activated[0], gate[0]);
        assert_eq!(result.silu[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(result.activated[1], 0x8000);
        let negative = f64::from(bf16_to_f32(gate[2]));
        let exponential = negative.exp();
        let expected = (negative * exponential / (1.0 + exponential)) as f32;
        assert_eq!(result.silu[2], expected);
        assert_ne!(result.activated[2] & 0x7fff, 0);
        assert_eq!(result.activated[3], bf16(0.73046875));
        assert_eq!(result.output[3], bf16(0.73046875));
    }

    #[test]
    fn rejects_bad_extents_nonfinite_bf16_and_fp32_or_bf16_overflow() {
        assert!(run(&[], &[]).is_err());
        assert!(run(&[bf16(1.0)], &[]).is_err());
        assert!(run(&[0x7f80], &[bf16(1.0)]).is_err());
        assert!(run(&[bf16(1.0)], &[0x7fc0]).is_err());
        let large = bf16(1.0e20);
        assert!(run(&[large], &[large]).is_err());
        let largest_finite = [0x7f7f];
        assert!(run(&largest_finite, &largest_finite).is_err());
    }
}
