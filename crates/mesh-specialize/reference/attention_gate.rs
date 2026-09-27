//! Independent scalar sigmoid and product reference for attention output gates.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

const MAX_ELEMENTS: usize = 67_108_864;

#[derive(Debug, PartialEq)]
pub struct Gate {
    /// Stable FP32 sigmoid before its BF16 rounding boundary.
    pub sigmoid: Vec<f32>,
    /// BF16-rounded sigmoid values used by the product.
    pub activated: Vec<u16>,
    /// FP32 product of decoded attention and decoded `activated` values.
    pub unrounded: Vec<f32>,
    /// BF16-rounded product.
    pub output: Vec<u16>,
}

/// Apply a stable sigmoid gate to BF16 attention values.
///
/// The sigmoid is computed in FP64, cast to FP32, then rounded to BF16 before
/// multiplication. Signed zero follows ordinary IEEE multiplication behavior.
pub fn run(attention: &[u16], gate: &[u16]) -> Result<Gate> {
    let count = validate(attention, gate)?;
    let mut result = Gate {
        sigmoid: reserve(count, "attention gate sigmoid")?,
        activated: reserve(count, "attention gate BF16 activation")?,
        unrounded: reserve(count, "attention gate FP32 product")?,
        output: reserve(count, "attention gate BF16 output")?,
    };

    for (&attention_bits, &gate_bits) in attention.iter().zip(gate) {
        let gate_value = bf16_to_f32(gate_bits);
        let sigmoid = sigmoid_f64(gate_value) as f32;
        ensure!(sigmoid.is_finite(), "attention sigmoid overflows FP32");
        let activated_bits = round_bf16(sigmoid);
        let activated = bf16_to_f32(activated_bits);
        ensure!(activated.is_finite(), "attention sigmoid overflows BF16");

        let product = bf16_to_f32(attention_bits) * activated;
        ensure!(
            product.is_finite(),
            "attention gated product overflows FP32"
        );
        let output_bits = round_bf16(product);
        ensure!(
            bf16_to_f32(output_bits).is_finite(),
            "attention gated product overflows BF16"
        );
        result.sigmoid.push(sigmoid);
        result.activated.push(activated_bits);
        result.unrounded.push(product);
        result.output.push(output_bits);
    }
    Ok(result)
}

fn validate(attention: &[u16], gate: &[u16]) -> Result<usize> {
    ensure!(
        (1..=MAX_ELEMENTS).contains(&attention.len()),
        "invalid attention gate element count"
    );
    ensure!(
        gate.len() == attention.len(),
        "attention gate input extent mismatch"
    );
    ensure!(
        attention
            .iter()
            .chain(gate)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention gate inputs must be finite BF16 values"
    );
    Ok(attention.len())
}

fn sigmoid_f64(value: f32) -> f64 {
    let value = f64::from(value);
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn reserve<T>(count: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .with_context(|| format!("cannot reserve {label}"))?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::entry_reference::{bf16_to_f32, round_bf16};

    #[test]
    fn zero_gate_has_half_sigmoid() {
        let result = run(&[round_bf16(2.0)], &[round_bf16(0.0)]).unwrap();
        assert_eq!(result.sigmoid, [0.5]);
        assert_eq!(result.activated, [round_bf16(0.5)]);
        assert_eq!(result.unrounded, [1.0]);
        assert_eq!(result.output, [round_bf16(1.0)]);
    }

    #[test]
    fn extreme_gates_saturate_stably() {
        let result = run(
            &[round_bf16(1.0), round_bf16(1.0)],
            &[round_bf16(1.0e20), round_bf16(-1.0e20)],
        )
        .unwrap();
        assert_eq!(result.sigmoid, [1.0, 0.0]);
        assert_eq!(result.activated, [round_bf16(1.0), round_bf16(0.0)]);
        assert_eq!(result.output, [round_bf16(1.0), round_bf16(0.0)]);
    }

    #[test]
    fn negative_ninety_retains_a_bf16_subnormal_sigmoid() {
        let result = run(&[round_bf16(1.0)], &[round_bf16(-90.0)]).unwrap();
        assert!(result.sigmoid[0] > 0.0);
        assert!(result.sigmoid[0] < f32::MIN_POSITIVE);
        assert_ne!(result.activated[0] & 0x7fff, 0);
        assert!(bf16_to_f32(result.activated[0]).is_finite());
    }

    #[test]
    fn signed_attention_zero_is_preserved_through_a_zero_gate() {
        let result = run(&[0x0000, 0x8000], &[0x8000, 0x0000]).unwrap();
        assert_eq!(result.sigmoid, [0.5, 0.5]);
        assert_eq!(result.activated, [round_bf16(0.5); 2]);
        assert_eq!(
            result
                .unrounded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x0000_0000, 0x8000_0000]
        );
        assert_eq!(result.output, [0x0000, 0x8000]);
    }

    #[test]
    fn sigmoid_midpoint_rounds_to_bf16_before_attention_product() {
        let attention = round_bf16(100.0);
        let gate = round_bf16(0.0078125);
        let result = run(&[attention], &[gate]).unwrap();
        assert_eq!(result.sigmoid, [0.5 + 1.0 / 512.0]);
        assert_eq!(result.activated, [round_bf16(0.5)]);
        assert_eq!(result.unrounded, [50.0]);
        assert_eq!(result.output, [round_bf16(50.0)]);
        assert_ne!(
            round_bf16(bf16_to_f32(attention) * result.sigmoid[0]),
            result.output[0]
        );
    }

    #[test]
    fn rejects_empty_mismatched_and_nonfinite_bf16_inputs() {
        assert!(run(&[], &[]).is_err());
        assert!(run(&[round_bf16(1.0)], &[]).is_err());
        assert!(run(&[0x7f80], &[round_bf16(0.0)]).is_err());
        assert!(run(&[round_bf16(1.0)], &[0x7fc0]).is_err());
    }
}
