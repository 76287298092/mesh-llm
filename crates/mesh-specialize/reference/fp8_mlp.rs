//! Independent scalar composition of the checkpoint's FP8 MLP projections.

use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::bf16_to_f32,
    mlp_activation_reference,
    projection_reference::{self, LinearReference, QuantizedRows},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub weights: Vec<u8>,
    pub scales: Vec<u16>,
    pub channels: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub gate: Projection,
    pub up: Projection,
    pub down: Projection,
}

#[derive(Debug, PartialEq)]
pub struct Output {
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub activation: Vec<u16>,
    pub down: Vec<u16>,
}

pub fn run(input: &[u16], rows: usize, width: usize, weights: &Weights) -> Result<Output> {
    validate(input, rows, width, weights)?;

    let input_rows = projection_reference::quantize(input, rows, width)?;
    let gate = project(&input_rows, &weights.gate, width)?;
    let up = project(&input_rows, &weights.up, width)?;
    let activation = mlp_activation_reference::run(&gate, &up)?.output;
    let activated_rows = projection_reference::quantize(&activation, rows, weights.gate.channels)?;
    let down = project(&activated_rows, &weights.down, weights.gate.channels)?;
    Ok(Output {
        gate,
        up,
        activation,
        down,
    })
}

fn validate(input: &[u16], rows: usize, width: usize, weights: &Weights) -> Result<()> {
    ensure!((1..=2048).contains(&rows), "invalid FP8 MLP row count");
    ensure!((1..=32768).contains(&width), "invalid FP8 MLP input width");
    let input_len = rows
        .checked_mul(width)
        .context("FP8 MLP input extent overflows usize")?;
    ensure!(input.len() == input_len, "FP8 MLP input extent mismatch");
    ensure!(
        input.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "FP8 MLP input must contain finite BF16 values"
    );
    ensure!(
        (1..=32768).contains(&weights.gate.channels)
            && weights.gate.channels == weights.up.channels,
        "FP8 MLP gate/up channel counts must match in 1..=32768"
    );
    ensure!(
        weights.down.channels == width,
        "FP8 MLP down channels must equal the hidden width"
    );
    validate_projection(&weights.gate, width, "gate")?;
    validate_projection(&weights.up, width, "up")?;
    validate_projection(&weights.down, weights.gate.channels, "down")?;
    Ok(())
}

fn validate_projection(projection: &Projection, input_width: usize, name: &str) -> Result<()> {
    ensure!(
        (1..=32768).contains(&projection.channels),
        "invalid FP8 MLP {name} channel count"
    );
    let elements = projection
        .channels
        .checked_mul(input_width)
        .with_context(|| format!("FP8 MLP {name} extent overflows usize"))?;
    ensure!(
        projection.weights.len() == elements,
        "FP8 MLP {name} weight extent mismatch"
    );
    ensure!(
        projection.scales.len() == projection.channels,
        "FP8 MLP {name} scale extent mismatch"
    );
    Ok(())
}

fn project(input: &QuantizedRows, weights: &Projection, width: usize) -> Result<Vec<u16>> {
    let result = projection_reference::linear(input, &weights.weights, &weights.scales, width)?;
    ensure_finite_output(&result)?;
    Ok(result.normalized)
}

fn ensure_finite_output(result: &LinearReference) -> Result<()> {
    ensure!(
        result.unrounded.iter().all(|value| value.is_finite())
            && result
                .normalized
                .iter()
                .all(|&bits| bf16_to_f32(bits).is_finite()),
        "FP8 MLP projection output overflows FP32 or BF16"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Output, Projection, Weights, run};

    fn projection(weight: u8, channels: usize, scale: u16) -> Projection {
        Projection {
            weights: vec![weight; channels],
            scales: vec![scale; channels],
            channels,
        }
    }

    fn scalar_weights(gate_weight: u8) -> Weights {
        Weights {
            gate: projection(gate_weight, 1, 0x3f80),
            up: projection(0x38, 1, 0x3f80),
            down: projection(0x38, 1, 0x3f80),
        }
    }

    #[test]
    fn zero_gate_produces_zero_activation_and_down_output() {
        let output = run(&[0x3f80], 1, 1, &scalar_weights(0)).unwrap();
        assert_eq!(
            output,
            Output {
                gate: vec![0],
                up: vec![0x3f80],
                activation: vec![0],
                down: vec![0],
            }
        );
    }

    #[test]
    fn unit_input_and_unit_weights_follow_explicit_fp8_quantization() {
        let output = run(&[0x3f80], 1, 1, &scalar_weights(0x38)).unwrap();
        // Input 1 uses scale 1/448 and E4M3 value 448; each unit matrix
        // therefore returns 1. SiLU(1) rounds to BF16 0x3f3b.
        assert_eq!(
            output,
            Output {
                gate: vec![0x3f80],
                up: vec![0x3f80],
                activation: vec![0x3f3b],
                down: vec![0x3f3b],
            }
        );
    }

    #[test]
    fn rejects_invalid_shapes_extents_and_nonfinite_values() {
        let weights = scalar_weights(0x38);
        assert!(run(&[0x3f80], 0, 1, &weights).is_err());
        assert!(run(&[0x3f80], 1, 0, &weights).is_err());
        assert!(run(&[], 1, 1, &weights).is_err());
        assert!(run(&[0x7f80], 1, 1, &weights).is_err());

        let mut bad_gate = weights.clone();
        bad_gate.gate.weights.clear();
        assert!(run(&[0x3f80], 1, 1, &bad_gate).is_err());
        let mut bad_scales = weights.clone();
        bad_scales.up.scales.clear();
        assert!(run(&[0x3f80], 1, 1, &bad_scales).is_err());
        let mut bad_channels = weights.clone();
        bad_channels.up.channels = 2;
        assert!(run(&[0x3f80], 1, 1, &bad_channels).is_err());
        let mut bad_down = weights;
        bad_down.down.channels = 2;
        assert!(run(&[0x3f80], 1, 1, &bad_down).is_err());

        let mut nonfinite_weight = scalar_weights(0x38);
        nonfinite_weight.gate.weights[0] = 0x7f;
        assert!(run(&[0x3f80], 1, 1, &nonfinite_weight).is_err());
        let mut invalid_scale = scalar_weights(0x38);
        invalid_scale.down.scales[0] = 0;
        assert!(run(&[0x3f80], 1, 1, &invalid_scale).is_err());
    }
}
