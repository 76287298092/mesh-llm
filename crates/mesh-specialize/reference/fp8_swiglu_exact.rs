//! Independent logical composition of exact FP8 gate/up projections and SwiGLU.

use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::bf16_to_f32,
    fp8_mlp_reference::Projection,
    mlp_activation_reference,
    projection_reference::{self, QuantizedRows},
};

const MAX_ROWS: usize = 2048;
const MAX_CHANNELS: usize = 262_144;
const MAX_WIDTH: usize = 32_768;

#[derive(Debug, PartialEq)]
pub struct Output {
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub activated: Vec<u16>,
    pub output: Vec<u16>,
    pub unrounded: Vec<f32>,
}

/// Run both logical projections from the same already-quantized activation rows.
///
/// Each projection uses the existing independent FP64 logical-dot reference;
/// the SwiGLU epilogue uses the independent BF16 activation reference. The
/// function models shared host quantization by accepting one `QuantizedRows`.
pub fn run(
    input: &QuantizedRows,
    gate: &Projection,
    up: &Projection,
    input_width: usize,
) -> Result<Output> {
    validate(input, gate, up, input_width)?;
    let gate_output = project(input, gate, input_width, "gate")?;
    let up_output = project(input, up, input_width, "up")?;
    let activation = mlp_activation_reference::run(&gate_output, &up_output)?;
    Ok(Output {
        gate: gate_output,
        up: up_output,
        activated: activation.activated,
        output: activation.output,
        unrounded: activation.unrounded,
    })
}

fn validate(
    input: &QuantizedRows,
    gate: &Projection,
    up: &Projection,
    input_width: usize,
) -> Result<()> {
    let rows = input.scales.len();
    ensure!((1..=MAX_ROWS).contains(&rows), "invalid SwiGLU row count");
    ensure!(
        (1..=MAX_WIDTH).contains(&input_width),
        "invalid SwiGLU input width"
    );
    let input_elements = rows
        .checked_mul(input_width)
        .context("SwiGLU input extent overflows usize")?;
    ensure!(
        input.codes.len() == input_elements,
        "SwiGLU input extent mismatch"
    );
    ensure!(
        (1..=MAX_CHANNELS).contains(&gate.channels) && gate.channels == up.channels,
        "SwiGLU gate/up channel counts must match"
    );
    validate_projection(gate, input_width, "gate")?;
    validate_projection(up, input_width, "up")?;
    Ok(())
}

fn validate_projection(projection: &Projection, input_width: usize, name: &str) -> Result<()> {
    let elements = projection
        .channels
        .checked_mul(input_width)
        .with_context(|| format!("SwiGLU {name} extent overflows usize"))?;
    ensure!(
        projection.weights.len() == elements,
        "SwiGLU {name} weight extent mismatch"
    );
    ensure!(
        projection.scales.len() == projection.channels,
        "SwiGLU {name} scale extent mismatch"
    );
    Ok(())
}

fn project(
    input: &QuantizedRows,
    projection: &Projection,
    input_width: usize,
    name: &str,
) -> Result<Vec<u16>> {
    let result =
        projection_reference::linear(input, &projection.weights, &projection.scales, input_width)
            .with_context(|| format!("SwiGLU {name} projection failed"))?;
    ensure!(
        result.unrounded.iter().all(|value| value.is_finite())
            && result
                .normalized
                .iter()
                .all(|&bits| bf16_to_f32(bits).is_finite()),
        "SwiGLU {name} projection overflows FP32 or BF16"
    );
    Ok(result.normalized)
}

#[cfg(test)]
mod tests {
    use super::{Output, Projection, run};
    use crate::{
        entry_reference::round_bf16,
        fp8_mlp_reference::Projection as MlpProjection,
        mlp_activation_reference,
        projection_reference::{self, QuantizedRows},
    };

    const M: usize = 3;
    const N: usize = 5;
    const K: usize = 7;

    fn projection(weights: &[u8], scales: &[f32]) -> Projection {
        Projection {
            weights: weights.to_vec(),
            scales: scales.iter().copied().map(round_bf16).collect(),
            channels: N,
        }
    }

    #[test]
    fn odd_signed_canceling_case_matches_separate_logical_projections_and_activation() {
        let input = QuantizedRows {
            codes: vec![
                0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x00, // signed cancellation row
                0xb8, 0x40, 0x38, 0xc0, 0xb0, 0x30, 0x38, 0x30, 0xb0, 0xb8, 0x38, 0x40, 0xc0, 0x00,
            ],
            scales: vec![0.125, 0.5, 1.75],
        };
        let gate = projection(
            &[
                0x38, 0x38, 0xb8, 0xb8, 0x38, 0x38, 0x00, // exact zero dot on row 0
                0xb8, 0x40, 0x38, 0xb0, 0x30, 0x38, 0x00, 0x40, 0xc0, 0xb8, 0x38, 0x30, 0xb0, 0x38,
                0x30, 0x38, 0xb0, 0xc0, 0xb8, 0x40, 0x00, 0xb0, 0x40, 0x38, 0x30, 0xc0, 0xb8, 0x00,
            ],
            &[0.5, 1.0, 1.5, 2.0, 0.75],
        );
        let up = projection(
            &[
                0xb8, 0xb8, 0x38, 0x38, 0xb8, 0xb8, 0x00, 0x38, 0xb0, 0x40, 0x30, 0xc0, 0xb8, 0x00,
                0xb0, 0x38, 0xc0, 0x40, 0x30, 0xb8, 0x00, 0xc0, 0x30, 0x38, 0xb8, 0x40, 0xb0, 0x00,
                0x40, 0xb8, 0xb0, 0x38, 0xc0, 0x30, 0x00,
            ],
            &[3.0, 0.25, 1.25, 0.5, 2.5],
        );

        let gate_projection =
            projection_reference::linear(&input, &gate.weights, &gate.scales, K).unwrap();
        let up_projection =
            projection_reference::linear(&input, &up.weights, &up.scales, K).unwrap();
        let activation =
            mlp_activation_reference::run(&gate_projection.normalized, &up_projection.normalized)
                .unwrap();
        let expected = Output {
            gate: gate_projection.normalized,
            up: up_projection.normalized,
            activated: activation.activated,
            output: activation.output,
            unrounded: activation.unrounded,
        };

        let actual = run(&input, &gate, &up, K).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.gate.len(), M * N);
        assert_eq!(actual.output.len(), M * N);
        assert_eq!(actual.unrounded.len(), M * N);
        assert_eq!(actual.gate[0], 0, "signed gate products should cancel");
        assert_ne!(gate.scales, up.scales, "projection scales are independent");
        assert_ne!(actual.gate, actual.up, "signed projections should differ");
    }

    #[test]
    fn rejects_dimension_mismatch_and_invalid_weight_extents() {
        let input = QuantizedRows {
            codes: vec![0x38; M * K],
            scales: vec![1.0; M],
        };
        let valid = MlpProjection {
            weights: vec![0x38; N * K],
            scales: vec![0x3f80; N],
            channels: N,
        };
        assert!(run(&input, &valid, &valid, 0).is_err());
        let mut mismatched_channels = valid.clone();
        mismatched_channels.channels -= 1;
        assert!(run(&input, &valid, &mismatched_channels, K).is_err());
        let mut short_weights = valid.clone();
        short_weights.weights.pop();
        assert!(run(&input, &short_weights, &valid, K).is_err());
    }
}
