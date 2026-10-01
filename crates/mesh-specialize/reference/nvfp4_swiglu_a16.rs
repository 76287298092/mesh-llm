use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    nvfp4_linear_reference::Matrix,
};

const GROUP_WIDTH: usize = 16;
const E2M1_LEVELS: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub struct SwiGluInput<'a> {
    pub activation_bf16: &'a [u16],
    pub gate: Matrix<'a>,
    pub up: Matrix<'a>,
    pub width: usize,
}

#[derive(Debug, PartialEq)]
pub struct SwiGluResult {
    pub gate_dot: Vec<f64>,
    pub gate_absolute_sum: Vec<f64>,
    pub up_dot: Vec<f64>,
    pub up_absolute_sum: Vec<f64>,
    pub activation_raw_fp32: Vec<f32>,
    pub activation_absolute_sum: Vec<f64>,
    pub activation_bf16: Vec<u16>,
}

/// Project gate/up directly from BF16, apply FP64 SiLU/product, then round once to BF16.
pub fn run(input: SwiGluInput<'_>) -> Result<SwiGluResult> {
    let SwiGluInput {
        activation_bf16,
        gate,
        up,
        width,
    } = input;
    ensure!(
        (GROUP_WIDTH..=32768).contains(&width) && width.is_multiple_of(GROUP_WIDTH),
        "NVFP4 A16 SwiGLU width must be a multiple of 16 in 16..=32768"
    );
    ensure!(gate.rows == up.rows, "NVFP4 A16 gate/up row counts differ");
    validate_matrix(&gate, width, "gate")?;
    validate_matrix(&up, width, "up")?;
    ensure!(
        activation_bf16.len() == width
            && activation_bf16
                .iter()
                .all(|&bits| bf16_to_f32(bits).is_finite()),
        "NVFP4 A16 activation extent or BF16 values are invalid"
    );

    let mut result = SwiGluResult {
        gate_dot: Vec::with_capacity(gate.rows),
        gate_absolute_sum: Vec::with_capacity(gate.rows),
        up_dot: Vec::with_capacity(up.rows),
        up_absolute_sum: Vec::with_capacity(up.rows),
        activation_raw_fp32: Vec::with_capacity(gate.rows),
        activation_absolute_sum: Vec::with_capacity(gate.rows),
        activation_bf16: Vec::with_capacity(gate.rows),
    };
    for channel in 0..gate.rows {
        let (gate_dot, gate_absolute_sum) = dot(activation_bf16, &gate, channel, width)?;
        let (up_dot, up_absolute_sum) = dot(activation_bf16, &up, channel, width)?;
        let (silu, silu_derivative) = silu_f64_with_derivative(gate_dot);
        let output_value = (silu * up_dot) as f32;
        let activation_absolute_sum =
            gate_absolute_sum * (silu_derivative * up_dot).abs() + up_absolute_sum * silu.abs();
        ensure!(
            output_value.is_finite() && activation_absolute_sum.is_finite(),
            "NVFP4 A16 SwiGLU result is nonfinite"
        );
        let rounded = round_bf16(output_value);
        ensure!(
            bf16_to_f32(rounded).is_finite(),
            "NVFP4 A16 SwiGLU output overflows BF16"
        );
        result.gate_dot.push(gate_dot);
        result.gate_absolute_sum.push(gate_absolute_sum);
        result.up_dot.push(up_dot);
        result.up_absolute_sum.push(up_absolute_sum);
        result.activation_raw_fp32.push(output_value);
        result.activation_absolute_sum.push(activation_absolute_sum);
        result.activation_bf16.push(rounded);
    }
    Ok(result)
}

fn validate_matrix(matrix: &Matrix<'_>, width: usize, name: &str) -> Result<()> {
    ensure!(
        (1..=32768).contains(&matrix.rows),
        "invalid NVFP4 A16 {name} row count"
    );
    ensure!(
        matrix.global.is_finite() && matrix.global > 0.0,
        "NVFP4 A16 {name} weight divisor must be positive and finite"
    );
    let values = matrix
        .rows
        .checked_mul(width)
        .context("NVFP4 A16 matrix extent overflows usize")?;
    ensure!(
        matrix.packed.len() == values / 2 && matrix.scales.len() == values / GROUP_WIDTH,
        "NVFP4 A16 {name} packed plane extent mismatch"
    );
    ensure!(
        matrix.scales.iter().all(|&code| code <= 126),
        "NVFP4 A16 {name} has a nonfinite E4M3 scale"
    );
    Ok(())
}

fn dot(activation: &[u16], matrix: &Matrix<'_>, row: usize, width: usize) -> Result<(f64, f64)> {
    let groups = width / GROUP_WIDTH;
    let mut sum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for group in 0..groups {
        let scale = f64::from(crate::projection_reference::decode(
            matrix.scales[row * groups + group],
        ));
        let start = group * GROUP_WIDTH;
        for index in 0..GROUP_WIDTH {
            let position = start + index;
            let packed = matrix.packed[row * (width / 2) + position / 2];
            let code = (packed >> ((position % 2) * 4)) & 0x0f;
            let weight = decode_e2m1(code) * scale / f64::from(matrix.global);
            let product = f64::from(bf16_to_f32(activation[position])) * weight;
            sum += product;
            absolute_sum += product.abs();
        }
    }
    ensure!(sum.is_finite(), "NVFP4 A16 dot is nonfinite");
    ensure!(
        absolute_sum.is_finite(),
        "NVFP4 A16 absolute product sum is nonfinite"
    );
    Ok((sum, absolute_sum))
}

fn decode_e2m1(code: u8) -> f64 {
    let magnitude = E2M1_LEVELS[usize::from(code & 0x07)];
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn silu_f64_with_derivative(value: f64) -> (f64, f64) {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    (value * sigmoid, sigmoid + value * sigmoid * (1.0 - sigmoid))
}

#[cfg(test)]
mod tests {
    use super::{SwiGluInput, run};
    use crate::{entry_reference::round_bf16, nvfp4_linear_reference::Matrix};

    fn matrix<'a>(packed: &'a [u8], scales: &'a [u8], global: f32) -> Matrix<'a> {
        Matrix {
            packed,
            scales,
            rows: 1,
            global,
        }
    }

    #[test]
    fn independent_a16_reference_preserves_gate_up_order_and_group_scales() {
        let activation = [round_bf16(1.0); 32];
        let gate = matrix(&[0x22; 16], &[0x38, 0x40], 2.0);
        let up = matrix(&[0x33; 16], &[0x30, 0x38], 4.0);

        let output = run(SwiGluInput {
            activation_bf16: &activation,
            gate,
            up,
            width: 32,
        })
        .unwrap();

        let expected = (24.0_f64 / (1.0 + (-24.0_f64).exp()) * 9.0) as f32;
        assert_eq!(output.gate_dot, [24.0]);
        assert_eq!(output.gate_absolute_sum, [24.0]);
        assert_eq!(output.up_dot, [9.0]);
        assert_eq!(output.up_absolute_sum, [9.0]);
        assert!((output.activation_absolute_sum[0] - 432.0).abs() < 1.0e-5);
        assert_eq!(output.activation_bf16, [round_bf16(expected)]);
    }

    #[test]
    fn direct_a16_reference_handles_negative_zero_extremes_and_group_tails() {
        let mut activation = [round_bf16(-0.0); 80];
        activation[64..].fill(round_bf16(1.0e20));
        let gate = matrix(&[0x88; 40], &[0x38; 5], 1.0);
        let up = matrix(&[0x77; 40], &[0x38; 5], 1.0);

        let output = run(SwiGluInput {
            activation_bf16: &activation,
            gate,
            up,
            width: 80,
        })
        .unwrap();

        assert_eq!(output.gate_dot, [0.0]);
        assert_eq!(
            output.up_dot,
            [f64::from(crate::entry_reference::bf16_to_f32(round_bf16(1.0e20))) * 96.0]
        );
        assert_eq!(output.activation_bf16, [0x0000]);
    }

    #[test]
    fn rejects_invalid_extents_and_nonfinite_inputs_or_scales() {
        let activation = [round_bf16(1.0); 16];
        let packed = [0x22; 8];
        let scales = [0x38];
        let valid = matrix(&packed, &scales, 1.0);
        assert!(
            run(SwiGluInput {
                activation_bf16: &activation[..15],
                gate: valid,
                up: valid,
                width: 16,
            })
            .is_err()
        );
        assert!(
            run(SwiGluInput {
                activation_bf16: &[0x7f80; 16],
                gate: valid,
                up: valid,
                width: 16,
            })
            .is_err()
        );
        assert!(
            run(SwiGluInput {
                activation_bf16: &activation,
                gate: matrix(&packed, &[0x7f], 1.0),
                up: valid,
                width: 16,
            })
            .is_err()
        );
    }
}
