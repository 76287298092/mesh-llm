//! Independent CPU composition for the decoder MLP's final BF16 output.

use anyhow::{Context as _, Result, anyhow, ensure};

use crate::{
    entry_reference::bf16_to_f32,
    fp8_mlp_reference,
    kernels::{Nvfp4Mlp, Nvfp4Projection},
    mlp_activation_reference, nvfp4_linear_reference, nvfp4_quantize_reference,
};

const MIN_NVFP4_WIDTH: usize = 16;
const MAX_NVFP4_WIDTH: usize = 32_768;
const MAX_ROWS: usize = 2_048;

pub enum Weights {
    Nvfp4(Nvfp4Mlp),
    Fp8(fp8_mlp_reference::Weights),
}

/// Run the MLP from BF16 hidden rows through the final down-projection BF16 output.
pub fn run(input: &[u16], rows: usize, width: usize, weights: &Weights) -> Result<Vec<u16>> {
    match weights {
        Weights::Nvfp4(weights) => run_nvfp4(input, rows, width, weights),
        Weights::Fp8(weights) => Ok(fp8_mlp_reference::run(input, rows, width, weights)?.down),
    }
}

fn run_nvfp4(input: &[u16], rows: usize, width: usize, weights: &Nvfp4Mlp) -> Result<Vec<u16>> {
    validate_nvfp4(input, rows, width, weights)?;
    let gate = project_nvfp4(input, rows, width, &weights.gate)?;
    let up = project_nvfp4(input, rows, width, &weights.up)?;
    let activated = mlp_activation_reference::run(&gate.normalized, &up.normalized)?;
    let down = project_nvfp4(
        &activated.output,
        rows,
        weights.gate.channels,
        &weights.down,
    )?;
    Ok(down.normalized)
}

fn validate_nvfp4(input: &[u16], rows: usize, width: usize, weights: &Nvfp4Mlp) -> Result<()> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "invalid decoder MLP row count"
    );
    ensure!(
        (MIN_NVFP4_WIDTH..=MAX_NVFP4_WIDTH).contains(&width) && width.is_multiple_of(16),
        "NVFP4 decoder MLP hidden width must be a multiple of 16 in {MIN_NVFP4_WIDTH}..={MAX_NVFP4_WIDTH}"
    );
    let input_values = checked_product(rows, width, "decoder MLP input")?;
    ensure!(
        input.len() == input_values,
        "decoder MLP input extent mismatch"
    );
    ensure!(
        input.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "decoder MLP input must contain finite BF16 values"
    );
    ensure!(
        (MIN_NVFP4_WIDTH..=MAX_NVFP4_WIDTH).contains(&weights.gate.channels)
            && weights.gate.channels.is_multiple_of(16)
            && weights.gate.channels == weights.up.channels,
        "NVFP4 decoder MLP gate/up channels must match and be a multiple of 16 in {MIN_NVFP4_WIDTH}..={MAX_NVFP4_WIDTH}"
    );
    ensure!(
        weights.down.channels == width,
        "NVFP4 decoder MLP down channels must equal the hidden width"
    );
    validate_projection(&weights.gate, width, "gate")?;
    validate_projection(&weights.up, width, "up")?;
    validate_projection(&weights.down, weights.gate.channels, "down")
}

fn validate_projection(projection: &Nvfp4Projection, width: usize, name: &str) -> Result<()> {
    let values = checked_product(projection.channels, width, name)?;
    ensure!(
        projection.packed.len() == values / 2,
        "NVFP4 decoder MLP {name} packed extent mismatch"
    );
    ensure!(
        projection.scales.len() == values / 16,
        "NVFP4 decoder MLP {name} scale extent mismatch"
    );
    ensure!(
        projection.scales.iter().all(|&code| code <= 126),
        "NVFP4 decoder MLP {name} scales must be finite E4M3FN codes"
    );
    ensure!(
        projection.input_global.is_finite()
            && projection.input_global > 0.0
            && projection.weight_global.is_finite()
            && projection.weight_global > 0.0,
        "NVFP4 decoder MLP {name} global scales must be finite and positive"
    );
    Ok(())
}

fn project_nvfp4(
    input: &[u16],
    rows: usize,
    width: usize,
    projection: &Nvfp4Projection,
) -> Result<crate::projection_reference::LinearReference> {
    let quantized = nvfp4_quantize_reference::run(input, rows, width, projection.input_global)?;
    nvfp4_linear_reference::run(
        nvfp4_linear_reference::Matrix {
            packed: &quantized.packed,
            scales: &quantized.scales,
            rows,
            global: projection.input_global,
        },
        nvfp4_linear_reference::Matrix {
            packed: &projection.packed,
            scales: &projection.scales,
            rows: projection.channels,
            global: projection.weight_global,
        },
        width,
    )
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
        .context("validate decoder MLP shape")
}

#[cfg(test)]
mod tests {
    use super::{Weights, run};
    use crate::kernels::{Nvfp4Mlp, Nvfp4Projection};

    fn projection(name: &str, channels: usize) -> Nvfp4Projection {
        Nvfp4Projection {
            name: name.to_owned(),
            packed: vec![0; channels * 16 / 2],
            scales: vec![0x38; channels],
            input_global: 1.0,
            weight_global: 1.0,
            channels,
        }
    }

    #[test]
    fn zero_nvfp4_weights_and_unit_scales_produce_zero_hidden_output() {
        let weights = Weights::Nvfp4(Nvfp4Mlp {
            gate: projection("gate", 16),
            up: projection("up", 16),
            down: projection("down", 16),
        });
        let output = run(&[0x3f80; 16], 1, 16, &weights).unwrap();
        assert_eq!(output, [0; 16]);
    }

    #[test]
    fn rejects_mismatched_gate_and_up_widths() {
        let weights = Weights::Nvfp4(Nvfp4Mlp {
            gate: projection("gate", 16),
            up: projection("up", 32),
            down: projection("down", 16),
        });
        assert!(run(&[0x3f80; 16], 1, 16, &weights).is_err());
    }
}
