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
    run_observed(input, rows, width, weights, &mut |_, _| Ok(()))
}

/// Run the MLP and report each BF16 boundary to a read-only observer.
pub fn run_observed(
    input: &[u16],
    rows: usize,
    width: usize,
    weights: &Weights,
    observer: &mut dyn FnMut(&str, &[u16]) -> Result<()>,
) -> Result<Vec<u16>> {
    match weights {
        Weights::Nvfp4(weights) => run_nvfp4(input, rows, width, weights, observer),
        Weights::Fp8(weights) => {
            let result = fp8_mlp_reference::run(input, rows, width, weights)?;
            observer("mlp_gate", &result.gate)?;
            observer("mlp_up", &result.up)?;
            observer("mlp_activation", &result.activation)?;
            observer("mlp_down", &result.down)?;
            Ok(result.down)
        }
    }
}

fn run_nvfp4(
    input: &[u16],
    rows: usize,
    width: usize,
    weights: &Nvfp4Mlp,
    observer: &mut dyn FnMut(&str, &[u16]) -> Result<()>,
) -> Result<Vec<u16>> {
    validate_nvfp4(input, rows, width, weights)?;
    let gate = project_nvfp4(input, rows, width, &weights.gate)?;
    observer("mlp_gate", &gate.normalized)?;
    let up = project_nvfp4(input, rows, width, &weights.up)?;
    observer("mlp_up", &up.normalized)?;
    let activated = mlp_activation_reference::run(&gate.normalized, &up.normalized)?;
    observer("mlp_activation", &activated.output)?;
    let down = project_nvfp4(
        &activated.output,
        rows,
        weights.gate.channels,
        &weights.down,
    )?;
    observer("mlp_down", &down.normalized)?;
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
    use super::{Weights, run, run_observed};
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
    fn observer_receives_ordered_bf16_boundaries_and_can_stop_the_run() {
        let weights = Weights::Nvfp4(Nvfp4Mlp {
            gate: projection("gate", 16),
            up: projection("up", 16),
            down: projection("down", 16),
        });
        let mut stages = Vec::new();
        let output = run_observed(&[0x3f80; 16], 1, 16, &weights, &mut |stage, values| {
            stages.push((stage.to_owned(), values.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            stages
                .iter()
                .map(|(stage, _)| stage.as_str())
                .collect::<Vec<_>>(),
            ["mlp_gate", "mlp_up", "mlp_activation", "mlp_down"]
        );
        assert!(stages.iter().all(|(_, values)| values == &[0; 16]));
        assert_eq!(output, stages[3].1);

        let error = run_observed(&[0x3f80; 16], 1, 16, &weights, &mut |_, _| {
            Err(anyhow::anyhow!("observer stopped"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("observer stopped"));
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
