//! Independent high-precision gate oracle accepting FP32 parameter arrays directly.

use crate::{entry_reference::bf16_to_f32, gdn_prepare_reference::prepare_gate};
use anyhow::{Result, ensure};

#[derive(Debug)]
pub struct Gates {
    pub beta: Vec<u16>,
    pub g: Vec<f32>,
    pub decay: Vec<f32>,
}

/// A/B and beta remain BF16. Parameters are never rounded to BF16.
///
/// Uses the existing independent FP64 exp/log1p oracle, not the device's
/// approximate PTX exponentials or polynomial. This is a comparison oracle,
/// not a claim of bit-exact agreement with device transcendental instructions.
pub fn run(
    a: &[u16],
    b: &[u16],
    a_log: &[f32],
    dt_bias: &[f32],
    rows: usize,
    heads: usize,
) -> Result<Gates> {
    ensure!((1..=2048).contains(&rows), "invalid gate rows");
    ensure!((1..=256).contains(&heads), "invalid gate heads");
    ensure!(
        a.len() == rows * heads && b.len() == a.len(),
        "gate input extent mismatch"
    );
    ensure!(
        a_log.len() == heads && dt_bias.len() == heads,
        "gate parameter extent mismatch"
    );
    ensure!(
        a.iter().chain(b).all(|&v| bf16_to_f32(v).is_finite()),
        "nonfinite gate input"
    );
    ensure!(
        a_log.iter().all(|v| (-80.0..=80.0).contains(v)),
        "A_log outside [-80, 80]"
    );
    ensure!(dt_bias.iter().all(|v| v.is_finite()), "nonfinite dt_bias");
    let mut output = Gates {
        beta: Vec::with_capacity(a.len()),
        g: Vec::with_capacity(a.len()),
        decay: Vec::with_capacity(a.len()),
    };
    for index in 0..a.len() {
        let head = index % heads;
        let (beta, g, decay) = prepare_gate(
            bf16_to_f32(a[index]),
            bf16_to_f32(b[index]),
            a_log[head],
            dt_bias[head],
        )?;
        output.beta.push(beta);
        output.g.push(g);
        output.decay.push(decay);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry_reference::round_bf16;

    #[test]
    fn retains_non_bf16_parameter_bits_independently() {
        let logs = [1.0001_f32];
        let biases = [0.5001_f32];
        let rounded_logs = [bf16_to_f32(round_bf16(logs[0]))];
        let rounded_biases = [bf16_to_f32(round_bf16(biases[0]))];
        assert_ne!(logs[0].to_bits(), rounded_logs[0].to_bits());
        assert_ne!(biases[0].to_bits(), rounded_biases[0].to_bits());
        let exact = run(&[0, 0x3f80], &[0, 0xbf80], &logs, &biases, 2, 1).unwrap();
        for (l, d) in [(&rounded_logs, &biases), (&logs, &rounded_biases)] {
            let narrowed = run(&[0, 0x3f80], &[0, 0xbf80], l, d, 2, 1).unwrap();
            assert_ne!(exact.g[0].to_bits(), narrowed.g[0].to_bits());
            assert_ne!(exact.decay[0].to_bits(), narrowed.decay[0].to_bits());
            assert_eq!(exact.beta, narrowed.beta);
        }
        assert_eq!(exact.beta[0], 0x3f00);
        assert_eq!(logs[0].to_bits(), 1.0001_f32.to_bits());
        assert_eq!(biases[0].to_bits(), 0.5001_f32.to_bits());
    }

    #[test]
    fn linear_softplus_keeps_bias_low_bits() {
        let bias = 21.0001_f32;
        let result = run(&[0], &[0], &[0.0], &[bias], 1, 1).unwrap();
        assert_eq!(result.g[0].to_bits(), (-bias).to_bits());
        assert_ne!(
            result.g[0].to_bits(),
            (-bf16_to_f32(round_bf16(bias))).to_bits()
        );
    }

    #[test]
    fn bf16_representable_parameters_match_existing_oracle() {
        let shape = crate::gdn_prepare_reference::Shape {
            rows: 2,
            key_heads: 1,
            value_heads: 2,
            width: 1,
        };
        let a = [0, 0x3f80, 0xbf80, 0x4000];
        let b = [0, 0xbf80, 0x3f80, 0x4000];
        let logs = [0, 0x3f80];
        let biases = [0x3f00, 0xbf00];
        let old =
            crate::gdn_prepare_reference::run(&[0; 8], &a, &b, &logs, &biases, &shape).unwrap();
        let new = run(
            &a,
            &b,
            &logs.map(bf16_to_f32),
            &biases.map(bf16_to_f32),
            2,
            2,
        )
        .unwrap();
        assert_eq!(new.beta, old.beta);
        assert_eq!(new.g, old.g);
        assert_eq!(new.decay, old.decay);
        assert!(run(&a, &b, &[f32::NAN; 2], &[0.0; 2], 2, 2).is_err());
        assert!(run(&a, &b, &[0.0; 2], &[0.0], 2, 2).is_err());
    }
}
