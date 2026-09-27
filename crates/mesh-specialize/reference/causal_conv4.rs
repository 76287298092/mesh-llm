//! Independent logical reference for the fixed width-four causal depthwise convolution.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

pub struct ConvReference {
    pub convolution: Vec<f32>,
    pub activated: Vec<f32>,
    pub output: Vec<u16>,
    pub next_history: Vec<u16>,
    pub absolute_sums: Vec<f64>,
}

/// Stable SiLU for finite FP32 input, evaluated with FP64 exponent arithmetic.
pub fn silu(value: f32) -> f32 {
    let value64 = f64::from(value);
    let exponential = (-value64.abs()).exp();
    let sigmoid = if value64 >= 0.0 {
        1.0 / (1.0 + exponential)
    } else {
        exponential / (1.0 + exponential)
    };
    (value64 * sigmoid) as f32
}

/// Run logical causal depthwise convolution and SiLU for a fixed width-four filter.
///
/// `input` is time-major `[rows, channels]`, `weights` is channel-major
/// `[channels, 4]`, and `history` is time-major `[3, channels]` from oldest to newest.
/// The convolution accumulates independently in FP64, rounds once to BF16 before
/// SiLU, and retains the final three raw BF16 input rows as the next state.
pub fn run(
    input: &[u16],
    weights: &[u16],
    history: &[u16],
    rows: usize,
    channels: usize,
) -> Result<ConvReference> {
    ensure!(
        (1..=2048).contains(&rows),
        "invalid causal convolution row count"
    );
    ensure!(
        (1..=32768).contains(&channels),
        "invalid causal convolution channel count"
    );
    let input_len = rows
        .checked_mul(channels)
        .context("causal convolution input length overflows usize")?;
    let weight_len = channels
        .checked_mul(4)
        .context("causal convolution weight length overflows usize")?;
    let history_len = channels
        .checked_mul(3)
        .context("causal convolution history length overflows usize")?;
    ensure!(
        input.len() == input_len,
        "causal convolution input extent mismatch"
    );
    ensure!(
        weights.len() == weight_len,
        "causal convolution weight extent mismatch"
    );
    ensure!(
        history.len() == history_len,
        "causal convolution history extent mismatch"
    );
    ensure!(
        input
            .iter()
            .chain(weights)
            .chain(history)
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "causal convolution inputs must be finite BF16 values"
    );

    let mut convolution = Vec::with_capacity(input_len);
    let mut activated = Vec::with_capacity(input_len);
    let mut output = Vec::with_capacity(input_len);
    let mut absolute_sums = Vec::with_capacity(input_len);
    for time in 0..rows {
        for channel in 0..channels {
            let (sum, absolute_sum) =
                convolve_element(input, weights, history, time, channel, channels);
            ensure!(
                sum.is_finite() && absolute_sum.is_finite(),
                "causal convolution FP64 accumulation overflow"
            );
            let convolution_value = sum as f32;
            ensure!(
                convolution_value.is_finite(),
                "causal convolution FP32 result overflow"
            );
            let rounded_convolution = bf16_to_f32(round_bf16(convolution_value));
            ensure!(
                rounded_convolution.is_finite(),
                "causal convolution BF16 result overflow"
            );
            let activated_value = silu(rounded_convolution);
            ensure!(activated_value.is_finite(), "causal SiLU result overflow");
            let output_bits = round_bf16(activated_value);
            ensure!(
                bf16_to_f32(output_bits).is_finite(),
                "causal SiLU BF16 result overflow"
            );

            convolution.push(convolution_value);
            activated.push(activated_value);
            output.push(output_bits);
            absolute_sums.push(absolute_sum);
        }
    }

    let next_history = update_history(input, history, rows, channels);
    Ok(ConvReference {
        convolution,
        activated,
        output,
        next_history,
        absolute_sums,
    })
}

fn convolve_element(
    input: &[u16],
    weights: &[u16],
    history: &[u16],
    time: usize,
    channel: usize,
    channels: usize,
) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for tap in 0..4 {
        let joined_time = time + tap;
        let input_bits = if joined_time < 3 {
            history[joined_time * channels + channel]
        } else {
            input[(joined_time - 3) * channels + channel]
        };
        let activation = f64::from(bf16_to_f32(input_bits));
        let weight = f64::from(bf16_to_f32(weights[channel * 4 + tap]));
        let product = activation * weight;
        sum += product;
        absolute_sum += product.abs();
    }
    (sum, absolute_sum)
}

fn update_history(input: &[u16], history: &[u16], rows: usize, channels: usize) -> Vec<u16> {
    let retained_input_rows = rows.min(3);
    let history_start = retained_input_rows * channels;
    let input_start = (rows - retained_input_rows) * channels;
    let mut next = Vec::with_capacity(3 * channels);
    next.extend_from_slice(&history[history_start..]);
    next.extend_from_slice(&input[input_start..]);
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    #[test]
    fn hand_computed_multichannel_convolution_uses_history_and_forward_tap_order() {
        let history = [
            bf16(1.0),
            bf16(-1.0),
            bf16(2.0),
            bf16(-0.0),
            bf16(-1.0),
            bf16(3.0),
        ];
        let input = [bf16(4.0), bf16(-2.0)];
        let weights = [
            bf16(2.0),
            bf16(-1.0),
            bf16(0.5),
            bf16(3.0),
            bf16(-1.0),
            bf16(2.0),
            bf16(0.5),
            bf16(-3.0),
        ];

        let result = run(&input, &weights, &history, 1, 2).unwrap();
        assert_eq!(result.convolution, [11.5, 8.5]);
        assert_eq!(result.absolute_sums, [16.5, 8.5]);
        assert_eq!(result.activated, [silu(11.5), silu(8.5)]);
        assert_eq!(result.output, [bf16(silu(11.5)), bf16(silu(8.5))]);
        assert_eq!(
            result.next_history,
            [
                bf16(2.0),
                bf16(-0.0),
                bf16(-1.0),
                bf16(3.0),
                bf16(4.0),
                bf16(-2.0)
            ]
        );

        let reversed: Vec<_> = weights
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|channel| channel.iter().rev().copied())
            .collect();
        assert_ne!(
            run(&input, &reversed, &history, 1, 2).unwrap().convolution,
            result.convolution
        );
    }

    #[test]
    fn full_sequence_chunked_and_tokenwise_runs_have_identical_outputs_and_state() {
        let channels = 3;
        let input: Vec<_> = (0..7 * channels)
            .map(|i| bf16(((i * 7 % 17) as f32 - 8.0) / 4.0))
            .collect();
        let weights = [
            bf16(1.0),
            bf16(-0.5),
            bf16(0.25),
            bf16(2.0),
            bf16(-1.0),
            bf16(0.5),
            bf16(1.5),
            bf16(-0.25),
            bf16(0.75),
            bf16(0.125),
            bf16(-2.0),
            bf16(1.0),
        ];
        let history = [
            bf16(1.0),
            bf16(-0.0),
            bf16(-2.0),
            bf16(0.5),
            bf16(3.0),
            bf16(-1.0),
            bf16(0.25),
            bf16(2.0),
            bf16(-0.5),
        ];
        let full = run(&input, &weights, &history, 7, channels).unwrap();
        let (chunked_output, chunked_state) =
            run_partitioned(&input, &weights, &history, channels, &[2, 1, 4]);
        let (tokenwise_output, tokenwise_state) =
            run_partitioned(&input, &weights, &history, channels, &[1, 1, 1, 1, 1, 1, 1]);

        assert_eq!(chunked_output, full.output);
        assert_eq!(chunked_state, full.next_history);
        assert_eq!(tokenwise_output, full.output);
        assert_eq!(tokenwise_state, full.next_history);
        assert_eq!(chunked_state, input[4 * channels..].to_vec());
    }

    #[test]
    fn short_sequences_preserve_prior_rows_and_reset_state_is_independent() {
        let channels = 2;
        let weights = [
            bf16(1.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(1.0),
        ];
        let history = [
            bf16(1.0),
            bf16(2.0),
            bf16(3.0),
            bf16(4.0),
            bf16(5.0),
            bf16(6.0),
        ];
        for rows in [1, 2, 7] {
            let input: Vec<_> = (0..rows * channels).map(|i| bf16(i as f32 + 7.0)).collect();
            let result = run(&input, &weights, &history, rows, channels).unwrap();
            assert_eq!(result.output.len(), rows * channels);
            assert_eq!(result.next_history.len(), 3 * channels);
        }

        let input = [bf16(9.0), bf16(10.0)];
        let zero_history = [0; 6];
        let from_previous_stream = run(&input, &weights, &history, 1, channels).unwrap();
        let after_reset = run(&input, &weights, &zero_history, 1, channels).unwrap();
        let repeated_reset = run(&input, &weights, &zero_history, 1, channels).unwrap();
        assert_ne!(from_previous_stream.convolution, after_reset.convolution);
        assert_eq!(after_reset.output, repeated_reset.output);
        assert_eq!(after_reset.next_history, repeated_reset.next_history);
    }

    fn run_partitioned(
        input: &[u16],
        weights: &[u16],
        initial_history: &[u16],
        channels: usize,
        chunks: &[usize],
    ) -> (Vec<u16>, Vec<u16>) {
        let mut output = Vec::new();
        let mut history = initial_history.to_vec();
        let mut first_row = 0;
        for &rows in chunks {
            let start = first_row * channels;
            let end = (first_row + rows) * channels;
            let result = run(&input[start..end], weights, &history, rows, channels).unwrap();
            output.extend(result.output);
            history = result.next_history;
            first_row += rows;
        }
        assert_eq!(first_row * channels, input.len());
        (output, history)
    }

    #[test]
    fn extreme_finite_silu_stays_finite_after_bf16_rounding() {
        let maximum = 0x7f7f;
        let zero_history = [0; 3];
        let weights = [0, 0, 0, bf16(1.0)];
        let result = run(&[maximum], &weights, &zero_history, 1, 1).unwrap();
        assert_eq!(result.convolution, [bf16_to_f32(maximum)]);
        assert!(result.activated[0].is_finite());
        assert_eq!(result.output, [maximum]);
        assert!(silu(-f32::MAX).is_finite());
        assert_eq!(silu(f32::MAX), f32::MAX);
    }

    #[test]
    fn rejects_invalid_dimensions_extents_nonfinite_values_and_overflow() {
        let input = [bf16(1.0)];
        let weights = [bf16(1.0); 4];
        let history = [bf16(0.0); 3];
        assert!(run(&input, &weights, &history, 0, 1).is_err());
        assert!(run(&input, &weights, &history, 1, 0).is_err());
        assert!(run(&input, &weights, &history, 2049, 1).is_err());
        assert!(run(&input, &weights, &history, 1, 32769).is_err());
        assert!(run(&[], &weights, &history, 1, 1).is_err());
        assert!(run(&input, &weights[..3], &history, 1, 1).is_err());
        assert!(run(&input, &weights, &history[..2], 1, 1).is_err());
        assert!(run(&[0x7f80], &weights, &history, 1, 1).is_err());
        assert!(run(&input, &[0x7fc0; 4], &history, 1, 1).is_err());
        assert!(run(&input, &weights, &[0x7f80; 3], 1, 1).is_err());
        assert!(run(&[0x7f7f], &[0x7f7f; 4], &history, 1, 1).is_err());
    }
}
