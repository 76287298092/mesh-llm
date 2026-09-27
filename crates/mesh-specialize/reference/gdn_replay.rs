//! Independent scalar reference for bounded GDN delta recording and replay.

use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    gdn_recurrent_reference::{Input, Shape},
};

pub const MAX_REPLAY_ROWS: usize = 5;

#[derive(Debug, PartialEq)]
pub struct Recorded {
    pub output: Vec<u16>,
    pub unrounded: Vec<f32>,
    pub delta: Vec<f32>,
    pub state: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayInput<'a> {
    pub k: &'a [f32],
    pub decay: &'a [f32],
    pub delta: &'a [f32],
}

struct Lengths {
    state: usize,
    output: usize,
    key_row: usize,
    row_width: usize,
    value_offset: usize,
    head_ratio: usize,
}

/// Independently compute the original ordered recurrence while recording deltas.
///
/// The recurrence keeps its BF16 inputs and outputs at the same boundaries as
/// `gdn_recurrent_reference::run`. All FP32 products and sums are evaluated in
/// the original explicit multiply-then-add order; each recorded value is the
/// delta for one `[row, value_head, value_column]` state column.
pub fn record(input: &Input<'_>, initial_state: &[f32], shape: &Shape) -> Result<Recorded> {
    let lengths = validate_record(input, initial_state, shape)?;
    let mut state = initial_state.to_vec();
    let mut output = Vec::with_capacity(lengths.output);
    let mut unrounded = Vec::with_capacity(lengths.output);
    let mut delta_records = Vec::with_capacity(lengths.output);

    for row in 0..shape.rows {
        let qk_row = row * lengths.key_row;
        let qkv_row = row * lengths.row_width;
        let gate_row = row * shape.value_heads;
        for value_head in 0..shape.value_heads {
            let key_head = value_head / lengths.head_ratio;
            let key_start = qk_row + key_head * shape.width;
            let state_head = value_head * shape.width * shape.width;
            let value_start = qkv_row + lengths.value_offset + value_head * shape.width;
            let decay = input.decay[gate_row + value_head];
            let beta = bf16_to_f32(input.beta[gate_row + value_head]);

            for value_column in 0..shape.width {
                let mut prediction = 0.0_f32;
                for key_dimension in 0..shape.width {
                    let state_index = state_head + key_dimension * shape.width + value_column;
                    let decayed = checked_mul(state[state_index], decay, "decayed state")?;
                    state[state_index] = decayed;
                    let term = checked_mul(
                        decayed,
                        input.k[key_start + key_dimension],
                        "prediction product",
                    )?;
                    prediction = checked_add(prediction, term, "prediction sum")?;
                }

                let value = bf16_to_f32(input.qkv[value_start + value_column]);
                let residual = checked_sub(value, prediction, "V minus prediction")?;
                let delta = checked_mul(residual, beta, "beta times residual")?;
                delta_records.push(delta);

                for key_dimension in 0..shape.width {
                    let state_index = state_head + key_dimension * shape.width + value_column;
                    let update =
                        checked_mul(input.k[key_start + key_dimension], delta, "key times delta")?;
                    state[state_index] =
                        checked_add(state[state_index], update, "updated recurrent state")?;
                }

                let mut result = 0.0_f32;
                for key_dimension in 0..shape.width {
                    let state_index = state_head + key_dimension * shape.width + value_column;
                    let term = checked_mul(
                        state[state_index],
                        input.q[key_start + key_dimension],
                        "output product",
                    )?;
                    result = checked_add(result, term, "output sum")?;
                }
                let rounded = round_bf16(result);
                ensure!(
                    bf16_to_f32(rounded).is_finite(),
                    "GDN BF16 output overflows"
                );
                unrounded.push(result);
                output.push(rounded);
            }
        }
    }

    debug_assert_eq!(state.len(), lengths.state);
    Ok(Recorded {
        output,
        unrounded,
        delta: delta_records,
        state,
    })
}

/// Apply zero through `prefix_rows` records to the untouched base state.
///
/// `shape.rows` describes the complete record buffer and must be 1..=5. A zero
/// prefix is the no-launch accepted-prefix case and returns an exact copy of the
/// base state. All supplied record rows are validated before replay.
pub fn replay_prefix(
    input: &ReplayInput<'_>,
    initial_state: &[f32],
    shape: &Shape,
    prefix_rows: usize,
) -> Result<Vec<f32>> {
    let lengths = validate_replay(input, initial_state, shape, prefix_rows)?;
    let mut state = initial_state.to_vec();
    let key_row = shape.key_heads * shape.width;

    for row in 0..prefix_rows {
        let qk_row = row * key_row;
        let gate_row = row * shape.value_heads;
        for value_head in 0..shape.value_heads {
            let key_head = value_head / lengths.head_ratio;
            let key_start = qk_row + key_head * shape.width;
            let state_head = value_head * shape.width * shape.width;
            let decay = input.decay[gate_row + value_head];
            for value_column in 0..shape.width {
                let delta_index =
                    (row * shape.value_heads + value_head) * shape.width + value_column;
                let delta = input.delta[delta_index];
                for key_dimension in 0..shape.width {
                    let state_index = state_head + key_dimension * shape.width + value_column;
                    let decayed = checked_mul(state[state_index], decay, "replay decay")?;
                    let update = checked_mul(
                        input.k[key_start + key_dimension],
                        delta,
                        "replay key times delta",
                    )?;
                    state[state_index] = checked_add(decayed, update, "replay state")?;
                }
            }
        }
    }

    ensure!(
        state.len() == lengths.state,
        "GDN replay state extent mismatch"
    );
    Ok(state)
}

fn validate_record(input: &Input<'_>, initial_state: &[f32], shape: &Shape) -> Result<Lengths> {
    let lengths = lengths(shape)?;
    let gate_len = shape.rows * shape.value_heads;
    let q_len = shape.rows * lengths.key_row;
    let qkv_len = shape.rows * lengths.row_width;
    ensure!(input.q.len() == q_len, "GDN replay Q extent mismatch");
    ensure!(input.k.len() == q_len, "GDN replay K extent mismatch");
    ensure!(input.qkv.len() == qkv_len, "GDN replay QKV extent mismatch");
    ensure!(
        input.beta.len() == gate_len,
        "GDN replay beta extent mismatch"
    );
    ensure!(
        input.decay.len() == gate_len,
        "GDN replay decay extent mismatch"
    );
    ensure!(
        initial_state.len() == lengths.state,
        "GDN replay state extent mismatch"
    );
    ensure!(
        input.q.iter().all(|value| value.is_finite()),
        "nonfinite GDN Q"
    );
    ensure!(
        input.k.iter().all(|value| value.is_finite()),
        "nonfinite GDN K"
    );
    ensure!(
        input.qkv.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "nonfinite BF16 GDN QKV"
    );
    ensure!(
        input.beta.iter().all(|&bits| {
            let value = bf16_to_f32(bits);
            value.is_finite() && (0.0..=1.0).contains(&value)
        }),
        "GDN beta outside [0, 1]"
    );
    validate_decay(input.decay)?;
    validate_state(initial_state)?;
    Ok(lengths)
}

fn validate_replay(
    input: &ReplayInput<'_>,
    initial_state: &[f32],
    shape: &Shape,
    prefix_rows: usize,
) -> Result<Lengths> {
    let lengths = lengths(shape)?;
    ensure!(
        prefix_rows <= shape.rows,
        "GDN replay prefix exceeds recorded rows"
    );
    ensure!(
        input.k.len() == shape.rows * lengths.key_row,
        "GDN replay K extent mismatch"
    );
    ensure!(
        input.decay.len() == shape.rows * shape.value_heads,
        "GDN replay decay extent mismatch"
    );
    ensure!(
        input.delta.len() == lengths.output,
        "GDN replay delta extent mismatch"
    );
    ensure!(
        initial_state.len() == lengths.state,
        "GDN replay state extent mismatch"
    );
    ensure!(
        input.k.iter().all(|value| value.is_finite()),
        "nonfinite GDN replay K"
    );
    validate_decay(input.decay)?;
    ensure!(
        input.delta.iter().all(|value| value.is_finite()),
        "nonfinite GDN replay delta"
    );
    validate_state(initial_state)?;
    Ok(lengths)
}

fn lengths(shape: &Shape) -> Result<Lengths> {
    ensure!(
        (1..=MAX_REPLAY_ROWS).contains(&shape.rows),
        "GDN replay rows must be in 1..={MAX_REPLAY_ROWS}"
    );
    ensure!(
        (1..=64).contains(&shape.key_heads),
        "invalid GDN key head count"
    );
    ensure!(
        (1..=256).contains(&shape.value_heads),
        "invalid GDN value head count"
    );
    ensure!(
        shape.value_heads.is_multiple_of(shape.key_heads),
        "GDN value heads must be divisible by key heads"
    );
    ensure!(
        (1..=256).contains(&shape.width) && shape.width.is_power_of_two(),
        "GDN width must be a power of two in 1..=256"
    );

    let key_row = checked_len(shape.key_heads, shape.width, "GDN replay Q/K row")?;
    let output = checked_len(
        checked_len(shape.rows, shape.value_heads, "GDN replay gate rows")?,
        shape.width,
        "GDN replay output",
    )?;
    let state_plane = checked_len(shape.width, shape.width, "GDN replay state plane")?;
    let state = checked_len(shape.value_heads, state_plane, "GDN replay state")?;
    let qkv_heads = shape
        .key_heads
        .checked_mul(2)
        .and_then(|value| value.checked_add(shape.value_heads))
        .context("GDN replay QKV head count overflows usize")?;
    let row_width = checked_len(qkv_heads, shape.width, "GDN replay QKV row")?;
    let value_offset = checked_len(
        shape
            .key_heads
            .checked_mul(2)
            .context("GDN replay QKV offset overflows usize")?,
        shape.width,
        "GDN replay value offset",
    )?;
    Ok(Lengths {
        state,
        output,
        key_row,
        row_width,
        value_offset,
        head_ratio: shape.value_heads / shape.key_heads,
    })
}

fn checked_len(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} extent overflows usize"))
}

fn validate_decay(decay: &[f32]) -> Result<()> {
    ensure!(
        decay
            .iter()
            .all(|&value| value.is_finite() && (0.0..=1.0).contains(&value)),
        "GDN decay outside [0, 1]"
    );
    Ok(())
}

fn validate_state(state: &[f32]) -> Result<()> {
    ensure!(
        state.iter().all(|value| value.is_finite()),
        "nonfinite initial GDN state"
    );
    Ok(())
}

fn checked_mul(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left * right, label)
}

fn checked_sub(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left - right, label)
}

fn checked_add(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left + right, label)
}

fn checked_f32(value: f32, label: &str) -> Result<f32> {
    ensure!(value.is_finite(), "GDN {label} overflows FP32");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn_recurrent_reference::{self as recurrence, Reduction};

    struct Fixture {
        shape: Shape,
        q: Vec<f32>,
        k: Vec<f32>,
        qkv: Vec<u16>,
        beta: Vec<u16>,
        decay: Vec<f32>,
        initial: Vec<f32>,
    }

    impl Fixture {
        fn new(width: usize) -> Self {
            let shape = Shape {
                rows: MAX_REPLAY_ROWS,
                key_heads: 2,
                value_heads: 4,
                width,
            };
            let key_row = shape.key_heads * shape.width;
            let row_width = (2 * shape.key_heads + shape.value_heads) * shape.width;
            let value_offset = 2 * shape.key_heads * shape.width;
            let q_values = [0.1_f32, -0.3, 0.7, -0.45, 0.23, -0.81, 0.57];
            let k_values = [-0.17_f32, 0.31, -0.62, 0.44, 0.09, -0.73, 0.86];
            let q = (0..shape.rows * key_row)
                .map(|index| q_values[index % q_values.len()] + index as f32 * 0.0013)
                .collect();
            let k = (0..shape.rows * key_row)
                .map(|index| k_values[index % k_values.len()] - index as f32 * 0.0009)
                .collect();
            let mut qkv = vec![round_bf16(0.0); shape.rows * row_width];
            let mut beta = Vec::with_capacity(shape.rows * shape.value_heads);
            let mut decay = Vec::with_capacity(shape.rows * shape.value_heads);
            let beta_values = [0.0, 0.17, 0.33, 0.68, 1.0];
            let decay_values = [0.0_f32, 0.13, 0.29, 0.47, 0.72, 0.91, 1.0];
            let value_values = [-0.73, -0.31, 0.0, 0.24, 0.59, 0.87];
            for row in 0..shape.rows {
                for head in 0..shape.value_heads {
                    beta.push(round_bf16(beta_values[(row + head) % beta_values.len()]));
                    decay.push(decay_values[(3 * row + 2 * head) % decay_values.len()]);
                    for column in 0..shape.width {
                        let index = row * row_width + value_offset + head * shape.width + column;
                        qkv[index] = round_bf16(
                            value_values[(row + 2 * head + column) % value_values.len()],
                        );
                    }
                }
            }
            let initial = (0..shape.value_heads * shape.width * shape.width)
                .map(|index| {
                    let base = [-0.37_f32, 0.28, -0.113, 0.19, 0.51, -0.67, 0.043];
                    base[index % base.len()] + index as f32 * 0.0007
                })
                .collect();
            Self {
                shape,
                q,
                k,
                qkv,
                beta,
                decay,
                initial,
            }
        }

        fn input(&self) -> Input<'_> {
            Input {
                q: &self.q,
                k: &self.k,
                qkv: &self.qkv,
                beta: &self.beta,
                decay: &self.decay,
            }
        }

        fn replay_input<'a>(&'a self, delta: &'a [f32]) -> ReplayInput<'a> {
            ReplayInput {
                k: &self.k,
                decay: &self.decay,
                delta,
            }
        }

        fn prefix_input(&self, rows: usize) -> Input<'_> {
            let key_row = self.shape.key_heads * self.shape.width;
            let row_width = (2 * self.shape.key_heads + self.shape.value_heads) * self.shape.width;
            let gates = self.shape.value_heads;
            Input {
                q: &self.q[..rows * key_row],
                k: &self.k[..rows * key_row],
                qkv: &self.qkv[..rows * row_width],
                beta: &self.beta[..rows * gates],
                decay: &self.decay[..rows * gates],
            }
        }
    }

    fn shape_with_rows(shape: &Shape, rows: usize) -> Shape {
        Shape {
            rows,
            key_heads: shape.key_heads,
            value_heads: shape.value_heads,
            width: shape.width,
        }
    }

    fn assert_same_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "FP32 value differs at element {index}: {actual} vs {expected}"
            );
        }
    }

    fn assert_every_prefix_matches_original(fixture: &Fixture, delta: &[f32]) {
        let replay_input = fixture.replay_input(delta);
        for accepted_rows in 0..=fixture.shape.rows {
            let replayed = replay_prefix(
                &replay_input,
                &fixture.initial,
                &fixture.shape,
                accepted_rows,
            )
            .unwrap();
            let expected = if accepted_rows == 0 {
                fixture.initial.clone()
            } else {
                recurrence::run(
                    &fixture.prefix_input(accepted_rows),
                    &fixture.initial,
                    &shape_with_rows(&fixture.shape, accepted_rows),
                    Reduction::OrderedF32,
                )
                .unwrap()
                .state
            };
            assert_same_bits(&replayed, &expected);
        }
    }

    fn assert_fixture_covers_arithmetic_edges(fixture: &Fixture) {
        assert_signed_inputs(fixture);
        assert_bf16_values_are_signed(fixture);
        assert_gate_endpoints(fixture);
        assert_f32_inputs_are_nonbinary(fixture);
    }

    fn assert_signed_inputs(fixture: &Fixture) {
        assert!(fixture.q.iter().any(|value| *value < 0.0));
        assert!(fixture.q.iter().any(|value| *value > 0.0));
        assert!(fixture.k.iter().any(|value| *value < 0.0));
        assert!(fixture.k.iter().any(|value| *value > 0.0));
        assert!(fixture.initial.iter().any(|value| *value != 0.0));
    }

    fn assert_bf16_values_are_signed(fixture: &Fixture) {
        assert!(fixture.qkv.iter().any(|bits| bf16_to_f32(*bits) < 0.0));
        assert!(fixture.qkv.iter().any(|bits| bf16_to_f32(*bits) > 0.0));
    }

    fn assert_gate_endpoints(fixture: &Fixture) {
        assert!(fixture.decay.contains(&0.0));
        assert!(fixture.decay.contains(&1.0));
        assert!(fixture.decay.windows(2).any(|pair| pair[0] != pair[1]));
        let mut beta_values = fixture.beta.iter().copied().map(bf16_to_f32);
        assert!(beta_values.clone().any(|value| value == 0.0));
        assert!(beta_values.any(|value| value == 1.0));
    }

    fn assert_f32_inputs_are_nonbinary(fixture: &Fixture) {
        assert_eq!(fixture.q[0], 0.1_f32);
        assert_eq!(fixture.k[0], -0.17_f32);
        assert!(fixture.decay.contains(&0.13_f32));
        assert_eq!(fixture.initial[0], -0.37_f32);
        assert_ne!(f64::from(0.1_f32), 0.1_f64);
        assert_ne!(f64::from(0.13_f32), 0.13_f64);
        assert_ne!(f64::from(-0.17_f32), -0.17_f64);
        assert_ne!(f64::from(-0.37_f32), -0.37_f64);
    }

    fn assert_one_width_replays(width: usize) {
        let fixture = Fixture::new(width);
        let recorded = record(&fixture.input(), &fixture.initial, &fixture.shape).unwrap();
        let full = recurrence::run(
            &fixture.input(),
            &fixture.initial,
            &fixture.shape,
            Reduction::OrderedF32,
        )
        .unwrap();
        assert_eq!(recorded.output, full.output);
        assert_same_bits(&recorded.unrounded, &full.unrounded);
        assert_same_bits(&recorded.state, &full.state);
        assert_every_prefix_matches_original(&fixture, &recorded.delta);
        assert_fixture_covers_arithmetic_edges(&fixture);
    }

    #[test]
    fn every_accepted_prefix_replays_to_the_original_recurrent_state() {
        for width in [1, 2, 128] {
            assert_one_width_replays(width);
        }
    }

    #[test]
    fn rejects_invalid_shapes_and_extents() {
        let fixture = Fixture::new(2);
        let good = record(&fixture.input(), &fixture.initial, &fixture.shape).unwrap();
        let replay = fixture.replay_input(&good.delta);

        for rows in [0, MAX_REPLAY_ROWS + 1] {
            assert!(
                record(
                    &fixture.input(),
                    &fixture.initial,
                    &shape_with_rows(&fixture.shape, rows)
                )
                .is_err()
            );
        }
        let bad_width = Shape {
            width: 3,
            ..fixture.shape.clone()
        };
        assert!(record(&fixture.input(), &fixture.initial, &bad_width).is_err());
        let bad_head_ratio = Shape {
            key_heads: 3,
            ..fixture.shape.clone()
        };
        assert!(record(&fixture.input(), &fixture.initial, &bad_head_ratio).is_err());
        assert!(
            replay_prefix(
                &replay,
                &fixture.initial,
                &fixture.shape,
                MAX_REPLAY_ROWS + 1
            )
            .is_err()
        );

        let short_k = &fixture.k[..fixture.k.len() - 1];
        assert!(
            replay_prefix(
                &ReplayInput {
                    k: short_k,
                    ..replay
                },
                &fixture.initial,
                &fixture.shape,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_nonfinite_and_out_of_domain_record_inputs() {
        let fixture = Fixture::new(2);

        let mut bad_q = fixture.q.clone();
        bad_q[0] = f32::INFINITY;
        assert!(
            record(
                &Input {
                    q: &bad_q,
                    ..fixture.input()
                },
                &fixture.initial,
                &fixture.shape
            )
            .is_err()
        );

        let mut bad_qkv = fixture.qkv.clone();
        bad_qkv[0] = 0x7f80;
        assert!(
            record(
                &Input {
                    qkv: &bad_qkv,
                    ..fixture.input()
                },
                &fixture.initial,
                &fixture.shape
            )
            .is_err()
        );

        let mut bad_beta = fixture.beta.clone();
        bad_beta[0] = round_bf16(1.5);
        assert!(
            record(
                &Input {
                    beta: &bad_beta,
                    ..fixture.input()
                },
                &fixture.initial,
                &fixture.shape
            )
            .is_err()
        );

        let mut bad_decay = fixture.decay.clone();
        bad_decay[0] = 1.5;
        assert!(
            record(
                &Input {
                    decay: &bad_decay,
                    ..fixture.input()
                },
                &fixture.initial,
                &fixture.shape
            )
            .is_err()
        );

        let mut bad_initial = fixture.initial.clone();
        bad_initial[0] = f32::NEG_INFINITY;
        assert!(record(&fixture.input(), &bad_initial, &fixture.shape).is_err());
    }

    #[test]
    fn rejects_nonfinite_replay_inputs_even_for_empty_prefix() {
        let fixture = Fixture::new(2);
        let good = record(&fixture.input(), &fixture.initial, &fixture.shape).unwrap();
        let replay = fixture.replay_input(&good.delta);

        let mut bad_k = fixture.k.clone();
        bad_k[0] = f32::INFINITY;
        assert!(
            replay_prefix(
                &ReplayInput {
                    k: &bad_k,
                    ..replay
                },
                &fixture.initial,
                &fixture.shape,
                0
            )
            .is_err()
        );

        let mut bad_decay = fixture.decay.clone();
        bad_decay[0] = f32::NAN;
        assert!(
            replay_prefix(
                &ReplayInput {
                    decay: &bad_decay,
                    ..replay
                },
                &fixture.initial,
                &fixture.shape,
                0
            )
            .is_err()
        );

        let mut bad_delta = good.delta.clone();
        bad_delta[0] = f32::NAN;
        assert!(
            replay_prefix(
                &fixture.replay_input(&bad_delta),
                &fixture.initial,
                &fixture.shape,
                0
            )
            .is_err()
        );

        let mut bad_initial = fixture.initial.clone();
        bad_initial[0] = f32::NEG_INFINITY;
        assert!(replay_prefix(&replay, &bad_initial, &fixture.shape, 0).is_err());
    }
}
