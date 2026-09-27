//! Experimental matrix-form GDN chunk reference.
//!
//! This keeps the logical input and shape contract of `gdn_recurrent_reference`
//! while reassociating a bounded sequence into a lower-triangular solve. It is a
//! correctness candidate, not the runtime default or a bit-exact replacement.

use anyhow::{Context, Result, ensure};

use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    gdn_recurrent_reference::{Input, Shape},
};

pub const MAX_CHUNK_ROWS: usize = 16;
pub const MAX_WIDTH: usize = 128;
const DECAY_STRIDE: usize = MAX_CHUNK_ROWS + 1;

#[derive(Debug, PartialEq)]
pub struct ChunkedOutput {
    pub output: Vec<u16>,
    pub unrounded: Vec<f32>,
    pub state: Vec<f32>,
}

struct Lengths {
    state_len: usize,
    output_len: usize,
    key_row: usize,
    row_width: usize,
    value_offset: usize,
    ratio: usize,
}

struct PreparedHead {
    rows: usize,
    width: usize,
    coefficients: Vec<f32>,
    rhs: Vec<f32>,
    decay_products: Vec<f32>,
}

struct FinishInput<'a, 'input> {
    input: &'a Input<'input>,
    initial_state: &'a [f32],
    shape: &'a Shape,
    lengths: &'a Lengths,
    value_head: usize,
    prepared: &'a PreparedHead,
    updates: &'a [f32],
}

struct OutputBuffers<'a> {
    output: &'a mut [u16],
    unrounded: &'a mut [f32],
    final_state: &'a mut [f32],
}

/// Run one chunk using the matrix-form GDN equations.
///
/// For row `t`, `D(t, j)` is the product of decays from `j + 1` through `t`,
/// and `D(t, -1)` is the product from zero through `t`. Products are formed
/// directly; no cumulative-decay division is used, so zero and underflowed
/// prefixes are well-defined. Scratch on the host is bounded by 16 rows and a
/// width of 128. The GPU candidate stores fixed-stride versions of these same
/// intermediates; see `kernels/nvptx/gdn_chunked.rs`.
pub fn run_chunked(
    input: &Input<'_>,
    initial_state: &[f32],
    shape: &Shape,
) -> Result<ChunkedOutput> {
    let lengths = validate(input, initial_state, shape)?;
    let mut output = vec![0; lengths.output_len];
    let mut unrounded = vec![0.0; lengths.output_len];
    let mut state = vec![0.0; lengths.state_len];

    for value_head in 0..shape.value_heads {
        let prepared = prepare_head(input, initial_state, shape, &lengths, value_head)?;
        let updates = solve_head(&prepared)?;
        finish_head(
            FinishInput {
                input,
                initial_state,
                shape,
                lengths: &lengths,
                value_head,
                prepared: &prepared,
                updates: &updates,
            },
            OutputBuffers {
                output: &mut output,
                unrounded: &mut unrounded,
                final_state: &mut state,
            },
        )?;
    }

    Ok(ChunkedOutput {
        output,
        unrounded,
        state,
    })
}

fn validate(input: &Input<'_>, initial_state: &[f32], shape: &Shape) -> Result<Lengths> {
    ensure!(
        (1..=MAX_CHUNK_ROWS).contains(&shape.rows),
        "chunked GDN rows must be in 1..={MAX_CHUNK_ROWS}"
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
        (1..=MAX_WIDTH).contains(&shape.width) && shape.width.is_power_of_two(),
        "chunked GDN width must be a power of two in 1..={MAX_WIDTH}"
    );

    let key_row = checked_len(shape.key_heads, shape.width, "Q/K row")?;
    let gate_len = checked_len(shape.rows, shape.value_heads, "gate input")?;
    let q_len = checked_len(shape.rows, key_row, "Q/K input")?;
    let output_len = checked_len(gate_len, shape.width, "GDN output")?;
    let state_plane = checked_len(shape.width, shape.width, "state plane")?;
    let state_len = checked_len(shape.value_heads, state_plane, "GDN state")?;
    let qkv_heads = shape
        .key_heads
        .checked_mul(2)
        .and_then(|count| count.checked_add(shape.value_heads))
        .context("GDN QKV head count overflows usize")?;
    let row_width = checked_len(qkv_heads, shape.width, "QKV row")?;
    let qkv_len = checked_len(shape.rows, row_width, "QKV input")?;
    let value_offset = checked_len(
        shape
            .key_heads
            .checked_mul(2)
            .context("QKV offset overflows usize")?,
        shape.width,
        "QKV value offset",
    )?;

    ensure!(input.q.len() == q_len, "GDN Q extent mismatch");
    ensure!(input.k.len() == q_len, "GDN K extent mismatch");
    ensure!(input.qkv.len() == qkv_len, "GDN QKV extent mismatch");
    ensure!(input.beta.len() == gate_len, "GDN beta extent mismatch");
    ensure!(input.decay.len() == gate_len, "GDN decay extent mismatch");
    ensure!(
        initial_state.len() == state_len,
        "GDN state extent mismatch"
    );
    ensure!(input.q.iter().all(|value| value.is_finite()), "nonfinite Q");
    ensure!(input.k.iter().all(|value| value.is_finite()), "nonfinite K");
    ensure!(
        input.qkv.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "nonfinite BF16 QKV"
    );
    ensure!(
        input.beta.iter().all(|&bits| {
            let value = bf16_to_f32(bits);
            value.is_finite() && (0.0..=1.0).contains(&value)
        }),
        "GDN beta outside [0, 1]"
    );
    ensure!(
        input
            .decay
            .iter()
            .all(|&value| value.is_finite() && (0.0..=1.0).contains(&value)),
        "GDN decay outside [0, 1]"
    );
    ensure!(
        initial_state.iter().all(|value| value.is_finite()),
        "nonfinite initial GDN state"
    );

    Ok(Lengths {
        state_len,
        output_len,
        key_row,
        row_width,
        value_offset,
        ratio: shape.value_heads / shape.key_heads,
    })
}

fn checked_len(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} extent overflows usize"))
}

fn prepare_head(
    input: &Input<'_>,
    initial_state: &[f32],
    shape: &Shape,
    lengths: &Lengths,
    value_head: usize,
) -> Result<PreparedHead> {
    let rows = shape.rows;
    let width = shape.width;
    let key_head = value_head / lengths.ratio;
    let state_head_start = value_head * width * width;
    let mut prepared = PreparedHead {
        rows,
        width,
        coefficients: vec![0.0; rows * rows],
        rhs: vec![0.0; rows * width],
        decay_products: vec![0.0; rows * DECAY_STRIDE],
    };

    for time in 0..rows {
        let gate_index = time * shape.value_heads + value_head;
        let beta = bf16_to_f32(input.beta[gate_index]);
        let prefix = product_range(input.decay, shape.value_heads, value_head, 0, time + 1)?;
        prepared.decay_products[time * DECAY_STRIDE + MAX_CHUNK_ROWS] = prefix;
        let key_start = time * lengths.key_row + key_head * width;

        for key_time in 0..=time {
            let decay = product_range(
                input.decay,
                shape.value_heads,
                value_head,
                key_time + 1,
                time + 1,
            )?;
            prepared.decay_products[time * DECAY_STRIDE + key_time] = decay;
            if key_time < time {
                let other_start = key_time * lengths.key_row + key_head * width;
                let key_dot = dot_f32(
                    &input.k[key_start..key_start + width],
                    &input.k[other_start..other_start + width],
                    "GDN key Gram dot",
                )?;
                let coefficient = checked_mul(checked_mul(beta, decay)?, key_dot)?;
                prepared.coefficients[time * rows + key_time] = coefficient;
            }
        }

        let value_start = time * lengths.row_width + lengths.value_offset + value_head * width;
        let initial_row = state_head_start;
        for value_dimension in 0..width {
            let state_dot = dot_state_column(
                initial_state,
                initial_row,
                width,
                value_dimension,
                &input.k[key_start..key_start + width],
            )?;
            let decayed_prediction = checked_mul(prefix, state_dot)?;
            let value = bf16_to_f32(input.qkv[value_start + value_dimension]);
            let residual = checked_sub(value, decayed_prediction)?;
            prepared.rhs[time * width + value_dimension] = checked_mul(beta, residual)?;
        }
    }

    Ok(prepared)
}

fn product_range(
    values: &[f32],
    value_heads: usize,
    value_head: usize,
    begin: usize,
    end: usize,
) -> Result<f32> {
    let mut product = 1.0_f32;
    for index in begin..end {
        product = checked_mul(product, values[index * value_heads + value_head])?;
    }
    Ok(product)
}

fn dot_f32(left: &[f32], right: &[f32], label: &str) -> Result<f32> {
    let mut sum = 0.0_f32;
    for (&left_value, &right_value) in left.iter().zip(right) {
        sum = checked_add(sum, checked_mul(left_value, right_value)?, label)?;
    }
    Ok(sum)
}

fn dot_state_column(
    state: &[f32],
    head_start: usize,
    width: usize,
    value_dimension: usize,
    vector: &[f32],
) -> Result<f32> {
    let mut sum = 0.0_f32;
    for (key_dimension, &key_value) in vector.iter().enumerate() {
        let state_value = state[head_start + key_dimension * width + value_dimension];
        sum = checked_add(
            sum,
            checked_mul(state_value, key_value)?,
            "GDN state-vector dot",
        )?;
    }
    Ok(sum)
}

fn solve_head(prepared: &PreparedHead) -> Result<Vec<f32>> {
    let mut updates = vec![0.0; prepared.rows * prepared.width];
    for time in 0..prepared.rows {
        for value_dimension in 0..prepared.width {
            let mut prior_updates = 0.0_f32;
            for key_time in 0..time {
                let product = checked_mul(
                    prepared.coefficients[time * prepared.rows + key_time],
                    updates[key_time * prepared.width + value_dimension],
                )?;
                prior_updates = checked_add(prior_updates, product, "GDN triangular sum")?;
            }
            updates[time * prepared.width + value_dimension] = checked_sub(
                prepared.rhs[time * prepared.width + value_dimension],
                prior_updates,
            )?;
        }
    }
    Ok(updates)
}

fn finish_head(work: FinishInput<'_, '_>, buffers: OutputBuffers<'_>) -> Result<()> {
    let FinishInput {
        input,
        initial_state,
        shape,
        lengths,
        value_head,
        prepared,
        updates,
    } = work;
    let OutputBuffers {
        output,
        unrounded,
        final_state,
    } = buffers;
    let width = shape.width;
    let key_head = value_head / lengths.ratio;
    let state_head_start = value_head * width * width;

    for time in 0..shape.rows {
        let key_start = time * lengths.key_row + key_head * width;
        let q = &input.q[key_start..key_start + width];
        let prefix = prepared.decay_products[time * DECAY_STRIDE + MAX_CHUNK_ROWS];
        for value_dimension in 0..width {
            let initial_dot =
                dot_state_column(initial_state, state_head_start, width, value_dimension, q)?;
            let mut result = checked_mul(prefix, initial_dot)?;
            for key_time in 0..=time {
                let other_start = key_time * lengths.key_row + key_head * width;
                let qk_dot = dot_f32(
                    q,
                    &input.k[other_start..other_start + width],
                    "GDN query-key dot",
                )?;
                let decay = prepared.decay_products[time * DECAY_STRIDE + key_time];
                let weighted_dot = checked_mul(decay, qk_dot)?;
                let term = checked_mul(weighted_dot, updates[key_time * width + value_dimension])?;
                result = checked_add(result, term, "GDN chunk output")?;
            }

            let output_index =
                time * shape.value_heads * width + value_head * width + value_dimension;
            let rounded = round_bf16(result);
            ensure!(
                bf16_to_f32(rounded).is_finite(),
                "GDN BF16 output overflows"
            );
            output[output_index] = rounded;
            unrounded[output_index] = result;
        }
    }

    let last_time = shape.rows - 1;
    let final_prefix = prepared.decay_products[last_time * DECAY_STRIDE + MAX_CHUNK_ROWS];
    for key_dimension in 0..width {
        for value_dimension in 0..width {
            let initial = initial_state[state_head_start + key_dimension * width + value_dimension];
            let mut update_sum = 0.0_f32;
            for time in 0..shape.rows {
                let key_start = time * lengths.key_row + key_head * width;
                let decay = prepared.decay_products[last_time * DECAY_STRIDE + time];
                let term = checked_mul(
                    checked_mul(decay, input.k[key_start + key_dimension])?,
                    updates[time * width + value_dimension],
                )?;
                update_sum = checked_add(update_sum, term, "GDN final-state update sum")?;
            }
            final_state[state_head_start + key_dimension * width + value_dimension] = checked_add(
                checked_mul(final_prefix, initial)?,
                update_sum,
                "GDN final state",
            )?;
        }
    }
    Ok(())
}

fn checked_mul(left: f32, right: f32) -> Result<f32> {
    checked_f32(left * right, "FP32 multiply")
}

fn checked_sub(left: f32, right: f32) -> Result<f32> {
    checked_f32(left - right, "FP32 subtract")
}

fn checked_add(left: f32, right: f32, label: &str) -> Result<f32> {
    checked_f32(left + right, label)
}

fn checked_f32(value: f32, label: &str) -> Result<f32> {
    ensure!(value.is_finite(), "GDN {label} overflows");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn_recurrent_reference::{self, Reduction};

    struct Fixture {
        shape: Shape,
        q: Vec<f32>,
        k: Vec<f32>,
        qkv: Vec<u16>,
        beta: Vec<u16>,
        decay: Vec<f32>,
        initial_state: Vec<f32>,
    }

    impl Fixture {
        fn input(&self) -> Input<'_> {
            Input {
                q: &self.q,
                k: &self.k,
                qkv: &self.qkv,
                beta: &self.beta,
                decay: &self.decay,
            }
        }
    }

    fn fixture(rows: usize, zero_and_unit_decay: bool) -> Fixture {
        let shape = Shape {
            rows,
            key_heads: 2,
            value_heads: 4,
            width: 8,
        };
        let key_len = rows * shape.key_heads * shape.width;
        let row_width = (2 * shape.key_heads + shape.value_heads) * shape.width;
        let state_len = shape.value_heads * shape.width * shape.width;
        let q = (0..key_len)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let k = (0..key_len)
            .map(|index| ((index * 7 % 29) as f32 - 14.0) / 32.0)
            .collect();
        let qkv: Vec<u16> = (0..rows * row_width)
            .map(|index| round_bf16(((index * 5 % 23) as f32 - 11.0) / 32.0))
            .collect();
        let beta = (0..rows * shape.value_heads)
            .map(|index| round_bf16([0.25_f32, 0.5, 0.75, 1.0][index % 4]))
            .collect();
        let decay = (0..rows * shape.value_heads)
            .map(|index| {
                if zero_and_unit_decay {
                    [0.0_f32, 1.0, 0.5, 0.0, 0.75][index % 5]
                } else {
                    [0.25_f32, 0.5, 0.75, 1.0][index % 4]
                }
            })
            .collect();
        let initial_state = (0..state_len)
            .map(|index| ((index * 13 % 41) as f32 - 20.0) / 64.0)
            .collect();
        assert_eq!(qkv.len(), rows * row_width);
        Fixture {
            shape,
            q,
            k,
            qkv,
            beta,
            decay,
            initial_state,
        }
    }

    fn scalar_f64(input: &Input<'_>, initial: &[f32], shape: &Shape) -> (Vec<f64>, Vec<f64>) {
        let width = shape.width;
        let key_row = shape.key_heads * width;
        let row_width = (2 * shape.key_heads + shape.value_heads) * width;
        let value_offset = 2 * shape.key_heads * width;
        let mut state: Vec<f64> = initial.iter().map(|&value| f64::from(value)).collect();
        let mut output = vec![0.0; shape.rows * shape.value_heads * width];

        for time in 0..shape.rows {
            for value_head in 0..shape.value_heads {
                let key_head = value_head / (shape.value_heads / shape.key_heads);
                let key_start = time * key_row + key_head * width;
                let state_start = value_head * width * width;
                let gate_index = time * shape.value_heads + value_head;
                let decay = f64::from(input.decay[gate_index]);
                let beta = f64::from(bf16_to_f32(input.beta[gate_index]));
                let value_start = time * row_width + value_offset + value_head * width;

                for value_dimension in 0..width {
                    for key_dimension in 0..width {
                        let index = state_start + key_dimension * width + value_dimension;
                        state[index] *= decay;
                    }
                    let mut prediction = 0.0_f64;
                    for key_dimension in 0..width {
                        let key = f64::from(input.k[key_start + key_dimension]);
                        let value = state[state_start + key_dimension * width + value_dimension];
                        prediction += key * value;
                    }
                    let residual = f64::from(bf16_to_f32(input.qkv[value_start + value_dimension]))
                        - prediction;
                    let delta = beta * residual;
                    for key_dimension in 0..width {
                        let index = state_start + key_dimension * width + value_dimension;
                        state[index] += f64::from(input.k[key_start + key_dimension]) * delta;
                    }
                    let mut result = 0.0_f64;
                    for key_dimension in 0..width {
                        result += f64::from(input.q[key_start + key_dimension])
                            * state[state_start + key_dimension * width + value_dimension];
                    }
                    output
                        [time * shape.value_heads * width + value_head * width + value_dimension] =
                        result;
                }
            }
        }
        (output, state)
    }

    fn max_abs_f32_f64(actual: &[f32], expected: &[f64]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(&left, &right)| (f64::from(left) - right).abs())
            .fold(0.0_f64, f64::max)
    }

    fn assert_candidate_matches_scalar(fixture: &Fixture) {
        let input = fixture.input();
        let candidate = run_chunked(&input, &fixture.initial_state, &fixture.shape).unwrap();
        let (oracle_output, oracle_state) =
            scalar_f64(&input, &fixture.initial_state, &fixture.shape);
        let output_error = max_abs_f32_f64(&candidate.unrounded, &oracle_output);
        let state_error = max_abs_f32_f64(&candidate.state, &oracle_state);
        assert!(
            output_error <= 2.0e-4,
            "F04 chunk/scalar-f64 report: rows={} heads={}/{} width={} output_max_abs={output_error:.3e} state_max_abs={state_error:.3e}",
            fixture.shape.rows,
            fixture.shape.key_heads,
            fixture.shape.value_heads,
            fixture.shape.width
        );
        assert!(
            state_error <= 2.0e-4,
            "F04 chunk/scalar-f64 report: rows={} heads={}/{} width={} output_max_abs={output_error:.3e} state_max_abs={state_error:.3e}",
            fixture.shape.rows,
            fixture.shape.key_heads,
            fixture.shape.value_heads,
            fixture.shape.width
        );

        let ordered = gdn_recurrent_reference::run(
            &input,
            &fixture.initial_state,
            &fixture.shape,
            Reduction::OrderedF32,
        )
        .unwrap();
        let ordered_output_error = candidate
            .unrounded
            .iter()
            .zip(&ordered.unrounded)
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let ordered_state_error = candidate
            .state
            .iter()
            .zip(&ordered.state)
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            ordered_output_error <= 2.0e-4 && ordered_state_error <= 2.0e-4,
            "F04 chunk/ordered-f32 report: rows={} output_max_abs={ordered_output_error:.3e} state_max_abs={ordered_state_error:.3e}",
            fixture.shape.rows
        );
        assert!(
            candidate
                .output
                .iter()
                .zip(&candidate.unrounded)
                .all(|(&rounded, &value)| rounded == round_bf16(value))
        );
    }

    fn slice_input(fixture: &Fixture, start: usize, end: usize) -> Input<'_> {
        let key_row = fixture.shape.key_heads * fixture.shape.width;
        let row_width =
            (2 * fixture.shape.key_heads + fixture.shape.value_heads) * fixture.shape.width;
        let gate_width = fixture.shape.value_heads;
        Input {
            q: &fixture.q[start * key_row..end * key_row],
            k: &fixture.k[start * key_row..end * key_row],
            qkv: &fixture.qkv[start * row_width..end * row_width],
            beta: &fixture.beta[start * gate_width..end * gate_width],
            decay: &fixture.decay[start * gate_width..end * gate_width],
        }
    }

    #[test]
    fn chunk_formula_matches_nonzero_state_with_grouped_heads_and_signed_vectors() {
        assert_candidate_matches_scalar(&fixture(16, false));
    }

    #[test]
    fn zero_and_unit_decay_prefixes_need_no_division() {
        assert_candidate_matches_scalar(&fixture(16, true));
    }

    #[test]
    fn one_row_and_full_boundary_chunks_match_scalar_oracle() {
        for rows in [1, 15, 16] {
            assert_candidate_matches_scalar(&fixture(rows, rows == 15));
        }
    }

    #[test]
    fn sixteen_plus_one_tail_matches_one_scalar_sequence() {
        let fixture = fixture(17, true);
        let first = run_chunked(
            &slice_input(&fixture, 0, 16),
            &fixture.initial_state,
            &Shape {
                rows: 16,
                ..fixture.shape.clone()
            },
        )
        .unwrap();
        let tail = run_chunked(
            &slice_input(&fixture, 16, 17),
            &first.state,
            &Shape {
                rows: 1,
                ..fixture.shape.clone()
            },
        )
        .unwrap();
        let (oracle_output, oracle_state) =
            scalar_f64(&fixture.input(), &fixture.initial_state, &fixture.shape);
        let chunked_output = [first.unrounded, tail.unrounded].concat();
        let output_error = max_abs_f32_f64(&chunked_output, &oracle_output);
        let state_error = max_abs_f32_f64(&tail.state, &oracle_state);
        assert!(
            output_error <= 2.0e-4 && state_error <= 2.0e-4,
            "F04 16+1 tail/scalar-f64 report: output_max_abs={output_error:.3e} state_max_abs={state_error:.3e}"
        );
    }
}
