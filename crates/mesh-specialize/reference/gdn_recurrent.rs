//! Independent logical scalar reference for the GDN recurrent update.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub qkv: &'a [u16],
    pub beta: &'a [u16],
    pub decay: &'a [f32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    /// Increasing-key FP32 multiply then add, with no fused multiply-add.
    OrderedF32,
    /// Accumulate exact products of the FP32 operands in FP64, then cast once.
    WideF64,
}

#[derive(Debug, PartialEq)]
pub struct Output {
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

/// Run the recurrence in time-major logical head order.
///
/// State is `[value_heads, key_dimension, value_dimension]`; value head `h`
/// uses key head `h / (value_heads / key_heads)`. Only the final output is
/// rounded to BF16. `WideF64` changes only the two dot-product reductions.
pub fn run(
    input: &Input<'_>,
    initial_state: &[f32],
    shape: &Shape,
    reduction: Reduction,
) -> Result<Output> {
    let lengths = validate(input, initial_state, shape)?;
    let mut state = initial_state.to_vec();
    let mut unrounded = Vec::with_capacity(lengths.output_len);
    let mut output = Vec::with_capacity(lengths.output_len);

    for row in 0..shape.rows {
        let row_q_start = row * lengths.key_row;
        let row_qkv_start = row * lengths.row_width;
        let row_gate_start = row * shape.value_heads;
        for value_head in 0..shape.value_heads {
            let key_head = value_head / lengths.ratio;
            let key_start = row_q_start + key_head * shape.width;
            let state_head_start = value_head * shape.width * shape.width;
            let decay = input.decay[row_gate_start + value_head];
            let beta = bf16_to_f32(input.beta[row_gate_start + value_head]);
            let value_start = row_qkv_start + lengths.value_offset + value_head * shape.width;

            for value_dimension in 0..shape.width {
                decay_state_column(
                    &mut state,
                    state_head_start,
                    shape.width,
                    value_dimension,
                    decay,
                )?;
                let prediction = dot_state_column(
                    &state,
                    state_head_start,
                    shape.width,
                    value_dimension,
                    &input.k[key_start..key_start + shape.width],
                    reduction,
                )?;
                let value = bf16_to_f32(input.qkv[value_start + value_dimension]);
                let residual = checked_sub(value, prediction, "V minus prediction")?;
                let delta = checked_mul(residual, beta, "beta times residual")?;
                update_state_column(
                    &mut state,
                    state_head_start,
                    shape.width,
                    value_dimension,
                    &input.k[key_start..key_start + shape.width],
                    delta,
                )?;
                let result = dot_state_column(
                    &state,
                    state_head_start,
                    shape.width,
                    value_dimension,
                    &input.q[key_start..key_start + shape.width],
                    reduction,
                )?;
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

    debug_assert_eq!(state.len(), lengths.state_len);
    Ok(Output {
        output,
        unrounded,
        state,
    })
}

fn validate(input: &Input<'_>, initial_state: &[f32], shape: &Shape) -> Result<Lengths> {
    ensure!((1..=2048).contains(&shape.rows), "invalid GDN row count");
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

    let key_row = checked_len(shape.key_heads, shape.width, "Q/K row")?;
    let gate_len = checked_len(shape.rows, shape.value_heads, "gate input")?;
    let q_len = checked_len(shape.rows, key_row, "Q/K input")?;
    let output_len = checked_len(gate_len, shape.width, "GDN output")?;
    let state_plane = checked_len(shape.width, shape.width, "state plane")?;
    let state_len = checked_len(shape.value_heads, state_plane, "GDN state")?;
    let qkv_heads = shape
        .key_heads
        .checked_mul(2)
        .and_then(|n| n.checked_add(shape.value_heads))
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

fn decay_state_column(
    state: &mut [f32],
    head_start: usize,
    width: usize,
    value: usize,
    decay: f32,
) -> Result<()> {
    for key in 0..width {
        let index = head_start + key * width + value;
        state[index] = checked_mul(state[index], decay, "decayed state")?;
    }
    Ok(())
}

fn dot_state_column(
    state: &[f32],
    head_start: usize,
    width: usize,
    value: usize,
    vector: &[f32],
    reduction: Reduction,
) -> Result<f32> {
    match reduction {
        Reduction::OrderedF32 => {
            let mut sum = 0.0_f32;
            for (key, &right) in vector.iter().enumerate() {
                let left = state[head_start + key * width + value];
                sum = checked_add(
                    sum,
                    checked_mul(left, right, "ordered dot product")?,
                    "ordered dot sum",
                )?;
            }
            Ok(sum)
        }
        Reduction::WideF64 => {
            let mut sum = 0.0_f64;
            for (key, &right) in vector.iter().enumerate() {
                let left = state[head_start + key * width + value];
                checked_mul(left, right, "wide dot FP32 product")?;
                sum += f64::from(left) * f64::from(right);
                ensure!(sum.is_finite(), "wide dot sum overflows FP64");
            }
            checked_f32(sum as f32, "wide dot result")
        }
    }
}

fn update_state_column(
    state: &mut [f32],
    head_start: usize,
    width: usize,
    value: usize,
    key: &[f32],
    delta: f32,
) -> Result<()> {
    for (key_dimension, &key_value) in key.iter().enumerate() {
        let index = head_start + key_dimension * width + value;
        let update = checked_mul(key_value, delta, "key times delta")?;
        state[index] = checked_add(state[index], update, "updated recurrent state")?;
    }
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

    fn shape(rows: usize, key_heads: usize, value_heads: usize, width: usize) -> Shape {
        Shape {
            rows,
            key_heads,
            value_heads,
            width,
        }
    }

    fn bf16(value: f32) -> u16 {
        round_bf16(value)
    }

    #[test]
    fn width_one_two_token_recurrence_matches_hand_calculation() {
        let input = Input {
            q: &[1.0, 2.0],
            k: &[1.0, 2.0],
            qkv: &[
                bf16(0.0),
                bf16(0.0),
                bf16(3.0),
                bf16(0.0),
                bf16(0.0),
                bf16(4.0),
            ],
            beta: &[bf16(0.5), bf16(1.0)],
            decay: &[1.0, 0.5],
        };
        let result = run(&input, &[2.0], &shape(2, 1, 1, 1), Reduction::OrderedF32).unwrap();
        assert_eq!(result.state, [4.25]);
        assert_eq!(result.unrounded, [2.5, 8.5]);
        assert_eq!(result.output, [bf16(2.5), bf16(8.5)]);
    }

    #[test]
    fn asymmetric_width_two_state_uses_value_head_division_mapping() {
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let qkv = [bf16(0.0); 16];
        let input = Input {
            q: &q,
            k: &k,
            qkv: &qkv,
            beta: &[bf16(0.0); 4],
            decay: &[1.0; 4],
        };
        let initial_state = [
            10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0, 40.0, 41.0,
            42.0, 43.0,
        ];
        let result = run(
            &input,
            &initial_state,
            &shape(1, 2, 4, 2),
            Reduction::OrderedF32,
        )
        .unwrap();
        assert_eq!(
            result.unrounded,
            [10.0, 11.0, 20.0, 21.0, 32.0, 33.0, 42.0, 43.0]
        );
        assert_eq!(result.state, initial_state);
    }

    #[test]
    fn beta_zero_decay_one_preserves_and_beta_one_decay_zero_overwrites() {
        let preserve = Input {
            q: &[1.0],
            k: &[9.0],
            qkv: &[bf16(0.0), bf16(0.0), bf16(20.0)],
            beta: &[bf16(0.0)],
            decay: &[1.0],
        };
        let preserved = run(&preserve, &[7.0], &shape(1, 1, 1, 1), Reduction::OrderedF32).unwrap();
        assert_eq!(preserved.state, [7.0]);
        assert_eq!(preserved.unrounded, [7.0]);

        let overwrite = Input {
            q: &[3.0],
            k: &[4.0],
            qkv: &[bf16(0.0), bf16(0.0), bf16(5.0)],
            beta: &[bf16(1.0)],
            decay: &[0.0],
        };
        let overwritten = run(
            &overwrite,
            &[99.0],
            &shape(1, 1, 1, 1),
            Reduction::OrderedF32,
        )
        .unwrap();
        assert_eq!(overwritten.state, [20.0]);
        assert_eq!(overwritten.unrounded, [60.0]);
    }

    fn chunk<'a>(input: &Input<'a>, shape: &Shape, start: usize, end: usize) -> Input<'a> {
        let key_row = shape.key_heads * shape.width;
        let row_width = (2 * shape.key_heads + shape.value_heads) * shape.width;
        let gate_row = shape.value_heads;
        Input {
            q: &input.q[start * key_row..end * key_row],
            k: &input.k[start * key_row..end * key_row],
            qkv: &input.qkv[start * row_width..end * row_width],
            beta: &input.beta[start * gate_row..end * gate_row],
            decay: &input.decay[start * gate_row..end * gate_row],
        }
    }

    #[test]
    fn whole_and_chunked_sequences_match_for_both_reductions() {
        let full_shape = shape(3, 1, 2, 2);
        let q = [0.5, -1.0, 1.5, 0.25, -0.75, 2.0];
        let k = [1.0, 0.5, -0.5, 1.25, 2.0, -1.0];
        let qkv = [
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(1.0),
            bf16(2.0),
            bf16(-1.0),
            bf16(0.5),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(3.0),
            bf16(-2.0),
            bf16(0.25),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(0.0),
            bf16(-1.5),
            bf16(4.0),
            bf16(1.0),
            bf16(0.0),
        ];
        let beta = [
            bf16(0.25),
            bf16(0.75),
            bf16(0.5),
            bf16(1.0),
            bf16(0.0),
            bf16(0.375),
        ];
        let decay = [0.5, 1.0, 0.75, 0.25, 1.0, 0.625];
        let input = Input {
            q: &q,
            k: &k,
            qkv: &qkv,
            beta: &beta,
            decay: &decay,
        };
        for reduction in [Reduction::OrderedF32, Reduction::WideF64] {
            let whole = run(
                &input,
                &[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 1.0, 0.0],
                &full_shape,
                reduction,
            )
            .unwrap();
            let first_input = chunk(&input, &full_shape, 0, 2);
            let first = run(
                &first_input,
                &[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 1.0, 0.0],
                &shape(2, 1, 2, 2),
                reduction,
            )
            .unwrap();
            let second_input = chunk(&input, &full_shape, 2, 3);
            let second = run(&second_input, &first.state, &shape(1, 1, 2, 2), reduction).unwrap();
            assert_eq!([first.output, second.output].concat(), whole.output);
            assert_eq!(
                [first.unrounded, second.unrounded].concat(),
                whole.unrounded
            );
            assert_eq!(second.state, whole.state);
        }
    }

    #[test]
    fn rejects_bad_shapes_extents_domains_nonfinite_values_and_overflow() {
        let valid = Input {
            q: &[1.0],
            k: &[1.0],
            qkv: &[bf16(0.0), bf16(0.0), bf16(1.0)],
            beta: &[bf16(0.5)],
            decay: &[1.0],
        };
        assert!(run(&valid, &[0.0], &shape(1, 1, 1, 3), Reduction::OrderedF32).is_err());
        assert!(run(&valid, &[0.0], &shape(1, 2, 1, 1), Reduction::OrderedF32).is_err());
        assert!(run(&valid, &[0.0], &shape(0, 1, 1, 1), Reduction::OrderedF32).is_err());

        let wrong_extent = Input { q: &[], ..valid };
        assert!(
            run(
                &wrong_extent,
                &[0.0],
                &shape(1, 1, 1, 1),
                Reduction::OrderedF32
            )
            .is_err()
        );
        let bad_beta = [bf16(1.5)];
        let wrong_domain = Input {
            beta: &bad_beta,
            ..valid
        };
        assert!(
            run(
                &wrong_domain,
                &[0.0],
                &shape(1, 1, 1, 1),
                Reduction::OrderedF32
            )
            .is_err()
        );
        let nonfinite_q = [f32::NAN];
        let nonfinite = Input {
            q: &nonfinite_q,
            ..valid
        };
        assert!(
            run(
                &nonfinite,
                &[0.0],
                &shape(1, 1, 1, 1),
                Reduction::OrderedF32
            )
            .is_err()
        );

        let overflow = Input {
            q: &[1.0],
            k: &[f32::MAX],
            qkv: &[bf16(0.0), bf16(0.0), bf16(3.0)],
            beta: &[bf16(1.0)],
            decay: &[0.0],
        };
        for reduction in [Reduction::OrderedF32, Reduction::WideF64] {
            assert!(run(&overflow, &[0.0], &shape(1, 1, 1, 1), reduction).is_err());
        }

        let overflowing_dot = Input {
            q: &[2.0],
            k: &[0.0],
            qkv: &[bf16(0.0), bf16(0.0), bf16(0.0)],
            beta: &[bf16(0.0)],
            decay: &[1.0],
        };
        for reduction in [Reduction::OrderedF32, Reduction::WideF64] {
            assert!(run(&overflowing_dot, &[f32::MAX], &shape(1, 1, 1, 1), reduction).is_err());
        }
    }
}
