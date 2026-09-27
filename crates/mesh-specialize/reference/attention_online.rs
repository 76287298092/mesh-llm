//! Independent FP64 CPU oracle for the tiled FP32 online-attention candidate.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

const MAX_ELEMENTS: usize = 67_108_864;

/// Existing causal-attention component allowance, unchanged for this candidate.
pub const COMPONENT_ABS_BUDGET: f32 = 5.0e-6;
/// Existing causal-attention component allowance, unchanged for this candidate.
pub const COMPONENT_REL_BUDGET: f32 = 3.0e-5;
/// Maximum absolute drift allowed when query rows are submitted in chunks.
pub const CHUNK_EQUIVALENCE_ABS_BUDGET: f32 = 1.0e-6;

#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub rows: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub past: usize,
    pub capacity: usize,
    pub scale: f32,
}

#[derive(Debug, PartialEq)]
pub struct Attention {
    /// BF16 output in `[rows, query_heads, width]` order.
    pub output: Vec<u16>,
    /// FP32 output before BF16 rounding.
    pub unrounded: Vec<f32>,
    /// Maximum absolute causally visible V value for each output channel.
    pub value_bounds: Vec<f32>,
}

struct Lengths {
    output: usize,
    cache: usize,
    initialized_cache: usize,
    kv_row: usize,
    heads_per_kv: usize,
}

struct AttentionInputs<'a> {
    q: &'a [u16],
    cache_k: &'a [u16],
    cache_v: &'a [u16],
    shape: &'a Shape,
    lengths: &'a Lengths,
}

/// Compute causal grouped-query attention from logical BF16 Q/K/V tensors.
///
/// Q is compact `[rows, query_heads, width]`. K/V are full-capacity,
/// token-major caches `[capacity, kv_heads, width]`. Query row `r` attends to
/// keys `0..=past + r`. This oracle uses FP64 dot products, exponentials, and
/// weighted-value sums, independent of the tiled FP32 kernel's operation order.
pub fn run(q: &[u16], cache_k: &[u16], cache_v: &[u16], shape: &Shape) -> Result<Attention> {
    let lengths = validate_shape(shape)?;
    ensure!(
        q.len() == lengths.output,
        "online attention Q extent mismatch"
    );
    ensure!(
        cache_k.len() == lengths.cache,
        "online attention K cache extent mismatch"
    );
    ensure!(
        cache_v.len() == lengths.cache,
        "online attention V cache extent mismatch"
    );
    ensure!(
        q.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "online attention Q values must be finite BF16 values"
    );
    validate_finite_prefix(cache_k, lengths.initialized_cache, "K")?;
    validate_finite_prefix(cache_v, lengths.initialized_cache, "V")?;

    let mut result = Attention {
        output: Vec::with_capacity(lengths.output),
        unrounded: Vec::with_capacity(lengths.output),
        value_bounds: Vec::with_capacity(lengths.output),
    };
    let input = AttentionInputs {
        q,
        cache_k,
        cache_v,
        shape,
        lengths: &lengths,
    };
    for row in 0..shape.rows {
        for query_head in 0..shape.query_heads {
            attend_head(&input, row, query_head, &mut result)?;
        }
    }
    debug_assert_eq!(result.output.len(), lengths.output);
    debug_assert_eq!(result.unrounded.len(), lengths.output);
    debug_assert_eq!(result.value_bounds.len(), lengths.output);
    Ok(result)
}

fn validate_shape(shape: &Shape) -> Result<Lengths> {
    ensure!(
        (1..=2048).contains(&shape.rows),
        "invalid online attention row count"
    );
    ensure!(
        (1..=128).contains(&shape.query_heads),
        "invalid online attention query head count"
    );
    ensure!(
        (1..=128).contains(&shape.kv_heads),
        "invalid online attention KV head count"
    );
    ensure!(
        shape.query_heads.is_multiple_of(shape.kv_heads),
        "online attention query heads must divide evenly across KV heads"
    );
    ensure!(
        (2..=256).contains(&shape.width),
        "invalid online attention head width"
    );
    ensure!(
        (1..=262_144).contains(&shape.capacity),
        "invalid online attention cache capacity"
    );
    ensure!(
        shape.scale.is_finite() && shape.scale > 0.0,
        "invalid online attention scale"
    );
    let available_end = shape
        .past
        .checked_add(shape.rows)
        .context("online attention past plus rows overflows usize")?;
    ensure!(
        available_end <= shape.capacity,
        "online attention chunk exceeds cache capacity"
    );

    let kv_row = checked_product(&[shape.kv_heads, shape.width], "online attention KV row")?;
    let output = checked_product(
        &[shape.rows, shape.query_heads, shape.width],
        "online attention output",
    )?;
    let cache = checked_product(
        &[shape.capacity, shape.kv_heads, shape.width],
        "online attention cache",
    )?;
    let initialized_cache = checked_product(
        &[available_end, kv_row],
        "initialized online attention cache",
    )?;
    ensure!(
        output <= MAX_ELEMENTS,
        "online attention output exceeds element limit"
    );
    ensure!(
        cache <= MAX_ELEMENTS,
        "online attention cache exceeds element limit"
    );
    Ok(Lengths {
        output,
        cache,
        initialized_cache,
        kv_row,
        heads_per_kv: shape.query_heads / shape.kv_heads,
    })
}

fn attend_head(
    input: &AttentionInputs<'_>,
    row: usize,
    query_head: usize,
    result: &mut Attention,
) -> Result<()> {
    let kv_head = query_head / input.lengths.heads_per_kv;
    let q_start = (row * input.shape.query_heads + query_head) * input.shape.width;
    let query = &input.q[q_start..q_start + input.shape.width];
    let key_count = input.shape.past + row + 1;
    let mut logits = Vec::with_capacity(key_count);
    for key_position in 0..key_count {
        logits.push(logical_score(input, query, kv_head, key_position)?);
    }
    let probabilities = stable_probabilities(&logits)?;
    append_weighted_channels(input, kv_head, &probabilities, result)
}

fn logical_score(
    input: &AttentionInputs<'_>,
    query: &[u16],
    kv_head: usize,
    key_position: usize,
) -> Result<f64> {
    let mut dot = 0.0_f64;
    for (channel, &query_bits) in query.iter().enumerate() {
        let cache_index =
            (key_position * input.shape.kv_heads + kv_head) * input.shape.width + channel;
        let query_value = f64::from(bf16_to_f32(query_bits));
        let key_value = f64::from(bf16_to_f32(input.cache_k[cache_index]));
        dot += query_value * key_value;
    }
    let scaled = dot * f64::from(input.shape.scale);
    ensure!(
        scaled.is_finite(),
        "online attention logit is nonfinite at KV row {} of {} elements",
        key_position,
        input.lengths.kv_row
    );
    Ok(scaled)
}

fn stable_probabilities(logits: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !logits.is_empty(),
        "online attention query has no valid keys"
    );
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(
        maximum.is_finite(),
        "online attention maximum logit is nonfinite"
    );
    let exponentials: Vec<f64> = logits.iter().map(|score| (score - maximum).exp()).collect();
    let denominator: f64 = exponentials.iter().sum();
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "online attention softmax denominator is invalid"
    );
    Ok(exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect())
}

fn append_weighted_channels(
    input: &AttentionInputs<'_>,
    kv_head: usize,
    probabilities: &[f64],
    result: &mut Attention,
) -> Result<()> {
    for channel in 0..input.shape.width {
        let mut weighted_sum = 0.0_f64;
        let mut value_bound = 0.0_f32;
        for (key_position, &probability) in probabilities.iter().enumerate() {
            let cache_index =
                (key_position * input.shape.kv_heads + kv_head) * input.shape.width + channel;
            let value = bf16_to_f32(input.cache_v[cache_index]);
            value_bound = value_bound.max(value.abs());
            weighted_sum += probability * f64::from(value);
        }
        ensure!(
            weighted_sum.is_finite(),
            "online attention weighted value is nonfinite"
        );
        let unrounded = weighted_sum as f32;
        ensure!(
            unrounded.is_finite(),
            "online attention output overflows FP32"
        );
        let rounded = round_bf16(unrounded);
        ensure!(
            bf16_to_f32(rounded).is_finite(),
            "online attention output overflows BF16"
        );
        result.unrounded.push(unrounded);
        result.output.push(rounded);
        result.value_bounds.push(value_bound);
    }
    debug_assert!(input.lengths.output >= result.unrounded.len());
    Ok(())
}

fn validate_finite_prefix(values: &[u16], prefix_len: usize, label: &str) -> Result<()> {
    ensure!(
        values[..prefix_len]
            .iter()
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "initialized online attention {label} cache prefix must be finite BF16"
    );
    Ok(())
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .with_context(|| format!("{label} extent overflows usize"))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CHUNK_EQUIVALENCE_ABS_BUDGET, COMPONENT_ABS_BUDGET, COMPONENT_REL_BUDGET, Shape, run,
        stable_probabilities,
    };
    use crate::entry_reference::{bf16_to_f32, round_bf16};

    const TILE_KEYS: usize = 8;

    fn shape(
        rows: usize,
        query_heads: usize,
        kv_heads: usize,
        width: usize,
        past: usize,
        capacity: usize,
        scale: f32,
    ) -> Shape {
        Shape {
            rows,
            query_heads,
            kv_heads,
            width,
            past,
            capacity,
            scale,
        }
    }

    fn words(values: &[f32]) -> Vec<u16> {
        values.iter().copied().map(round_bf16).collect()
    }

    fn poisoned_cache(capacity: usize, heads: usize, width: usize, values: &[f32]) -> Vec<u16> {
        let mut result = words(values);
        result.resize(capacity * heads * width, 0x7fc0);
        result
    }

    fn floats(values: &[u16]) -> Vec<f32> {
        values.iter().copied().map(bf16_to_f32).collect()
    }

    fn candidate_online_f32(q: &[u16], k: &[u16], v: &[u16], shape: &Shape) -> Vec<f32> {
        let mut output = Vec::with_capacity(shape.rows * shape.query_heads * shape.width);
        let heads_per_kv = shape.query_heads / shape.kv_heads;
        for row in 0..shape.rows {
            for query_head in 0..shape.query_heads {
                let kv_head = query_head / heads_per_kv;
                let query_start = (row * shape.query_heads + query_head) * shape.width;
                let query = &q[query_start..query_start + shape.width];
                let visible = shape.past + row + 1;
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0_f32;
                let mut accumulator = vec![0.0_f32; shape.width];
                let mut tile_start = 0;
                while tile_start < visible {
                    let mut scores = [f32::NEG_INFINITY; TILE_KEYS];
                    let mut tile_maximum = f32::NEG_INFINITY;
                    for (tile_key, score) in scores.iter_mut().enumerate() {
                        let key_position = tile_start + tile_key;
                        if key_position < visible {
                            let mut dot = 0.0_f32;
                            for (channel, &query_bits) in query.iter().enumerate() {
                                let cache_index = (key_position * shape.kv_heads + kv_head)
                                    * shape.width
                                    + channel;
                                dot += bf16_to_f32(query_bits) * bf16_to_f32(k[cache_index]);
                            }
                            *score = dot * shape.scale;
                            tile_maximum = tile_maximum.max(*score);
                        }
                    }
                    let next_maximum = maximum.max(tile_maximum);
                    let alpha = if denominator == 0.0 {
                        0.0
                    } else {
                        (maximum - next_maximum).exp()
                    };
                    let mut weights = [0.0_f32; TILE_KEYS];
                    let mut tile_denominator = 0.0_f32;
                    for (tile_key, &score) in scores.iter().enumerate() {
                        if tile_start + tile_key < visible {
                            let weight = (score - next_maximum).exp();
                            weights[tile_key] = weight;
                            tile_denominator += weight;
                        }
                    }
                    denominator = denominator * alpha + tile_denominator;
                    for (channel, value) in accumulator.iter_mut().enumerate().take(shape.width) {
                        let mut weighted_tile = 0.0_f32;
                        for (tile_key, &weight) in weights.iter().enumerate() {
                            if weight != 0.0 {
                                let key_position = tile_start + tile_key;
                                let cache_index = (key_position * shape.kv_heads + kv_head)
                                    * shape.width
                                    + channel;
                                weighted_tile += weight * bf16_to_f32(v[cache_index]);
                            }
                        }
                        *value = *value * alpha + weighted_tile;
                    }
                    maximum = next_maximum;
                    tile_start += TILE_KEYS;
                }
                output.extend(accumulator.into_iter().map(|value| value / denominator));
            }
        }
        output
    }

    fn assert_profile_budget(candidate: &[f32], oracle: &[f32], bounds: &[f32]) {
        assert_eq!(candidate.len(), oracle.len());
        assert_eq!(candidate.len(), bounds.len());
        for ((&actual, &expected), &bound) in candidate.iter().zip(oracle).zip(bounds) {
            let allowance = COMPONENT_ABS_BUDGET + COMPONENT_REL_BUDGET * bound;
            assert!(
                (actual - expected).abs() <= allowance,
                "candidate {actual} differs from oracle {expected} by more than {allowance}"
            );
        }
    }

    #[test]
    fn past_zero_groups_query_heads_and_masks_later_rows() {
        let shape = shape(2, 4, 2, 2, 0, 5, 1.0);
        let q = vec![round_bf16(0.0); 16];
        let k = poisoned_cache(5, 2, 2, &[0.0; 8]);
        let v = poisoned_cache(5, 2, 2, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(
            floats(&result.output),
            [
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.5, 11.0, 5.5, 11.0, 16.5, 22.0, 16.5,
                22.0,
            ]
        );
    }

    #[test]
    fn nonzero_past_and_odd_cache_capacity_ignore_poisoned_tail() {
        let shape = shape(2, 1, 1, 2, 3, 7, 1.0);
        let q = vec![round_bf16(0.0); 4];
        let k = poisoned_cache(7, 1, 2, &[0.0; 10]);
        let v = poisoned_cache(
            7,
            1,
            2,
            &[0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0],
        );
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(result.unrounded, [3.0, 5.0, 4.0, 6.0]);
    }

    #[test]
    fn extreme_positive_and_negative_logits_remain_finite() {
        let shape = shape(1, 1, 1, 2, 1, 3, 1.0);
        let q = words(&[1_000.0, 0.0]);
        let k = poisoned_cache(3, 1, 2, &[1_000.0, 0.0, -1_000.0, 0.0]);
        let v = poisoned_cache(3, 1, 2, &[8.0, -4.0, -8.0, 4.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(result.unrounded, [8.0, -4.0]);
        assert!(result.unrounded.iter().all(|value| value.is_finite()));
        let candidate = candidate_online_f32(&q, &k, &v, &shape);
        assert_profile_budget(&candidate, &result.unrounded, &result.value_bounds);
    }

    #[test]
    fn online_fp32_fixture_stays_inside_existing_component_budget() {
        let rows = 3;
        let q = words(
            &(0..rows * 6 * 13)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
                .collect::<Vec<_>>(),
        );
        let k = poisoned_cache(
            15,
            2,
            13,
            &(0..13 * 2 * 13)
                .map(|index| ((index % 23) as f32 - 11.0) * 0.015625)
                .collect::<Vec<_>>(),
        );
        let v = poisoned_cache(
            15,
            2,
            13,
            &(0..13 * 2 * 13)
                .map(|index| ((index % 19) as f32 - 9.0) * 0.125)
                .collect::<Vec<_>>(),
        );
        let shape = shape(rows, 6, 2, 13, 10, 15, 0.25);
        let oracle = run(&q, &k, &v, &shape).unwrap();
        let candidate = candidate_online_f32(&q, &k, &v, &shape);
        assert_profile_budget(&candidate, &oracle.unrounded, &oracle.value_bounds);
    }

    #[test]
    fn query_chunking_matches_within_declared_absolute_budget() {
        let rows = 4;
        let query_heads = 4;
        let kv_heads = 2;
        let width = 8;
        let past = 7;
        let capacity = 13;
        let q = words(
            &(0..rows * query_heads * width)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.0625)
                .collect::<Vec<_>>(),
        );
        let k = poisoned_cache(
            capacity,
            kv_heads,
            width,
            &(0..(past + rows) * kv_heads * width)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.125)
                .collect::<Vec<_>>(),
        );
        let v = poisoned_cache(
            capacity,
            kv_heads,
            width,
            &(0..(past + rows) * kv_heads * width)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.25)
                .collect::<Vec<_>>(),
        );
        let whole_shape = shape(rows, query_heads, kv_heads, width, past, capacity, 0.5);
        let whole = run(&q, &k, &v, &whole_shape).unwrap();
        let whole_candidate = candidate_online_f32(&q, &k, &v, &whole_shape);
        let partitions = [1, 2, 1];
        let mut offset = 0;
        let mut chunk_outputs = Vec::new();
        let mut chunk_candidate_outputs = Vec::new();
        for chunk_rows in partitions {
            let chunk_shape = shape(
                chunk_rows,
                query_heads,
                kv_heads,
                width,
                past + offset,
                capacity,
                whole_shape.scale,
            );
            let query_start = offset * query_heads * width;
            let query_end = query_start + chunk_rows * query_heads * width;
            let chunk_q = &q[query_start..query_end];
            let chunk = run(chunk_q, &k, &v, &chunk_shape).unwrap();
            chunk_candidate_outputs.extend(candidate_online_f32(chunk_q, &k, &v, &chunk_shape));
            chunk_outputs.extend(chunk.unrounded);
            offset += chunk_rows;
        }
        assert_eq!(offset, rows);
        for (&whole_value, &chunk_value) in whole.unrounded.iter().zip(&chunk_outputs) {
            assert!((whole_value - chunk_value).abs() <= CHUNK_EQUIVALENCE_ABS_BUDGET);
        }
        for (&whole_value, &chunk_value) in whole_candidate.iter().zip(&chunk_candidate_outputs) {
            assert!((whole_value - chunk_value).abs() <= CHUNK_EQUIVALENCE_ABS_BUDGET);
        }
        assert_eq!(
            whole.output,
            chunk_outputs
                .iter()
                .copied()
                .map(round_bf16)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_zero_extents_and_an_all_masked_softmax() {
        for invalid in [
            shape(0, 1, 1, 2, 0, 1, 1.0),
            shape(1, 0, 1, 2, 0, 1, 1.0),
            shape(1, 1, 0, 2, 0, 1, 1.0),
            shape(1, 1, 1, 0, 0, 1, 1.0),
            shape(1, 1, 1, 2, 0, 0, 1.0),
        ] {
            assert!(run(&[], &[], &[], &invalid).is_err());
        }
        assert!(stable_probabilities(&[]).is_err());
        assert!(stable_probabilities(&[f64::NEG_INFINITY; 2]).is_err());
    }
}
