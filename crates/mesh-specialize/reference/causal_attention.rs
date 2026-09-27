//! Independent high-precision CPU reference for causal grouped-query attention.

use anyhow::{Context, Result, ensure};

use crate::entry_reference::{bf16_to_f32, round_bf16};

const MAX_ELEMENTS: usize = 67_108_864;

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
    /// FP32 attention output before BF16 rounding.
    pub unrounded: Vec<f32>,
    /// Per-output maximum absolute legal V value, for comparison tolerances.
    pub value_bounds: Vec<f32>,
}

struct Lengths {
    output: usize,
    query: usize,
    cache: usize,
    initialized_cache: usize,
    append: usize,
    prefix: usize,
    kv_row: usize,
    ratio: usize,
}

/// Run causal grouped-query attention over a cache containing the current chunk.
///
/// Q is compact `[rows, query_heads, width]`. K/V are full-capacity,
/// time-major caches `[capacity, kv_heads, width]`. Query row `r` attends only
/// to keys `0..=past + r`, even though later chunk rows are already present.
/// Logits, probabilities, and weighted values use FP64 arithmetic; only the
/// final output is converted to FP32 and rounded to BF16.
pub fn run(q: &[u16], cache_k: &[u16], cache_v: &[u16], shape: &Shape) -> Result<Attention> {
    let lengths = validate_shape(shape)?;
    ensure!(q.len() == lengths.query, "attention Q extent mismatch");
    ensure!(
        cache_k.len() == lengths.cache,
        "attention K cache extent mismatch"
    );
    ensure!(
        cache_v.len() == lengths.cache,
        "attention V cache extent mismatch"
    );
    ensure!(
        q.iter().all(|&bits| bf16_to_f32(bits).is_finite()),
        "attention Q values must be finite BF16 values"
    );
    validate_finite_prefix(cache_k, lengths.initialized_cache, "K")?;
    validate_finite_prefix(cache_v, lengths.initialized_cache, "V")?;

    let mut result = Attention {
        output: Vec::with_capacity(lengths.output),
        unrounded: Vec::with_capacity(lengths.output),
        value_bounds: Vec::with_capacity(lengths.output),
    };
    for row in 0..shape.rows {
        for query_head in 0..shape.query_heads {
            attend_head(
                q,
                cache_k,
                cache_v,
                shape,
                &lengths,
                [row, query_head],
                &mut result,
            )?;
        }
    }
    debug_assert_eq!(result.output.len(), lengths.output);
    debug_assert_eq!(result.unrounded.len(), lengths.output);
    debug_assert_eq!(result.value_bounds.len(), lengths.output);
    Ok(result)
}

/// Append a chunk of K/V rows at `past`, preserving earlier rows and cache tail.
///
/// The incoming tensors are `[rows, kv_heads, width]`; only cache rows
/// `[past, past + rows)` are written. The initialized prefix is checked for
/// finite BF16 values. Uninitialized cache-tail words are neither read nor
/// changed, so callers may use poison values there during validation.
pub fn append(
    k: &[u16],
    v: &[u16],
    cache_k: &mut [u16],
    cache_v: &mut [u16],
    shape: &Shape,
) -> Result<()> {
    let lengths = validate_shape(shape)?;
    ensure!(
        k.len() == lengths.append,
        "attention appended K extent mismatch"
    );
    ensure!(
        v.len() == lengths.append,
        "attention appended V extent mismatch"
    );
    ensure!(
        cache_k.len() == lengths.cache,
        "attention K cache extent mismatch"
    );
    ensure!(
        cache_v.len() == lengths.cache,
        "attention V cache extent mismatch"
    );
    ensure!(
        k.iter().chain(v).all(|&bits| bf16_to_f32(bits).is_finite()),
        "appended attention K/V values must be finite BF16 values"
    );
    validate_finite_prefix(cache_k, lengths.prefix, "K")?;
    validate_finite_prefix(cache_v, lengths.prefix, "V")?;

    let append_start = lengths.prefix;
    let append_end = append_start
        .checked_add(lengths.append)
        .context("attention append end overflows usize")?;
    cache_k[append_start..append_end].copy_from_slice(k);
    cache_v[append_start..append_end].copy_from_slice(v);
    Ok(())
}

fn validate_shape(shape: &Shape) -> Result<Lengths> {
    ensure!(
        (1..=2048).contains(&shape.rows),
        "invalid attention row count"
    );
    ensure!(
        (1..=128).contains(&shape.query_heads),
        "invalid attention query head count"
    );
    ensure!(
        (1..=128).contains(&shape.kv_heads),
        "invalid attention KV head count"
    );
    ensure!(
        shape.query_heads.is_multiple_of(shape.kv_heads),
        "query heads must be divisible by KV heads"
    );
    ensure!(
        (2..=256).contains(&shape.width),
        "invalid attention head width"
    );
    ensure!(
        (1..=262_144).contains(&shape.capacity),
        "invalid attention cache capacity"
    );
    ensure!(
        shape.scale.is_finite() && shape.scale > 0.0,
        "invalid attention scale"
    );
    let available_end = shape
        .past
        .checked_add(shape.rows)
        .context("attention past plus rows overflows usize")?;
    ensure!(
        available_end <= shape.capacity,
        "attention chunk exceeds cache capacity"
    );

    let kv_row = checked_product(&[shape.kv_heads, shape.width], "attention KV row")?;
    let output = checked_product(
        &[shape.rows, shape.query_heads, shape.width],
        "attention output",
    )?;
    let query = checked_product(&[shape.rows, shape.query_heads, shape.width], "attention Q")?;
    let cache = checked_product(
        &[shape.capacity, shape.kv_heads, shape.width],
        "attention cache",
    )?;
    let initialized_cache =
        checked_product(&[available_end, kv_row], "initialized attention cache")?;
    let append = checked_product(&[shape.rows, kv_row], "attention appended rows")?;
    let prefix = checked_product(&[shape.past, kv_row], "attention cache prefix")?;
    ensure!(
        output <= MAX_ELEMENTS,
        "attention output exceeds element limit"
    );
    ensure!(query <= MAX_ELEMENTS, "attention Q exceeds element limit");
    ensure!(
        cache <= MAX_ELEMENTS,
        "attention cache exceeds element limit"
    );
    Ok(Lengths {
        output,
        query,
        cache,
        initialized_cache,
        append,
        prefix,
        kv_row,
        ratio: shape.query_heads / shape.kv_heads,
    })
}

fn attend_head(
    q: &[u16],
    cache_k: &[u16],
    cache_v: &[u16],
    shape: &Shape,
    lengths: &Lengths,
    coordinates: [usize; 2],
    result: &mut Attention,
) -> Result<()> {
    let [row, query_head] = coordinates;
    let kv_head = query_head / lengths.ratio;
    let q_start = (row * shape.query_heads + query_head) * shape.width;
    let query = &q[q_start..q_start + shape.width];
    let key_count = shape.past + row + 1;
    let mut scores = Vec::with_capacity(key_count);
    for key_position in 0..key_count {
        scores.push(score(
            query,
            cache_k,
            shape,
            lengths,
            kv_head,
            key_position,
        )?);
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(maximum.is_finite(), "attention maximum logit is nonfinite");
    let exponentials: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let denominator: f64 = exponentials.iter().sum();
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "attention softmax denominator is invalid"
    );
    let probabilities: Vec<f64> = exponentials
        .iter()
        .map(|value| value / denominator)
        .collect();
    for column in 0..shape.width {
        append_weighted_value(
            cache_v,
            lengths,
            kv_head,
            column,
            &probabilities,
            shape,
            result,
        )?;
    }
    Ok(())
}

fn score(
    query: &[u16],
    cache_k: &[u16],
    shape: &Shape,
    lengths: &Lengths,
    kv_head: usize,
    key_position: usize,
) -> Result<f64> {
    let key_start = key_position * lengths.kv_row + kv_head * shape.width;
    let mut dot = 0.0_f64;
    for (query_bits, key_bits) in query
        .iter()
        .zip(&cache_k[key_start..key_start + shape.width])
    {
        let product = f64::from(bf16_to_f32(*query_bits)) * f64::from(bf16_to_f32(*key_bits));
        dot += product;
    }
    let scaled = dot * f64::from(shape.scale);
    ensure!(scaled.is_finite(), "attention logit is nonfinite");
    Ok(scaled)
}

fn append_weighted_value(
    cache_v: &[u16],
    lengths: &Lengths,
    kv_head: usize,
    column: usize,
    probabilities: &[f64],
    shape: &Shape,
    result: &mut Attention,
) -> Result<()> {
    let mut weighted_sum = 0.0_f64;
    let mut value_bound = 0.0_f32;
    for (key_position, &probability) in probabilities.iter().enumerate() {
        let index = key_position * lengths.kv_row + kv_head * shape.width + column;
        let value = bf16_to_f32(cache_v[index]);
        value_bound = value_bound.max(value.abs());
        weighted_sum += probability * f64::from(value);
    }
    ensure!(
        weighted_sum.is_finite(),
        "attention weighted value is nonfinite"
    );
    let unrounded = weighted_sum as f32;
    ensure!(unrounded.is_finite(), "attention output overflows FP32");
    let rounded = round_bf16(unrounded);
    ensure!(
        bf16_to_f32(rounded).is_finite(),
        "attention output overflows BF16"
    );
    result.unrounded.push(unrounded);
    result.output.push(rounded);
    result.value_bounds.push(value_bound);
    Ok(())
}

fn validate_finite_prefix(values: &[u16], prefix_len: usize, label: &str) -> Result<()> {
    ensure!(
        values[..prefix_len]
            .iter()
            .all(|&bits| bf16_to_f32(bits).is_finite()),
        "initialized attention {label} cache prefix must be finite BF16"
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
    use super::{Shape, append, run};
    use crate::entry_reference::{bf16_to_f32, round_bf16};

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

    fn cache(capacity: usize, heads: usize, width: usize, values: &[f32]) -> Vec<u16> {
        let mut result = values.iter().copied().map(round_bf16).collect::<Vec<_>>();
        result.resize(capacity * heads * width, 0);
        result
    }

    fn floats(words: &[u16]) -> Vec<f32> {
        words.iter().copied().map(bf16_to_f32).collect()
    }

    #[test]
    fn zero_query_gives_hand_computed_uniform_mean_and_bounds() {
        let shape = shape(1, 1, 1, 2, 1, 2, 1.0);
        let q = [0, 0];
        let k = cache(2, 1, 2, &[1.0, 0.0, 0.0, 1.0]);
        let v = cache(2, 1, 2, &[2.0, -4.0, 4.0, 2.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(result.unrounded, [3.0, -1.0]);
        assert_eq!(floats(&result.output), [3.0, -1.0]);
        assert_eq!(result.value_bounds, [4.0, 4.0]);
    }

    #[test]
    fn grouped_query_heads_map_to_their_kv_head() {
        let shape = shape(1, 4, 2, 2, 0, 1, 1.0);
        let q = [0; 8];
        let k = cache(1, 2, 2, &[0.0; 4]);
        let v = cache(1, 2, 2, &[1.0, 2.0, 10.0, 20.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(
            result.unrounded,
            [1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0]
        );
    }

    #[test]
    fn each_query_row_ignores_later_chunk_rows() {
        let shape = shape(2, 1, 1, 2, 0, 2, 1.0);
        let q = [0; 4];
        let k = cache(2, 1, 2, &[0.0; 4]);
        let v = cache(2, 1, 2, &[1.0, 2.0, 1_048_576.0, -1_048_576.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(&result.unrounded[..2], &[1.0, 2.0]);
        assert_eq!(result.unrounded[2], 524_288.5);
        assert_eq!(result.unrounded[3], -524_287.0);
    }

    #[test]
    fn append_preserves_prefix_and_poisoned_tail_and_run_ignores_tail() {
        let shape = shape(2, 1, 1, 2, 1, 5, 1.0);
        let mut cache_k = vec![round_bf16(0.5), round_bf16(1.0)];
        let mut cache_v = vec![round_bf16(5.0), round_bf16(6.0)];
        let poison = 0x7fc0;
        cache_k.extend([poison; 8]);
        cache_v.extend([poison; 8]);
        let original_prefix_k = cache_k[..2].to_vec();
        let original_prefix_v = cache_v[..2].to_vec();
        let incoming_k = [
            round_bf16(2.0),
            round_bf16(3.0),
            round_bf16(4.0),
            round_bf16(5.0),
        ];
        let incoming_v = [
            round_bf16(7.0),
            round_bf16(8.0),
            round_bf16(9.0),
            round_bf16(10.0),
        ];
        append(&incoming_k, &incoming_v, &mut cache_k, &mut cache_v, &shape).unwrap();
        assert_eq!(&cache_k[..2], original_prefix_k);
        assert_eq!(&cache_v[..2], original_prefix_v);
        assert_eq!(&cache_k[2..6], incoming_k);
        assert_eq!(&cache_v[2..6], incoming_v);
        assert_eq!(&cache_k[6..], &[poison; 4]);
        assert_eq!(&cache_v[6..], &[poison; 4]);
        let result = run(&[0; 4], &cache_k, &cache_v, &shape).unwrap();
        assert_eq!(result.output.len(), 4);
    }

    #[test]
    fn extreme_finite_logits_use_stable_softmax() {
        let shape = shape(1, 1, 1, 2, 1, 2, 1.0);
        let q = [round_bf16(10_000.0), 0];
        let k = cache(2, 1, 2, &[10_000.0, 0.0, -10_000.0, 0.0]);
        let v = cache(2, 1, 2, &[3.0, -5.0, -7.0, 9.0]);
        let result = run(&q, &k, &v, &shape).unwrap();
        assert_eq!(result.unrounded, [3.0, -5.0]);
        assert!(result.unrounded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_invalid_shape_inputs_and_initialized_cache_values_only() {
        let good = shape(1, 1, 1, 2, 0, 2, 1.0);
        let q = [0, 0];
        let k = cache(2, 1, 2, &[1.0, 0.0]);
        let mut v = cache(2, 1, 2, &[1.0, -1.0]);
        v[2] = 0x7fc0;
        assert!(run(&q, &k, &v, &good).is_ok());
        assert!(run(&q, &k, &v, &shape(1, 1, 1, 2, 0, 2, 0.0)).is_err());
        assert!(run(&[0], &k, &v, &good).is_err());
        assert!(run(&[0x7f80, 0], &k, &v, &good).is_err());
        assert!(run(&q, &[0x7f80, 0, 0, 0], &v, &good).is_err());
        assert!(run(&q, &k, &[0x7fc0, 0, 0, 0], &good).is_err());
        assert!(run(&q, &k, &v, &shape(1, 3, 2, 2, 0, 2, 1.0)).is_err());
        assert!(run(&q, &k, &v, &shape(1, 1, 1, 2, 2, 2, 1.0)).is_err());
        let mut empty_cache_k = cache(2, 1, 2, &[0.0; 2]);
        let mut empty_cache_v = cache(2, 1, 2, &[0.0; 2]);
        assert!(append(&[], &[], &mut empty_cache_k, &mut empty_cache_v, &good).is_err());
    }

    #[test]
    fn zero_result_preserves_nonnegative_rounding_for_signed_values() {
        let shape = shape(1, 1, 1, 2, 0, 1, 1.0);
        let result = run(&[0, 0], &[0, 0], &[0x8000, 0x8000], &shape).unwrap();
        assert_eq!(result.output, [0, 0]);
        assert_eq!(result.unrounded, [0.0, 0.0]);
    }
}
