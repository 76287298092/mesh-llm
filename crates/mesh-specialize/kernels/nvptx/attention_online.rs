//! Tiled FP32 online-softmax attention for BF16 grouped-query cache data.

use core::arch::asm;

const BLOCK_THREADS: u32 = 256;
const TILE_KEYS: u32 = 8;
const CONTROL_ALPHA: u32 = 0;
const CONTROL_NORMALIZER: u32 = 1;
const CONTROL_WEIGHT_BASE: u32 = 2;

#[derive(Clone, Copy)]
struct SharedBuffers {
    keys: u32,
    values: u32,
    scores: u32,
    controls: u32,
}

#[derive(Clone, Copy)]
struct CacheView {
    keys: *const u16,
    values: *const u16,
    row_width: usize,
    head_offset: usize,
}

#[derive(Clone, Copy)]
struct KeyTile {
    width: u32,
    start: u32,
    visible: u32,
}

struct OnlineState {
    maximum: f32,
    normalizer: f32,
}

#[inline(always)]
fn block_and_thread() -> (u32, u32) {
    let block: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's one-dimensional coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {block}, %ctaid.x;",
            "mov.u32 {thread}, %tid.x;",
            block = out(reg32) block,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    (block, thread)
}

#[inline(always)]
fn shared_buffers() -> SharedBuffers {
    let key_base: u32;
    let value_base: u32;
    let score_base: u32;
    let control_base: u32;
    // SAFETY: Declares the bounded BF16-decoded tiles, score partials, and scalars for one CTA.
    unsafe {
        asm!(
            ".shared .align 4 .b8 attention_online_keys[8192];",
            ".shared .align 4 .b8 attention_online_values[8192];",
            ".shared .align 4 .b8 attention_online_score_partials[8192];",
            ".shared .align 4 .b8 attention_online_controls[64];",
            "mov.u32 {key_base}, attention_online_keys;",
            "mov.u32 {value_base}, attention_online_values;",
            "mov.u32 {score_base}, attention_online_score_partials;",
            "mov.u32 {control_base}, attention_online_controls;",
            key_base = out(reg32) key_base,
            value_base = out(reg32) value_base,
            score_base = out(reg32) score_base,
            control_base = out(reg32) control_base,
            options(nostack),
        )
    };
    SharedBuffers {
        keys: key_base,
        values: value_base,
        scores: score_base,
        controls: control_base,
    }
}

#[inline(always)]
fn store_shared_f32(base: u32, index: u32, value: f32) {
    // SAFETY: Callers keep `index` inside the array identified by `base`.
    let address = base + index * 4;
    unsafe {
        asm!(
            "st.shared.f32 [{address}], {value};",
            address = in(reg32) address,
            value = in(reg32) value,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_f32(base: u32, index: u32) -> f32 {
    let value: f32;
    // SAFETY: Callers keep `index` inside the array identified by `base`.
    let address = base + index * 4;
    unsafe {
        asm!(
            "ld.shared.f32 {value}, [{address}];",
            value = out(reg32) value,
            address = in(reg32) address,
            options(nostack),
        )
    };
    value
}

#[inline(always)]
fn block_barrier() {
    // SAFETY: The entrypoint keeps every 256-thread CTA converged at each barrier.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn fp32_add_rn(left: f32, right: f32) -> f32 {
    let sum: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {sum}, {left}, {right};",
            sum = out(reg32) sum,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
fn fp32_subtract_rn(left: f32, right: f32) -> f32 {
    let difference: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "sub.rn.f32 {difference}, {left}, {right};",
            difference = out(reg32) difference,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    difference
}

#[inline(always)]
fn fp32_multiply_rn(left: f32, right: f32) -> f32 {
    let product: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f32 {product}, {left}, {right};",
            product = out(reg32) product,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    product
}

#[inline(always)]
fn fp32_exp_nonpositive(argument: f32) -> f32 {
    let exponent = fp32_multiply_rn(argument, core::f32::consts::LOG2_E);
    let result: f32;
    // SAFETY: PTX approximate base-2 exponent has no memory or stack effects.
    unsafe {
        asm!(
            "ex2.approx.f32 {result}, {exponent};",
            result = out(reg32) result,
            exponent = in(reg32) exponent,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn encode_bf16_rne(value: f32) -> u16 {
    let rounded: u16;
    // SAFETY: Converts one scalar FP32 value to BF16 with round-to-nearest-even.
    unsafe {
        asm!(
            "cvt.rn.bf16.f32 {rounded}, {value};",
            rounded = out(reg16) rounded,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    rounded
}

#[inline(always)]
fn stage_key_value_tile(shared: SharedBuffers, cache: CacheView, tile: KeyTile, thread: u32) {
    let width_usize = tile.width as usize;
    let tile_elements = TILE_KEYS as usize * width_usize;
    let mut element = thread as usize;
    while element < tile_elements {
        let tile_key = element / width_usize;
        let channel = element % width_usize;
        let key_position = tile.start + tile_key as u32;
        let (key, value) = if key_position < tile.visible {
            let cache_index = key_position as usize * cache.row_width + cache.head_offset + channel;
            // SAFETY: The caller validates capacity and cache extents for every visible position.
            unsafe {
                (
                    decode_bf16(cache.keys.add(cache_index).read()),
                    decode_bf16(cache.values.add(cache_index).read()),
                )
            }
        } else {
            (0.0, 0.0)
        };
        store_shared_f32(shared.keys, element as u32, key);
        store_shared_f32(shared.values, element as u32, value);
        element += BLOCK_THREADS as usize;
    }
}

#[inline(always)]
fn reduce_tile_dots(shared: SharedBuffers, tile: KeyTile, query_value: f32, thread: u32) {
    let mut tile_key = 0;
    while tile_key < TILE_KEYS {
        let key = if thread < tile.width {
            load_shared_f32(shared.keys, tile_key * tile.width + thread)
        } else {
            0.0
        };
        let partial = fp32_multiply_rn(query_value, key);
        store_shared_f32(shared.scores, tile_key * BLOCK_THREADS + thread, partial);
        tile_key += 1;
    }
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let mut key = 0;
            while key < TILE_KEYS {
                let index = key * BLOCK_THREADS + thread;
                let left = load_shared_f32(shared.scores, index);
                let right = load_shared_f32(shared.scores, index + stride);
                store_shared_f32(shared.scores, index, fp32_add_rn(left, right));
                key += 1;
            }
        }
        block_barrier();
        stride >>= 1;
    }
}

#[inline(always)]
fn publish_tile_weights(
    shared: SharedBuffers,
    thread: u32,
    tile: KeyTile,
    scale: f32,
    state: &mut OnlineState,
) {
    if thread != 0 {
        return;
    }

    let mut tile_maximum = f32::NEG_INFINITY;
    let mut key = 0;
    while key < TILE_KEYS {
        if tile.start + key < tile.visible {
            let dot = load_shared_f32(shared.scores, key * BLOCK_THREADS);
            let score = fp32_multiply_rn(dot, scale);
            if score > tile_maximum {
                tile_maximum = score;
            }
        }
        key += 1;
    }

    let next_maximum = if state.maximum > tile_maximum {
        state.maximum
    } else {
        tile_maximum
    };
    let alpha = if state.normalizer == 0.0 {
        0.0
    } else {
        fp32_exp_nonpositive(fp32_subtract_rn(state.maximum, next_maximum))
    };
    let old_normalizer = state.normalizer;
    let mut tile_normalizer = 0.0;
    key = 0;
    while key < TILE_KEYS {
        let weight = if tile.start + key < tile.visible {
            let dot = load_shared_f32(shared.scores, key * BLOCK_THREADS);
            let score = fp32_multiply_rn(dot, scale);
            fp32_exp_nonpositive(fp32_subtract_rn(score, next_maximum))
        } else {
            0.0
        };
        tile_normalizer = fp32_add_rn(tile_normalizer, weight);
        store_shared_f32(shared.controls, CONTROL_WEIGHT_BASE + key, weight);
        key += 1;
    }
    state.normalizer = fp32_add_rn(fp32_multiply_rn(old_normalizer, alpha), tile_normalizer);
    state.maximum = next_maximum;
    store_shared_f32(shared.controls, CONTROL_ALPHA, alpha);
    store_shared_f32(shared.controls, CONTROL_NORMALIZER, state.normalizer);
}

#[inline(always)]
fn accumulate_value_tile(shared: SharedBuffers, thread: u32, tile: KeyTile, accumulator: &mut f32) {
    if thread < tile.width {
        let alpha = load_shared_f32(shared.controls, CONTROL_ALPHA);
        let mut weighted_tile = 0.0;
        let mut key = 0;
        while key < TILE_KEYS {
            let weight = load_shared_f32(shared.controls, CONTROL_WEIGHT_BASE + key);
            let value = load_shared_f32(shared.values, key * tile.width + thread);
            weighted_tile = fp32_add_rn(weighted_tile, fp32_multiply_rn(weight, value));
            key += 1;
        }
        *accumulator = fp32_add_rn(fp32_multiply_rn(*accumulator, alpha), weighted_tile);
    }
}

#[inline(always)]
fn valid_launch(
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    scale: f32,
) -> bool {
    rows > 0
        && rows <= 2048
        && query_heads > 0
        && query_heads <= 128
        && kv_heads > 0
        && kv_heads <= 128
        && query_heads >= kv_heads
        && query_heads % kv_heads == 0
        && (2..=BLOCK_THREADS).contains(&width)
        && capacity > 0
        && capacity <= 262_144
        && past <= capacity
        && rows <= capacity - past
        && scale > 0.0
        && scale <= f32::MAX
}

/// Run causal grouped-query attention with eight-key shared-memory tiles.
///
/// Q and BF16 output use compact `[rows, query_heads, width]` layout. K/V use
/// full-capacity token-major `[capacity, kv_heads, width]` layout. Query head h
/// maps to KV head `h / (query_heads / kv_heads)`, and row r reads keys through
/// `past + r`. Scores and the online softmax/value recurrence use FP32. This is
/// a separate arithmetic profile from `causal_attention_bf16`.
///
/// # Safety
/// Launch `grid = [rows * query_heads, 1, 1]` and `block = [256, 1, 1]`. Require
/// `rows` in `1..=2048`, both head counts in `1..=128`, `query_heads >= kv_heads`
/// with exact divisibility, `width` in `2..=256`, and `capacity` in
/// `1..=262144`. Require a positive finite `scale` and `past + rows <= capacity`.
/// Q must cover `rows * query_heads * width` readable BF16 elements. Each cache
/// must cover `capacity * kv_heads * width` readable BF16 elements. Output must
/// cover the Q extent in writable BF16 elements; `unrounded` must cover that
/// extent in writable FP32 elements. Every initialized Q/K/V value, scaled dot,
/// and final FP32 output must be finite. Read ranges must not overlap either
/// output range, and the output ranges must be disjoint. All pointers must be
/// aligned and live through completion; the host must check extent arithmetic.
/// Every thread in each launched CTA participates in every shared barrier.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_online_bf16(
    q: *const u16,
    cache_k: *const u16,
    cache_v: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    scale: f32,
) {
    if !valid_launch(rows, query_heads, kv_heads, width, past, capacity, scale) {
        return;
    }

    let (block, thread) = block_and_thread();
    let block_count = rows as u64 * query_heads as u64;
    if block as u64 >= block_count {
        return;
    }

    let shared = shared_buffers();
    let block_index = block as usize;
    let width_usize = width as usize;
    let row = block_index / query_heads as usize;
    let query_head = block_index % query_heads as usize;
    let heads_per_kv = (query_heads / kv_heads) as usize;
    let kv_head = query_head / heads_per_kv;
    let query_start = block_index * width_usize;
    let cache = CacheView {
        keys: cache_k,
        values: cache_v,
        row_width: kv_heads as usize * width_usize,
        head_offset: kv_head * width_usize,
    };
    let query_value = if thread < width {
        // SAFETY: The host launch contract bounds this query/head/channel coordinate.
        unsafe { decode_bf16(q.add(query_start + thread as usize).read()) }
    } else {
        0.0
    };
    let visible_keys = past + row as u32 + 1;
    let mut state = OnlineState {
        maximum: f32::NEG_INFINITY,
        normalizer: 0.0,
    };
    let mut accumulator = 0.0_f32;
    let mut tile_start = 0;
    while tile_start < visible_keys {
        let tile = KeyTile {
            width,
            start: tile_start,
            visible: visible_keys,
        };
        stage_key_value_tile(shared, cache, tile, thread);
        block_barrier();
        reduce_tile_dots(shared, tile, query_value, thread);
        publish_tile_weights(shared, thread, tile, scale, &mut state);
        block_barrier();
        accumulate_value_tile(shared, thread, tile, &mut accumulator);
        block_barrier();
        tile_start += TILE_KEYS;
    }

    if thread < width {
        let output_index = query_start + thread as usize;
        let final_normalizer = load_shared_f32(shared.controls, CONTROL_NORMALIZER);
        let normalized = fp32_divide_rn(accumulator, final_normalizer);
        let rounded = encode_bf16_rne(normalized);
        // SAFETY: This CTA exclusively owns the query/head output row and diagnostic values.
        unsafe {
            output.add(output_index).write(rounded);
            unrounded.add(output_index).write(normalized);
        }
    }
}

#[inline(always)]
fn fp32_divide_rn(numerator: f32, denominator: f32) -> f32 {
    let quotient: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "div.rn.f32 {quotient}, {numerator}, {denominator};",
            quotient = out(reg32) quotient,
            numerator = in(reg32) numerator,
            denominator = in(reg32) denominator,
            options(nomem, nostack),
        )
    };
    quotient
}
