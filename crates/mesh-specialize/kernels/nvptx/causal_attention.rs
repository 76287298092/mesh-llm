use core::arch::asm;

const BLOCK_THREADS: u32 = 256;

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
fn linear_thread_index() -> usize {
    let (block, thread) = block_and_thread();
    block as usize * BLOCK_THREADS as usize + thread as usize
}

#[inline(always)]
fn attention_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one CTA-local FP64 partial for each of its 256 threads.
    unsafe {
        asm!(
            ".shared .align 8 .b8 causal_attention_partials[2048];",
            "mov.u32 {base}, causal_attention_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
pub(super) fn store_partial(base: u32, index: u32, value: f64) {
    // SAFETY: Callers use indices 0..256 in the CTA's 2048-byte shared array.
    let address = base + index * 8;
    unsafe {
        asm!(
            "st.shared.f64 [{address}], {value};",
            address = in(reg32) address,
            value = in(reg64) value,
            options(nostack),
        )
    };
}

#[inline(always)]
pub(super) fn load_partial(base: u32, index: u32) -> f64 {
    let value: f64;
    // SAFETY: Callers use indices 0..256 in the CTA's 2048-byte shared array.
    let address = base + index * 8;
    unsafe {
        asm!(
            "ld.shared.f64 {value}, [{address}];",
            value = out(reg64) value,
            address = in(reg32) address,
            options(nostack),
        )
    };
    value
}

#[inline(always)]
pub(super) fn block_barrier() {
    // SAFETY: All 256 threads reach every reduction and iteration barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
pub(super) fn add_rn(left: f64, right: f64) -> f64 {
    let sum: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f64 {sum}, {left}, {right};",
            sum = out(reg64) sum,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
pub(super) fn multiply_rn(left: f64, right: f64) -> f64 {
    let product: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f64 {product}, {left}, {right};",
            product = out(reg64) product,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    product
}

#[inline(always)]
pub(super) fn subtract_rn(left: f64, right: f64) -> f64 {
    let difference: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "sub.rn.f64 {difference}, {left}, {right};",
            difference = out(reg64) difference,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    difference
}

#[inline(always)]
pub(super) fn divide_rn(numerator: f64, denominator: f64) -> f64 {
    let quotient: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "div.rn.f64 {quotient}, {numerator}, {denominator};",
            quotient = out(reg64) quotient,
            numerator = in(reg64) numerator,
            denominator = in(reg64) denominator,
            options(nomem, nostack),
        )
    };
    quotient
}

#[inline(always)]
pub(super) fn fp32_to_fp64_exact(value: f32) -> f64 {
    let converted: f64;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.f64.f32 {converted}, {value};",
            converted = out(reg64) converted,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
pub(super) fn fp64_to_fp32_rn(value: f64) -> f32 {
    let converted: f32;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.rn.f32.f64 {converted}, {value};",
            converted = out(reg32) converted,
            value = in(reg64) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
pub(super) fn decode_bf16(bits: u16) -> f64 {
    fp32_to_fp64_exact(f32::from_bits((bits as u32) << 16))
}

#[inline(always)]
pub(super) fn encode_bf16_rn(value: f32) -> u16 {
    let rounded: u16;
    // SAFETY: Converts one scalar FP32 value to BF16 using round-to-nearest-even.
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

/// Append compact BF16 key/value rows to their persistent token-major caches.
///
/// `k` and `v` are compact `[rows, kv_heads, width]` matrices. The destination
/// caches have `[capacity, kv_heads, width]` layout; this kernel writes the rows
/// beginning at token `past` and does not touch any earlier or later cache slots.
///
/// # Safety
/// Launch `ceil(rows * kv_heads * width / 256)` blocks of 256 threads. Require
/// `rows` in `1..=2048`, `kv_heads` in `1..=128`, `width` in `2..=256`,
/// `past + rows <= capacity <= 262144`, and all element counts and indices to fit
/// `usize`. `k` and `v` must each cover `rows * kv_heads * width` readable BF16
/// values. `cache_k` and `cache_v` must each cover `capacity * kv_heads * width`
/// writable BF16 values. `cache_k` and `cache_v` must be disjoint from each other
/// and from both source ranges. All pointers must be aligned and live through
/// completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_kv_append(
    k: *const u16,
    v: *const u16,
    cache_k: *mut u16,
    cache_v: *mut u16,
    rows: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
) {
    // SAFETY: The entry's contract is exactly the shared body's contract.
    unsafe {
        append_body(
            k, v, cache_k, cache_v, rows, kv_heads, width, past, capacity,
        );
    }
}

/// Shared exact append body; callers uphold `attention_kv_append`'s contract.
#[inline(always)]
pub(super) unsafe fn append_body(
    k: *const u16,
    v: *const u16,
    cache_k: *mut u16,
    cache_v: *mut u16,
    rows: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
) {
    if rows == 0
        || rows > 2048
        || kv_heads == 0
        || kv_heads > 128
        || width < 2
        || width > BLOCK_THREADS
        || capacity > 262_144
        || past > capacity
        || rows > capacity - past
    {
        return;
    }

    let index = linear_thread_index();
    let count = rows as usize * kv_heads as usize * width as usize;
    if index >= count {
        return;
    }
    let cache_start = past as usize * kv_heads as usize * width as usize;
    // SAFETY: The grid bounds `index`; the capacity contract bounds its append destination.
    let (key, value) = unsafe { (*k.add(index), *v.add(index)) };
    // SAFETY: Each active thread owns a distinct in-range K/V cache element.
    unsafe {
        cache_k.add(cache_start + index).write(key);
        cache_v.add(cache_start + index).write(value);
    }
}

/// Apply causal online softmax attention for one BF16 query row per CTA.
///
/// `q` and `output` use compact `[rows, query_heads, width]` layout. The persistent
/// K/V caches use `[capacity, kv_heads, width]`; query heads map evenly onto KV
/// heads. Row `r` attends to cache tokens `0..=past + r`. The implementation keeps
/// only one FP64 score reduction and one online softmax accumulator per output
/// channel, without materializing scores or probabilities. The online maximum,
/// weights, normalizer, and weighted-value accumulator also use FP64. A range-reduced
/// exponential avoids flushing small tail contributions. This accurate baseline is
/// untuned; performance has not been measured.
///
/// The numerical test profile uses finite Q values and finite K/V values in the
/// initialized cache prefix `[0, past + rows)` with a positive finite scale. The
/// unused cache suffix may contain NaN poison and is not read. The harness checks
/// final FP32 diagnostics for finiteness after launch; sufficiently extreme finite
/// inputs may still overflow when the result is converted to FP32.
///
/// # Safety
/// Launch `grid = [rows * query_heads, 1, 1]` and `block = [256, 1, 1]`. Require
/// positive `rows`, `query_heads`, and `kv_heads`; require `width` in `1..=256`.
/// `query_heads` must be at least `kv_heads` and divisible by it. Require
/// `past + rows <= capacity <= 262144`; all index arithmetic must fit `usize`.
/// `q` must cover `rows * query_heads * width` readable BF16 values. Each cache
/// must cover `capacity * kv_heads * width` readable BF16 values. `output` must
/// cover `rows * query_heads * width` writable BF16 values and `unrounded` that
/// many writable FP32 values. Pointers must be aligned, nonaliasing where writes
/// are involved, and live through completion. Read ranges must not overlap the
/// writable output ranges, which must also be disjoint from each other.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn causal_attention_bf16(
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
    // SAFETY: The entry's contract is exactly the shared body's contract.
    unsafe {
        attention_body::<false>(
            q,
            cache_k,
            cache_v,
            output,
            unrounded,
            rows,
            query_heads,
            kv_heads,
            width,
            past,
            capacity,
            scale,
        );
    }
}

#[inline(always)]
fn attention_exp<const UNROLLED: bool>(value: f64) -> f64 {
    if UNROLLED {
        super::exponential_unrolled::exp_nonpositive(value)
    } else {
        super::exponential::exp_nonpositive(value)
    }
}

/// Shared control schedule; false retains the original exponential implementation.
/// Both specializations inherit `causal_attention_bf16`'s full safety contract.
#[inline(always)]
pub(super) unsafe fn attention_body<const UNROLLED_EXP: bool>(
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
    if rows == 0
        || query_heads == 0
        || kv_heads == 0
        || query_heads < kv_heads
        || query_heads % kv_heads != 0
        || width == 0
        || width > BLOCK_THREADS
        || capacity > 262_144
        || past > capacity
        || rows > capacity - past
    {
        return;
    }

    let (block, thread) = block_and_thread();
    let block_count = rows as u64 * query_heads as u64;
    if block as u64 >= block_count {
        return;
    }

    let shared = attention_partials_base();
    let block_index = block as usize;
    let thread_index = thread as usize;
    let width_usize = width as usize;
    let row = block_index / query_heads as usize;
    let query_head = block_index % query_heads as usize;
    let heads_per_kv = (query_heads / kv_heads) as usize;
    let kv_head = query_head / heads_per_kv;
    let q_start = block_index * width_usize;
    let kv_row_width = kv_heads as usize * width_usize;
    let kv_head_offset = kv_head * width_usize;
    let mut query_value = 0.0_f64;
    if thread < width {
        // SAFETY: This active lane owns one in-range BF16 query channel.
        query_value = unsafe { decode_bf16(*q.add(q_start + thread_index)) };
    }

    let sequence_len = past + row as u32 + 1;
    let mut maximum = f64::NEG_INFINITY;
    let mut normalizer = 0.0_f64;
    let mut accumulator = 0.0_f64;
    let scale = fp32_to_fp64_exact(scale);
    let mut token = 0_u32;
    while token < sequence_len {
        let cache_row = token as usize * kv_row_width + kv_head_offset;
        let key_value = if thread < width {
            // SAFETY: Active channels address this KV head in a token below sequence_len <= capacity.
            unsafe { decode_bf16(*cache_k.add(cache_row + thread_index)) }
        } else {
            0.0_f64
        };
        let partial = multiply_rn(query_value, key_value);
        let dot = super::attention_reduction::reduce_dot(shared, thread, partial);
        if thread == 0 {
            let score = multiply_rn(dot, scale);
            let next_maximum = if score > maximum { score } else { maximum };
            let alpha = if normalizer == 0.0 {
                0.0_f64
            } else {
                attention_exp::<UNROLLED_EXP>(subtract_rn(maximum, next_maximum))
            };
            let beta = attention_exp::<UNROLLED_EXP>(subtract_rn(score, next_maximum));
            normalizer = add_rn(multiply_rn(normalizer, alpha), beta);
            maximum = next_maximum;
            // Only slot zero is still being consumed as the dot by other threads.
            // These disjoint scalar slots are published at the barrier below.
            store_partial(shared, 1, alpha);
            store_partial(shared, 2, beta);
            store_partial(shared, 3, normalizer);
        }
        block_barrier();
        let alpha = load_partial(shared, 1);
        let beta = load_partial(shared, 2);
        normalizer = load_partial(shared, 3);
        let value = if thread < width {
            // SAFETY: Active channels address the same in-range token and KV head in the V cache.
            unsafe { decode_bf16(*cache_v.add(cache_row + thread_index)) }
        } else {
            0.0_f64
        };
        accumulator = add_rn(multiply_rn(accumulator, alpha), multiply_rn(beta, value));

        // SAFETY: Prevents faster lanes from overwriting shared score partials before every
        // lane has consumed shared[0] and completed the online update for this token.
        block_barrier();
        token += 1;
    }

    if thread < width {
        let output_index = q_start + thread_index;
        let normalized = fp64_to_fp32_rn(divide_rn(accumulator, normalizer));
        let rounded = encode_bf16_rn(normalized);
        // SAFETY: This active lane exclusively owns its output and diagnostic channels.
        unsafe {
            output.add(output_index).write(rounded);
            unrounded.add(output_index).write(normalized);
        }
    }
}
