use core::arch::asm;

const BLOCK_THREADS: u32 = 256;
const LOG2_E: f32 = core::f32::consts::LOG2_E;

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
    // SAFETY: Declares one CTA-local FP32 partial for each of its 256 threads.
    unsafe {
        asm!(
            ".shared .align 4 .b8 causal_attention_partials[1024];",
            "mov.u32 {base}, causal_attention_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_partial(base: u32, index: u32, value: f32) {
    // SAFETY: Callers use indices 0..256 in the CTA's 1024-byte shared array.
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
fn load_partial(base: u32, index: u32) -> f32 {
    let value: f32;
    // SAFETY: Callers use indices 0..256 in the CTA's 1024-byte shared array.
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
    // SAFETY: All 256 threads reach every reduction and iteration barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn add_rn(left: f32, right: f32) -> f32 {
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
fn multiply_rn(left: f32, right: f32) -> f32 {
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
fn subtract_rn(left: f32, right: f32) -> f32 {
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
fn divide_rn(numerator: f32, denominator: f32) -> f32 {
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

#[inline(always)]
fn exp_approx(value: f32) -> f32 {
    let exponent = multiply_rn(value, LOG2_E);
    let result: f32;
    // SAFETY: This scalar approximate base-2 exponential has no memory or stack effects.
    unsafe {
        asm!(
            "ex2.approx.ftz.f32 {result}, {exponent};",
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
fn encode_bf16_rn(value: f32) -> u16 {
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

#[inline(always)]
fn reduce_dot(shared: u32, thread: u32, partial: f32) -> f32 {
    store_partial(shared, thread, partial);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_partial(shared, thread);
            let right = load_partial(shared, thread + stride);
            store_partial(shared, thread, add_rn(left, right));
        }
        block_barrier();
        stride /= 2;
    }
    load_partial(shared, 0)
}

/// Apply causal online softmax attention for one BF16 query row per CTA.
///
/// `q` and `output` use compact `[rows, query_heads, width]` layout. The persistent
/// K/V caches use `[capacity, kv_heads, width]`; query heads map evenly onto KV
/// heads. Row `r` attends to cache tokens `0..=past + r`. The implementation keeps
/// only one FP32 score reduction and one online softmax accumulator per output
/// channel, without materializing scores or probabilities. Qualification assumes
/// finite normal-range scores; approximate exponentiation can flush very small
/// tail contributions to zero.
///
/// The numerical test profile uses finite Q values and finite K/V values in the
/// initialized cache prefix `[0, past + rows)` with a positive finite scale. The
/// unused cache suffix may contain NaN poison and is not read. The harness checks
/// final FP32 diagnostics for finiteness after launch; arbitrary extreme finite
/// inputs may overflow FP32 intermediates.
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
    let mut query_value = 0.0_f32;
    if thread < width {
        // SAFETY: This active lane owns one in-range BF16 query channel.
        query_value = unsafe { decode_bf16(*q.add(q_start + thread_index)) };
    }

    let sequence_len = past + row as u32 + 1;
    let mut maximum = f32::NEG_INFINITY;
    let mut normalizer = 0.0_f32;
    let mut accumulator = 0.0_f32;
    let mut token = 0_u32;
    while token < sequence_len {
        let cache_row = token as usize * kv_row_width + kv_head_offset;
        let key_value = if thread < width {
            // SAFETY: Active channels address this KV head in a token below sequence_len <= capacity.
            unsafe { decode_bf16(*cache_k.add(cache_row + thread_index)) }
        } else {
            0.0_f32
        };
        let partial = multiply_rn(query_value, key_value);
        let dot = reduce_dot(shared, thread, partial);
        let score = multiply_rn(dot, scale);
        let next_maximum = if score > maximum { score } else { maximum };
        let alpha = if normalizer == 0.0 {
            0.0_f32
        } else {
            exp_approx(subtract_rn(maximum, next_maximum))
        };
        let beta = exp_approx(subtract_rn(score, next_maximum));
        let next_normalizer = add_rn(multiply_rn(normalizer, alpha), beta);
        let value = if thread < width {
            // SAFETY: Active channels address the same in-range token and KV head in the V cache.
            unsafe { decode_bf16(*cache_v.add(cache_row + thread_index)) }
        } else {
            0.0_f32
        };
        accumulator = add_rn(multiply_rn(accumulator, alpha), multiply_rn(beta, value));
        maximum = next_maximum;
        normalizer = next_normalizer;

        // SAFETY: Prevents faster lanes from overwriting shared score partials before every
        // lane has consumed shared[0] and completed the online update for this token.
        block_barrier();
        token += 1;
    }

    if thread < width {
        let output_index = q_start + thread_index;
        let normalized = divide_rn(accumulator, normalizer);
        let rounded = encode_bf16_rn(normalized);
        // SAFETY: This active lane exclusively owns its output and diagnostic channels.
        unsafe {
            output.add(output_index).write(rounded);
            unrounded.add(output_index).write(normalized);
        }
    }
}
