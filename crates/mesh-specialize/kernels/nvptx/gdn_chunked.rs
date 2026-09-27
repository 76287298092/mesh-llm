use core::arch::asm;

const MAX_CHUNK_ROWS: usize = 16;
const MAX_WIDTH: usize = 128;
const DECAY_STRIDE: usize = MAX_CHUNK_ROWS + 1;
const SCRATCH_WIDTH: usize = MAX_WIDTH;
const PREPARE_THREADS: u32 = 256;

#[inline(always)]
fn cta_tile_thread() -> (u32, u32, u32) {
    let head: u32;
    let tile: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's CTA and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {head}, %ctaid.x;",
            "mov.u32 {tile}, %ctaid.y;",
            "mov.u32 {thread}, %tid.x;",
            head = out(reg32) head,
            tile = out(reg32) tile,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    (head, tile, thread)
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
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
unsafe fn decay_product(
    decay: *const f32,
    value_heads: usize,
    value_head: usize,
    begin: usize,
    end: usize,
) -> f32 {
    let mut product = 1.0_f32;
    let mut time = begin;
    while time < end {
        let index = time * value_heads + value_head;
        // SAFETY: The launch contract covers every `[time, value_head]` decay entry.
        let value = unsafe { *decay.add(index) };
        product = fp32_multiply_rn(product, value);
        time += 1;
    }
    product
}

#[inline(always)]
unsafe fn dot_keys(
    left: *const f32,
    left_start: usize,
    right: *const f32,
    right_start: usize,
    width: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut index = 0;
    while index < width {
        // SAFETY: The launch contract covers both selected key vectors for `width` elements.
        let (left_value, right_value) = unsafe {
            (
                *left.add(left_start + index),
                *right.add(right_start + index),
            )
        };
        sum = fp32_add_rn(sum, fp32_multiply_rn(left_value, right_value));
        index += 1;
    }
    sum
}

#[inline(always)]
unsafe fn dot_state_column(
    vector: *const f32,
    vector_start: usize,
    state: *const f32,
    state_start: usize,
    width: usize,
    value_dimension: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut key_dimension = 0;
    while key_dimension < width {
        // SAFETY: The launch contract covers the selected vector and state column.
        let (vector_value, state_value) = unsafe {
            (
                *vector.add(vector_start + key_dimension),
                *state.add(state_start + key_dimension * width + value_dimension),
            )
        };
        sum = fp32_add_rn(sum, fp32_multiply_rn(vector_value, state_value));
        key_dimension += 1;
    }
    sum
}

/// Prepare one chunk's lower-triangular coefficients, right-hand side, and decay products.
///
/// Logical inputs are the existing GDN layout: K is `[rows, key_heads, width]`,
/// V is the last section of time-major QKV `[rows, 2*key_heads + value_heads,
/// width]`, beta and decay are `[rows, value_heads]`, and initial state is
/// `[value_heads, width, width]`. Value head `h` maps to key head
/// `h / (value_heads / key_heads)`. For each head this writes the strict lower
/// triangle `L[t,j] = beta[t] * D(t,j) * dot(K[t], K[j])`,
/// `B[t] = beta[t] * (V[t] - D(t,-1) * dot(K[t], S0_column))`, and all valid
/// `D(t,j)` plus `D(t,-1)`. Scratch uses fixed strides `[head,16,16]`,
/// `[head,16,128]`, and `[head,16,17]`; invalid upper-triangle coefficients
/// and decay cells are initialized to zero. Every product is built directly,
/// without dividing by a cumulative decay.
///
/// # Safety
/// Launch `grid = [value_heads, ceil((rows*rows + 17*rows + rows*width)/256), 1]`
/// and `block = [256, 1, 1]`. Require `rows` in 1..=16, `key_heads` in 1..=64,
/// `value_heads` in 1..=256 and divisible by `key_heads`, and power-of-two
/// `width` in 1..=128. `k` must cover `rows*key_heads*width` readable `f32`
/// values; `qkv` must cover `rows*(2*key_heads + value_heads)*width` readable
/// BF16 values with V in the last section; `beta` and `decay` must each cover
/// `rows*value_heads` readable BF16 and `f32` values; `initial_state` must cover
/// `value_heads*width*width` readable `f32` values. `coefficients`, `rhs`, and
/// `decay_products` must cover respectively `value_heads*16*16`,
/// `value_heads*16*128`, and `value_heads*16*17` writable `f32` values. Host
/// validation must enforce finite Q/K/QKV/state, beta and decay in [0,1], exact
/// extents, nonoverlapping aligned pointers, and checked size arithmetic. Inputs
/// and scratch remain live until the launch completes; the host rejects any
/// nonfinite or overflowing intermediate before accepting results.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_chunk_prepare(
    k: *const f32,
    qkv: *const u16,
    beta: *const u16,
    decay: *const f32,
    initial_state: *const f32,
    coefficients: *mut f32,
    rhs: *mut f32,
    decay_products: *mut f32,
    rows: u32,
    key_heads: u32,
    value_heads: u32,
    width: u32,
) {
    let (value_head, tile, thread) = cta_tile_thread();
    if value_head >= value_heads
        || key_heads == 0
        || value_heads == 0
        || rows == 0
        || rows as usize > MAX_CHUNK_ROWS
        || width == 0
        || width as usize > MAX_WIDTH
    {
        return;
    }

    let rows = rows as usize;
    let key_heads = key_heads as usize;
    let value_heads = value_heads as usize;
    let width = width as usize;
    let head = value_head as usize;
    let key_head = head / (value_heads / key_heads);
    let coefficient_count = rows * rows;
    let decay_count = rows * DECAY_STRIDE;
    let rhs_count = rows * width;
    let total = coefficient_count + decay_count + rhs_count;
    let flat = tile as usize * PREPARE_THREADS as usize + thread as usize;
    if flat >= total {
        return;
    }

    if flat < coefficient_count {
        let time = flat / rows;
        let key_time = flat % rows;
        let coefficient = if key_time < time {
            // SAFETY: This row range is within the validated K and decay tensors.
            let (decay_factor, key_dot, beta_value) = unsafe {
                let decay_factor = decay_product(decay, value_heads, head, key_time + 1, time + 1);
                let k_row = (time * key_heads + key_head) * width;
                let j_row = (key_time * key_heads + key_head) * width;
                let key_dot = dot_keys(k, k_row, k, j_row, width);
                let beta_value = decode_bf16(*beta.add(time * value_heads + head));
                (decay_factor, key_dot, beta_value)
            };
            fp32_multiply_rn(fp32_multiply_rn(beta_value, decay_factor), key_dot)
        } else {
            0.0_f32
        };
        let index =
            head * MAX_CHUNK_ROWS * MAX_CHUNK_ROWS + (flat / rows) * MAX_CHUNK_ROWS + flat % rows;
        // SAFETY: Each flattened coefficient worker owns one element in the fixed-stride scratch.
        unsafe { coefficients.add(index).write(coefficient) };
        return;
    }

    let decay_flat = flat - coefficient_count;
    if decay_flat < decay_count {
        let time = decay_flat / DECAY_STRIDE;
        let slot = decay_flat % DECAY_STRIDE;
        let product = if slot == MAX_CHUNK_ROWS {
            // SAFETY: The prefix range is within this value head's validated decay rows.
            unsafe { decay_product(decay, value_heads, head, 0, time + 1) }
        } else if slot <= time {
            // SAFETY: This decay interval is within this value head's validated rows.
            unsafe { decay_product(decay, value_heads, head, slot + 1, time + 1) }
        } else {
            0.0_f32
        };
        let index = head * MAX_CHUNK_ROWS * DECAY_STRIDE + time * DECAY_STRIDE + slot;
        // SAFETY: Every time/slot worker owns one element in the fixed-stride scratch.
        unsafe { decay_products.add(index).write(product) };
        return;
    }

    let rhs_flat = decay_flat - decay_count;
    let time = rhs_flat / width;
    let value_dimension = rhs_flat % width;
    let key_start = (time * key_heads + key_head) * width;
    let state_start = head * width * width;
    // SAFETY: The flattened RHS worker selects one valid row/value dimension.
    let (beta_value, decay_prefix, value) = unsafe {
        let beta_value = decode_bf16(*beta.add(time * value_heads + head));
        let prefix = decay_product(decay, value_heads, head, 0, time + 1);
        let input_row = time * (2 * key_heads + value_heads) * width;
        let value_index = input_row + (2 * key_heads + head) * width + value_dimension;
        (beta_value, prefix, decode_bf16(*qkv.add(value_index)))
    };
    // SAFETY: The host contract covers this K vector and initial-state column.
    let state_dot = unsafe {
        dot_state_column(
            k,
            key_start,
            initial_state,
            state_start,
            width,
            value_dimension,
        )
    };
    let decayed_prediction = fp32_multiply_rn(decay_prefix, state_dot);
    let residual = fp32_subtract_rn(value, decayed_prediction);
    let right_hand_side = fp32_multiply_rn(beta_value, residual);
    let index = head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + time * SCRATCH_WIDTH + value_dimension;
    // SAFETY: Each flattened RHS worker owns one element in the fixed-stride scratch.
    unsafe { rhs.add(index).write(right_hand_side) };
}

/// Solve the prepared lower-triangular update system independently per value column.
///
/// For every head and value column, this computes `u[t] = B[t] -
/// sum(j < t, L[t,j] * u[j])` in increasing `j` order. Different heads and
/// value columns are independent. This serial time loop is the small triangular
/// solve itself; it does not execute the original recurrent state mutation.
///
/// # Safety
/// Launch `grid = [value_heads, 1, 1]` and `block = [width, 1, 1]`, with the
/// same bounded rows/head/width constraints as `gdn_chunk_prepare`. `coefficients`
/// must cover `value_heads*16*16`, `rhs` and writable `updates` must each cover
/// `value_heads*16*128` readable/writable `f32` values. All extents and index
/// arithmetic must fit `usize`; pointers must be aligned, mutually disjoint,
/// and live through kernel completion. The host validates finite inputs and
/// rejects nonfinite or overflowing updates.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_chunk_solve(
    coefficients: *const f32,
    rhs: *const f32,
    updates: *mut f32,
    rows: u32,
    value_heads: u32,
    width: u32,
) {
    let (value_head, _tile, column) = cta_tile_thread();
    if value_head >= value_heads
        || column >= width
        || rows == 0
        || rows as usize > MAX_CHUNK_ROWS
        || width == 0
        || width as usize > MAX_WIDTH
    {
        return;
    }

    let head = value_head as usize;
    let rows = rows as usize;
    let column = column as usize;
    let mut time = 0;
    while time < rows {
        let rhs_index = head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + time * SCRATCH_WIDTH + column;
        // SAFETY: The validated launch dimensions select an initialized RHS element.
        let mut value = unsafe { *rhs.add(rhs_index) };
        let mut prior_time = 0;
        let mut sum = 0.0_f32;
        while prior_time < time {
            let coefficient_index =
                head * MAX_CHUNK_ROWS * MAX_CHUNK_ROWS + time * MAX_CHUNK_ROWS + prior_time;
            let prior_update_index =
                head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + prior_time * SCRATCH_WIDTH + column;
            // SAFETY: The strict lower triangle and earlier updates were initialized by prior stages.
            let (coefficient, prior_update) = unsafe {
                (
                    *coefficients.add(coefficient_index),
                    *updates.add(prior_update_index),
                )
            };
            sum = fp32_add_rn(sum, fp32_multiply_rn(coefficient, prior_update));
            prior_time += 1;
        }
        value = fp32_subtract_rn(value, sum);
        let update_index = head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + time * SCRATCH_WIDTH + column;
        // SAFETY: One CTA/thread owns each head/column and writes each time row once.
        unsafe { updates.add(update_index).write(value) };
        time += 1;
    }
}

/// Form chunk outputs and the final recurrent state from solved updates.
///
/// Output uses `D(t,-1) * Q[t]^T * S0 + sum(j <= t,
/// D(t,j) * dot(Q[t],K[j]) * u[j])`; final state uses
/// `D(last,-1) * S0 + sum(j, D(last,j) * K[j] * u[j]^T)`. Output is rounded
/// to BF16 RNE only after the FP32 result is stored in `unrounded`. The two
/// work ranges are flat and each thread owns one output or state matrix element.
///
/// # Safety
/// Launch `grid = [value_heads, ceil((rows*width + width*width)/256), 1]` and
/// `block = [256, 1, 1]`, with the same bounded row/head/width constraints as
/// `gdn_chunk_prepare`. `q` and `k` must each cover `rows*key_heads*width`
/// readable `f32` values; `initial_state` must cover `value_heads*width*width`
/// readable `f32`; `decay_products` must cover `value_heads*16*17` readable
/// `f32`; `updates` must cover `value_heads*16*128` readable `f32`; `out` must
/// cover `rows*value_heads*width` writable BF16 values; `unrounded` must cover
/// that many writable `f32` values; and `final_state` must cover
/// `value_heads*width*width` writable `f32` values. Inputs, scratch, output and
/// state buffers must have checked extents, required alignment, pairwise
/// disjoint storage and lifetimes through completion. Host validation must
/// reject nonfinite/overflowing inputs and outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_chunk_finish(
    q: *const f32,
    k: *const f32,
    initial_state: *const f32,
    decay_products: *const f32,
    updates: *const f32,
    out: *mut u16,
    unrounded: *mut f32,
    final_state: *mut f32,
    rows: u32,
    key_heads: u32,
    value_heads: u32,
    width: u32,
) {
    let (value_head, tile, thread) = cta_tile_thread();
    if value_head >= value_heads
        || key_heads == 0
        || value_heads == 0
        || rows == 0
        || rows as usize > MAX_CHUNK_ROWS
        || width == 0
        || width as usize > MAX_WIDTH
    {
        return;
    }

    let head = value_head as usize;
    let rows = rows as usize;
    let key_heads = key_heads as usize;
    let value_heads = value_heads as usize;
    let width = width as usize;
    let key_head = head / (value_heads / key_heads);
    let output_count = rows * width;
    let state_count = width * width;
    let total = output_count + state_count;
    let flat = tile as usize * PREPARE_THREADS as usize + thread as usize;
    if flat >= total {
        return;
    }

    let initial_state_start = head * width * width;
    if flat < output_count {
        let time = flat / width;
        let value_dimension = flat % width;
        let key_start = (time * key_heads + key_head) * width;
        let state_dot =
            // SAFETY: This output worker owns a valid input query and initial-state column.
            unsafe { dot_state_column(q, key_start, initial_state, initial_state_start, width, value_dimension) };
        let decay_index =
            head * MAX_CHUNK_ROWS * DECAY_STRIDE + time * DECAY_STRIDE + MAX_CHUNK_ROWS;
        // SAFETY: The preparation stage initialized this valid row's prefix product.
        let prefix = unsafe { *decay_products.add(decay_index) };
        let mut result = fp32_multiply_rn(prefix, state_dot);
        let mut key_time = 0;
        while key_time <= time {
            let other_start = (key_time * key_heads + key_head) * width;
            // SAFETY: Both Q and K arrays cover every row and mapped key head.
            let qk_dot = unsafe { dot_keys(q, key_start, k, other_start, width) };
            let product_index =
                head * MAX_CHUNK_ROWS * DECAY_STRIDE + time * DECAY_STRIDE + key_time;
            let update_index =
                head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + key_time * SCRATCH_WIDTH + value_dimension;
            // SAFETY: These entries were initialized by preparation and triangular solve.
            let (decay, update) = unsafe {
                (
                    *decay_products.add(product_index),
                    *updates.add(update_index),
                )
            };
            let term = fp32_multiply_rn(fp32_multiply_rn(decay, qk_dot), update);
            result = fp32_add_rn(result, term);
            key_time += 1;
        }

        let output_index = time * value_heads * width + head * width + value_dimension;
        // SAFETY: The flat output worker owns this distinct output and unrounded slot.
        unsafe {
            unrounded.add(output_index).write(result);
            out.add(output_index).write(encode_bf16_rne(result));
        }
        return;
    }

    let state_flat = flat - output_count;
    let key_dimension = state_flat / width;
    let value_dimension = state_flat % width;
    let last_time = rows - 1;
    let final_prefix_index =
        head * MAX_CHUNK_ROWS * DECAY_STRIDE + last_time * DECAY_STRIDE + MAX_CHUNK_ROWS;
    // SAFETY: The preparation stage initialized the last row's prefix product.
    let prefix = unsafe { *decay_products.add(final_prefix_index) };
    // SAFETY: The state worker selects one valid initial-state element.
    let initial = unsafe {
        *initial_state.add(initial_state_start + key_dimension * width + value_dimension)
    };
    let mut update_sum = 0.0_f32;
    let mut time = 0;
    while time < rows {
        let key_start = (time * key_heads + key_head) * width;
        let decay_index = head * MAX_CHUNK_ROWS * DECAY_STRIDE + last_time * DECAY_STRIDE + time;
        let update_index =
            head * MAX_CHUNK_ROWS * SCRATCH_WIDTH + time * SCRATCH_WIDTH + value_dimension;
        // SAFETY: The valid `[last_time,time]` decay and time update entries are initialized.
        let (decay, key_value, update) = unsafe {
            (
                *decay_products.add(decay_index),
                *k.add(key_start + key_dimension),
                *updates.add(update_index),
            )
        };
        let term = fp32_multiply_rn(fp32_multiply_rn(decay, key_value), update);
        update_sum = fp32_add_rn(update_sum, term);
        time += 1;
    }
    let updated = fp32_add_rn(fp32_multiply_rn(prefix, initial), update_sum);
    // SAFETY: Each flat state worker owns one final-state matrix element.
    unsafe {
        final_state
            .add(initial_state_start + key_dimension * width + value_dimension)
            .write(updated);
    }
}
