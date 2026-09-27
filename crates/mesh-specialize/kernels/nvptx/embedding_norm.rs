use core::arch::asm;

const BLOCK_THREADS: u32 = 256;

#[inline(always)]
fn thread_and_row() -> (u32, u32) {
    let thread: u32;
    let row: u32;
    // SAFETY: Reads the calling thread's coordinates without changing memory.
    unsafe {
        asm!(
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {row}, %ctaid.x;",
            thread = out(reg32) thread,
            row = out(reg32) row,
            options(nomem, nostack),
        )
    };
    (thread, row)
}

#[inline(always)]
fn embedding_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares this CTA's 256-element partial array and returns its shared address.
    unsafe {
        asm!(
            ".shared .align 4 .b8 embedding_partials[1024];",
            "mov.u32 {base}, embedding_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_partial(base: u32, index: u32, value: f32) {
    // SAFETY: The kernel uses indices 0..256 in this CTA's 1024-byte shared array.
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
    // SAFETY: The kernel uses indices 0..256 in this CTA's 1024-byte shared array.
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
fn reduction_barrier() {
    // SAFETY: Every thread in the 256-thread CTA reaches each reduction barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn fp32_add_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {result}, {left}, {right};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    result
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
fn fp32_sum_square_rn(sum: f32, value: f32) -> f32 {
    let square = fp32_multiply_rn(value, value);
    fp32_add_rn(sum, square)
}

#[inline(always)]
fn inverse_rms_rn(sum_squares: f32, width: u32, epsilon: f32) -> f32 {
    let factor: f32;
    // SAFETY: Kernel preconditions make width nonzero and epsilon finite and positive.
    unsafe {
        asm!(
            "div.rn.f32 {mean}, {sum_squares}, {width};",
            "add.rn.f32 {denominator}, {mean}, {epsilon};",
            "sqrt.rn.f32 {root}, {denominator};",
            "div.rn.f32 {factor}, {one}, {root};",
            mean = out(reg32) _,
            denominator = out(reg32) _,
            root = out(reg32) _,
            factor = out(reg32) factor,
            sum_squares = in(reg32) sum_squares,
            width = in(reg32) (width as f32),
            epsilon = in(reg32) epsilon,
            one = in(reg32) 1.0_f32,
            options(nomem, nostack),
        )
    };
    factor
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

/// Look up BF16 embeddings, preserve their source bits, and apply zero-centered RMSNorm.
///
/// For each row, this kernel gathers the token's BF16 embedding and copies those exact
/// bits to `residual`. It then computes `decoded * rsqrt(mean(decoded^2) + epsilon)`
/// and multiplies by `1 + decoded_weight`, writing the FP32 result and its BF16
/// round-to-nearest-even encoding.
///
/// # Safety
/// Launch a 1D grid with one 256-thread block per token row (`block = [256, 1, 1]`;
/// `grid = [rows, 1, 1]`). `width` must be 1..=32768 and `epsilon` must be finite and
/// positive. `tokens` must contain at least `rows` readable `u32` indices; the host
/// must validate every index against the vocabulary and ensure the row-major BF16
/// `table` contains at least `vocabulary * width` values. `weight` must contain at
/// least `width` readable BF16 values. `residual` and `normalized` must each contain
/// at least `rows * width` writable BF16 values, and `unrounded` must contain at
/// least `rows * width` writable `f32` values. All device pointers must be correctly
/// aligned, live until kernel completion, and non-overlapping with one another so
/// concurrent rows cannot race and writes cannot alter inputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn embedding_norm_bf16(
    table: *const u16,
    tokens: *const u32,
    weight: *const u16,
    residual: *mut u16,
    normalized: *mut u16,
    unrounded: *mut f32,
    width: u32,
    epsilon: f32,
) {
    let shared = embedding_partials_base();
    let (thread, row) = thread_and_row();
    let width_usize = width as usize;
    let output_row_start = row as usize * width_usize;

    // SAFETY: The launch contract provides one valid token index for each grid row.
    let token = unsafe { *tokens.add(row as usize) };
    let table_row_start = token as usize * width_usize;

    let mut partial = 0.0_f32;
    let mut column = thread;
    while column < width {
        let table_index = table_row_start + column as usize;
        // SAFETY: The host-validated token and width keep this index inside the table.
        let embedding_bits = unsafe { *table.add(table_index) };
        let decoded = decode_bf16(embedding_bits);
        // SAFETY: Each CTA owns its row, and threads write distinct residual columns.
        unsafe {
            residual
                .add(output_row_start + column as usize)
                .write(embedding_bits)
        };
        partial = fp32_sum_square_rn(partial, decoded);
        column += BLOCK_THREADS;
    }
    store_partial(shared, thread, partial);
    reduction_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_partial(shared, thread);
            let right = load_partial(shared, thread + stride);
            store_partial(shared, thread, fp32_add_rn(left, right));
        }
        reduction_barrier();
        stride /= 2;
    }

    let sum_squares = load_partial(shared, 0);
    let factor = inverse_rms_rn(sum_squares, width, epsilon);
    column = thread;
    while column < width {
        let table_index = table_row_start + column as usize;
        let output_index = output_row_start + column as usize;
        // SAFETY: Table and weight extents cover each column for every valid token row.
        let (embedding_bits, weight_bits) =
            unsafe { (*table.add(table_index), *weight.add(column as usize)) };
        let decoded = decode_bf16(embedding_bits);
        let decoded_weight = decode_bf16(weight_bits);
        let normalized_value = fp32_multiply_rn(decoded, factor);
        let centered_weight = fp32_add_rn(1.0_f32, decoded_weight);
        let output_value = fp32_multiply_rn(normalized_value, centered_weight);
        // SAFETY: Each CTA owns its output row and each thread writes distinct columns.
        unsafe {
            unrounded.add(output_index).write(output_value);
            normalized
                .add(output_index)
                .write(encode_bf16_rne(output_value));
        }
        column += BLOCK_THREADS;
    }
}
