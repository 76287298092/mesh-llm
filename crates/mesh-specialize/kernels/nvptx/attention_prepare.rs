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
fn partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one CTA-local FP32 partial for every block thread.
    unsafe {
        asm!(
            ".shared .align 4 .b8 attention_prepare_partials[1024];",
            "mov.u32 {base}, attention_prepare_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_partial(base: u32, index: u32, value: f32) {
    // SAFETY: Callers use indices 0..256 in this CTA's 1024-byte shared array.
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
    // SAFETY: Callers use indices 0..256 in this CTA's 1024-byte shared array.
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
    // SAFETY: All 256 threads reach every reduction barrier uniformly.
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
fn rsqrt_approx(value: f32) -> f32 {
    let inverse: f32;
    // SAFETY: This scalar approximate reciprocal square root has no memory or stack effects.
    unsafe {
        asm!(
            "rsqrt.approx.f32 {inverse}, {value};",
            inverse = out(reg32) inverse,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    inverse
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

#[inline(always)]
fn inverse_rms(sum_squares: f32, width: u32, epsilon: f32) -> f32 {
    let mean = divide_rn(sum_squares, width as f32);
    rsqrt_approx(add_rn(mean, epsilon))
}

#[inline(always)]
fn normalized_value(input_bits: u16, weight_bits: u16, inverse: f32) -> f32 {
    let scaled_input = multiply_rn(decode_bf16(input_bits), inverse);
    let centered_weight = add_rn(1.0_f32, decode_bf16(weight_bits));
    multiply_rn(scaled_input, centered_weight)
}

/// Read and normalize one BF16 channel from the current Q head.
///
/// # Safety
/// `input_start + column` must address a readable Q value and `column` must
/// address a readable weight value. Both pointers must be aligned and live.
#[inline(always)]
unsafe fn normalized_channel(
    input: *const u16,
    weight: *const u16,
    input_start: usize,
    column: usize,
    inverse: f32,
) -> (f32, u16) {
    // SAFETY: The caller guarantees both indices are within their BF16 allocations.
    let (input_bits, weight_bits) =
        unsafe { (*input.add(input_start + column), *weight.add(column)) };
    let value = normalized_value(input_bits, weight_bits, inverse);
    (value, encode_bf16_rn(value))
}

#[inline(always)]
fn rounded_product(left: u16, right: u16) -> f32 {
    decode_bf16(encode_bf16_rn(multiply_rn(
        decode_bf16(left),
        decode_bf16(right),
    )))
}

/// Normalize each Q head, preserve optional raw gates, and apply split-half RoPE.
///
/// The BF16 `input` has shape `[rows, heads, width * (1 + with_gate)]`; each
/// head contains Q followed by its optional gate. The shared `weight` has
/// `width` BF16 elements. `cos` and `sin` contain `[rows, rotary_dim / 2]` BF16
/// values, shared by all heads in each row. `normalized` records the rounded
/// pre-RoPE values, `unrounded` records their FP32 values, and `output` contains
/// the final BF16 values. When `with_gate == 1`, `gate` receives an exact BF16
/// copy of the raw gate channels; otherwise its contents are left untouched.
///
/// # Safety
/// Launch `grid = [rows * heads, 1, 1]` and `block = [256, 1, 1]`. Require
/// `rows` in `1..=2048`, `heads` in `1..=128`, `width` in `2..=1024`, an even
/// `rotary_dim` in `2..=width`, and `with_gate` equal to zero or one. `input`
/// must cover `rows * heads * width * (1 + with_gate)` readable BF16 values;
/// `weight` must cover `width` readable BF16 values; `cos` and `sin` must each
/// cover `rows * (rotary_dim / 2)` readable BF16 values. `output`, `normalized`,
/// and `gate` must each cover `rows * heads * width` writable BF16 values;
/// `unrounded` must cover that many writable FP32 values. All pointers must be
/// correctly aligned, mutually disjoint, and remain live through completion.
/// All index arithmetic must fit `usize`. The host must validate finite Q,
/// weight, cosine, sine, and epsilon values, with epsilon positive and finite.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_qk_prepare(
    input: *const u16,
    weight: *const u16,
    cos: *const u16,
    sin: *const u16,
    output: *mut u16,
    normalized: *mut u16,
    unrounded: *mut f32,
    gate: *mut u16,
    rows: u32,
    heads: u32,
    width: u32,
    rotary_dim: u32,
    with_gate: u32,
    epsilon: f32,
) {
    // SAFETY: The entry's contract is exactly the shared body's contract.
    unsafe {
        prepare_body(
            input, weight, cos, sin, output, normalized, unrounded, gate, rows, heads, width,
            rotary_dim, with_gate, epsilon,
        );
    }
}

/// Shared exact body; callers uphold `attention_qk_prepare`'s contract.
#[inline(always)]
pub(super) unsafe fn prepare_body(
    input: *const u16,
    weight: *const u16,
    cos: *const u16,
    sin: *const u16,
    output: *mut u16,
    normalized: *mut u16,
    unrounded: *mut f32,
    gate: *mut u16,
    rows: u32,
    heads: u32,
    width: u32,
    rotary_dim: u32,
    with_gate: u32,
    epsilon: f32,
) {
    let (block, thread) = block_and_thread();
    if block >= rows * heads {
        return;
    }

    let partials = partials_base();
    let width_usize = width as usize;
    let block_usize = block as usize;
    let input_head_width = width_usize * (1 + with_gate as usize);
    let input_start = block_usize * input_head_width;
    let mut partial = 0.0_f32;
    let mut column = thread;
    while column < width {
        // SAFETY: The exact grid and validated dimensions place this Q element in range.
        let input_bits = unsafe { *input.add(input_start + column as usize) };
        let value = decode_bf16(input_bits);
        partial = add_rn(partial, multiply_rn(value, value));
        column += BLOCK_THREADS;
    }
    store_partial(partials, thread, partial);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_partial(partials, thread);
            let right = load_partial(partials, thread + stride);
            store_partial(partials, thread, add_rn(left, right));
        }
        block_barrier();
        stride /= 2;
    }

    let inverse = inverse_rms(load_partial(partials, 0), width, epsilon);
    let output_start = block_usize * width_usize;
    let row = (block / heads) as usize;
    let half = (rotary_dim / 2) as usize;
    column = thread;
    while column < width {
        let column_usize = column as usize;
        // SAFETY: This CTA owns a Q value and the corresponding shared weight element.
        let (normalized_value, normalized_bits) =
            unsafe { normalized_channel(input, weight, input_start, column_usize, inverse) };
        let output_index = output_start + column_usize;
        // SAFETY: This CTA/thread owns the distinct compact diagnostic output element.
        unsafe { unrounded.add(output_index).write(normalized_value) };
        // SAFETY: This CTA/thread owns the distinct compact normalized BF16 output element.
        unsafe { normalized.add(output_index).write(normalized_bits) };

        if with_gate == 1 {
            // SAFETY: The validated per-head input has a gate value after its Q channels.
            let gate_bits = unsafe { *input.add(input_start + width_usize + column_usize) };
            // SAFETY: This CTA/thread owns the distinct compact gate output element.
            unsafe { gate.add(output_index).write(gate_bits) };
        }

        let output_bits = if column < rotary_dim {
            let rotary_column = column_usize % half;
            let partner_column = if column_usize < half {
                column_usize + half
            } else {
                column_usize - half
            };
            // SAFETY: The partner is within the first rotary_dim Q channels and its weight.
            let (_, partner_bits) =
                unsafe { normalized_channel(input, weight, input_start, partner_column, inverse) };
            let trig_index = row * half + rotary_column;
            // SAFETY: The valid row/rotation bounds cover one cosine and sine BF16 value.
            let (cosine_bits, sine_bits) = unsafe { (*cos.add(trig_index), *sin.add(trig_index)) };
            let current_cosine = rounded_product(normalized_bits, cosine_bits);
            let partner_sine = rounded_product(partner_bits, sine_bits);
            let rotated = if column_usize < half {
                subtract_rn(current_cosine, partner_sine)
            } else {
                add_rn(current_cosine, partner_sine)
            };
            encode_bf16_rn(rotated)
        } else {
            normalized_bits
        };
        // SAFETY: This CTA/thread owns the distinct compact final BF16 output element.
        unsafe { output.add(output_index).write(output_bits) };
        column += BLOCK_THREADS;
    }
}
