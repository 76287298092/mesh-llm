use core::arch::asm;

const BLOCK_THREADS: u32 = 256;

#[inline(always)]
fn block_and_thread() -> (u32, u32) {
    let row: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's block and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {row}, %ctaid.x;",
            "mov.u32 {thread}, %tid.x;",
            row = out(reg32) row,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    (row, thread)
}

#[inline(always)]
fn residual_norm_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one CTA-local array with a slot for every block thread.
    unsafe {
        asm!(
            ".shared .align 4 .b8 residual_norm_partials[1024];",
            "mov.u32 {base}, residual_norm_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_shared_partial(base: u32, index: u32, value: f32) {
    // SAFETY: The caller uses indices 0..256 in this CTA's 1024-byte shared array.
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
fn load_shared_partial(base: u32, index: u32) -> f32 {
    let value: f32;
    // SAFETY: The caller uses indices 0..256 in this CTA's 1024-byte shared array.
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

#[inline(always)]
fn fp32_sqrt_rn(value: f32) -> f32 {
    let root: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "sqrt.rn.f32 {root}, {value};",
            root = out(reg32) root,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    root
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
fn fp32_sum_square_rn(sum: f32, value: f32) -> f32 {
    fp32_add_rn(sum, fp32_multiply_rn(value, value))
}

#[inline(always)]
fn inverse_rms_rn(sum_squares: f32, width: u32, epsilon: f32) -> f32 {
    let mean = fp32_divide_rn(sum_squares, width as f32);
    let denominator = fp32_add_rn(mean, epsilon);
    fp32_divide_rn(1.0_f32, fp32_sqrt_rn(denominator))
}

/// Add a BF16 residual and branch with BF16 rounding, then apply zero-centered RMSNorm.
///
/// `residual` and `branch` are row-major `[rows, width]` BF16 values, and `weight`
/// is a shared `[width]` BF16 vector. Each FP32 residual-plus-branch sum is rounded
/// to BF16 before contributing to the RMS reduction. The rounded sum is normalized,
/// multiplied by `1 + weight` in FP32, and rounded to BF16 for `normalized`; the
/// final FP32 value is also written to `unrounded`.
///
/// # Safety
/// Launch exactly `grid = [rows, 1, 1]` and `block = [256, 1, 1]`. `rows` must be
/// 1..=2048, `width` must be 1..=32768, and `epsilon` must be positive and finite.
/// `residual` and `branch` must each cover `rows * width` readable BF16 values;
/// `weight` must cover `width` readable BF16 values. `sum` and `normalized` must
/// each cover `rows * width` writable BF16 values, and `unrounded` must cover
/// `rows * width` writable `f32` values. Extents and index arithmetic must fit
/// `usize`; pointers must be correctly aligned, mutually disjoint, and live until
/// completion. The host must validate finite inputs and reject non-finite or
/// overflowing intermediate and output values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn residual_norm_bf16(
    residual: *const u16,
    branch: *const u16,
    weight: *const u16,
    sum: *mut u16,
    normalized: *mut u16,
    unrounded: *mut f32,
    width: u32,
    epsilon: f32,
) {
    let shared = residual_norm_partials_base();
    let (row, thread) = block_and_thread();
    let width_usize = width as usize;
    let row_start = row as usize * width_usize;

    let mut partial = 0.0_f32;
    let mut column = thread;
    while column < width {
        let index = row_start + column as usize;
        // SAFETY: Exact row grid and validated width cover both BF16 input elements.
        let (residual_bits, branch_bits) = unsafe { (*residual.add(index), *branch.add(index)) };
        let sum_value = fp32_add_rn(decode_bf16(residual_bits), decode_bf16(branch_bits));
        let sum_bits = encode_bf16_rne(sum_value);
        // SAFETY: This row/thread owns the distinct sum output element.
        unsafe { sum.add(index).write(sum_bits) };
        partial = fp32_sum_square_rn(partial, decode_bf16(sum_bits));
        column += BLOCK_THREADS;
    }
    store_shared_partial(shared, thread, partial);
    reduction_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_shared_partial(shared, thread);
            let right = load_shared_partial(shared, thread + stride);
            store_shared_partial(shared, thread, fp32_add_rn(left, right));
        }
        reduction_barrier();
        stride /= 2;
    }

    let factor = inverse_rms_rn(load_shared_partial(shared, 0), width, epsilon);
    column = thread;
    while column < width {
        let index = row_start + column as usize;
        // SAFETY: The first pass wrote every BF16 sum and the shared weight spans width.
        let (sum_bits, weight_bits) = unsafe { (*sum.add(index), *weight.add(column as usize)) };
        let normalized_value = fp32_multiply_rn(decode_bf16(sum_bits), factor);
        let centered_weight = fp32_add_rn(1.0_f32, decode_bf16(weight_bits));
        let output_value = fp32_multiply_rn(normalized_value, centered_weight);
        // SAFETY: Each row/thread writes a distinct normalized and FP32 output element.
        unsafe {
            normalized.add(index).write(encode_bf16_rne(output_value));
            unrounded.add(index).write(output_value);
        }
        column += BLOCK_THREADS;
    }
}
