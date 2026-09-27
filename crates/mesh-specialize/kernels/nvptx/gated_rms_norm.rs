use core::arch::asm;

const BLOCK_THREADS: u32 = 256;

#[inline(always)]
fn block_and_thread() -> (u32, u32) {
    let group: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's block and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {group}, %ctaid.x;",
            "mov.u32 {thread}, %tid.x;",
            group = out(reg32) group,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    (group, thread)
}

#[inline(always)]
fn shared_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one CTA-local array with a slot for every block thread.
    unsafe {
        asm!(
            ".shared .align 4 .b8 gated_norm_partials[1024];",
            "mov.u32 {base}, gated_norm_partials;",
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
fn block_barrier() {
    // SAFETY: All 256 threads reach every reduction barrier uniformly.
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

/// Apply Qwen's gated RMSNorm in FP32 with the model's BF16 rounding points.
///
/// `x` and `z` are row-major `[groups, width]` BF16 inputs. The shared `weight`
/// vector is direct gamma, with no added one. Each row is RMS-normalized first,
/// then its FP32 normalized values are recorded, BF16-rounded, multiplied by gamma,
/// and rounded to BF16 as `weighted`. The shared FP64 reference-profile SiLU from
/// FP32 `z` is converted once to FP32; the final FP32 result multiplies decoded
/// `weighted` by that value before BF16 rounding.
///
/// # Safety
/// Launch exactly `grid = [groups, 1, 1]` and `block = [256, 1, 1]`. `groups` must
/// be in 1..=131072, `width` must be a power of two in 1..=256, and `epsilon` must
/// be positive and finite. `x` and `z` must each cover `groups * width` readable
/// BF16 values; `weight` must cover `width` readable BF16 values. `output` and
/// `weighted` must each cover `groups * width` writable BF16 values; `normalized`,
/// `silu`, and `unrounded` must each cover `groups * width` writable `f32` values.
/// Extents and index arithmetic must fit `usize`; pointers must be correctly aligned,
/// mutually disjoint, and live until kernel completion. The host must validate finite
/// input BF16 values and reject non-finite or overflowing intermediate and output
/// values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_gated_rms_norm(
    x: *const u16,
    z: *const u16,
    weight: *const u16,
    output: *mut u16,
    normalized: *mut f32,
    weighted: *mut u16,
    silu: *mut f32,
    unrounded: *mut f32,
    groups: u32,
    width: u32,
    epsilon: f32,
) {
    let shared = shared_partials_base();
    let (group, thread) = block_and_thread();
    if group >= groups {
        return;
    }
    let width_usize = width as usize;
    let row_start = group as usize * width_usize;

    let mut x_value = 0.0_f32;
    if thread < width {
        // SAFETY: Exact grid and validated dimensions cover this row and column.
        x_value = unsafe { decode_bf16(*x.add(row_start + thread as usize)) };
    }
    let square = fp32_multiply_rn(x_value, x_value);
    store_shared_partial(shared, thread, square);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_shared_partial(shared, thread);
            let right = load_shared_partial(shared, thread + stride);
            store_shared_partial(shared, thread, fp32_add_rn(left, right));
        }
        block_barrier();
        stride /= 2;
    }

    if thread >= width {
        return;
    }

    let column = thread as usize;
    let index = row_start + column;
    let sum_squares = load_shared_partial(shared, 0);
    let mean = fp32_divide_rn(sum_squares, width as f32);
    let denominator = fp32_add_rn(mean, epsilon);
    let inverse_rms = fp32_divide_rn(1.0_f32, fp32_sqrt_rn(denominator));

    // SAFETY: The active thread owns this input/output column and the shared gamma element.
    let (gate, gamma) = unsafe { (decode_bf16(*z.add(index)), decode_bf16(*weight.add(column))) };
    let normalized_value = fp32_multiply_rn(x_value, inverse_rms);
    let normalized_bf16 = encode_bf16_rne(normalized_value);
    let rounded_normalized = decode_bf16(normalized_bf16);
    let weighted_value = fp32_multiply_rn(gamma, rounded_normalized);
    let weighted_bf16 = encode_bf16_rne(weighted_value);
    let silu_value = super::silu::silu(gate);
    let result = fp32_multiply_rn(decode_bf16(weighted_bf16), silu_value);

    // SAFETY: Each active thread writes its unique element in all output extents.
    unsafe {
        normalized.add(index).write(normalized_value);
        weighted.add(index).write(weighted_bf16);
        silu.add(index).write(silu_value);
        unrounded.add(index).write(result);
        output.add(index).write(encode_bf16_rne(result));
    }
}
