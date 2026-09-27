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
fn fp32_exp2_approx(exponent: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar approximate exponential has no memory or stack effects.
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
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
fn stable_silu(value: f32) -> f32 {
    let absolute = f32::from_bits(value.to_bits() & 0x7fff_ffff);
    let negative_absolute = f32::from_bits(absolute.to_bits() | 0x8000_0000);
    let exponent = fp32_multiply_rn(negative_absolute, core::f32::consts::LOG2_E);
    let exponential = fp32_exp2_approx(exponent);
    let denominator = fp32_add_rn(1.0_f32, exponential);
    let sigmoid = if value >= 0.0_f32 {
        fp32_divide_rn(1.0_f32, denominator)
    } else {
        fp32_divide_rn(exponential, denominator)
    };
    fp32_multiply_rn(value, sigmoid)
}

/// Compute SiLU on BF16 gate values, round the activation to BF16, then multiply by up.
///
/// `gate` and `up` are BF16 vectors. SiLU is evaluated in FP32 with the same stable
/// profile used by the gated RMSNorm kernel. `activated` records its BF16 rounding
/// boundary; the final product consumes that decoded BF16 activation and is stored
/// both as FP32 diagnostics and BF16 round-to-nearest-even.
///
/// # Safety
/// Launch `grid = [ceil(count / 256), 1, 1]` and `block = [256, 1, 1]`, with
/// `count` in `1..=67108864`. `gate` and `up` must each cover `count` readable BF16
/// values. `out` and `activated` must each cover `count` writable BF16 values;
/// `silu` and `unrounded` must each cover `count` writable FP32 values. Extents and
/// index arithmetic must fit `usize`; pointers must be correctly aligned, pairwise
/// disjoint, and live until completion. The host must validate finite BF16 inputs
/// and reject nonfinite SiLU/product values or BF16 overflow before accepting output.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn mlp_silu_product(
    gate: *const u16,
    up: *const u16,
    out: *mut u16,
    silu: *mut f32,
    activated: *mut u16,
    unrounded: *mut f32,
    count: u32,
) {
    let (group, thread) = block_and_thread();
    let index = group * BLOCK_THREADS + thread;
    if index >= count {
        return;
    }

    let index = index as usize;
    // SAFETY: The launch contract covers each in-range input index and the outputs
    // are disjoint arrays of the same exact element count.
    let (gate_value, up_value) =
        unsafe { (decode_bf16(*gate.add(index)), decode_bf16(*up.add(index))) };
    let silu_value = stable_silu(gate_value);
    let activated_bits = encode_bf16_rne(silu_value);
    let product = fp32_multiply_rn(decode_bf16(activated_bits), up_value);

    // SAFETY: Each in-range thread uniquely owns this index in every output buffer.
    unsafe {
        silu.add(index).write(silu_value);
        activated.add(index).write(activated_bits);
        unrounded.add(index).write(product);
        out.add(index).write(encode_bf16_rne(product));
    }
}
