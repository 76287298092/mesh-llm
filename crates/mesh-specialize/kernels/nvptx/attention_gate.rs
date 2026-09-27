use core::arch::asm;

const BLOCK_THREADS: u32 = 256;
const MAX_COUNT: u32 = 67_108_864;
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
fn exp2_approx(exponent: f32) -> f32 {
    let result: f32;
    // SAFETY: This non-FTZ scalar approximate exponential has no memory or stack effects.
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
fn stable_sigmoid(value: f32) -> f32 {
    let negative_value = f32::from_bits(value.to_bits() ^ 0x8000_0000);
    let exponent_input = if value >= 0.0_f32 {
        negative_value
    } else {
        value
    };
    let exponent = multiply_rn(exponent_input, LOG2_E);
    let exponential = exp2_approx(exponent);
    let denominator = add_rn(1.0_f32, exponential);
    if value >= 0.0_f32 {
        divide_rn(1.0_f32, denominator)
    } else {
        divide_rn(exponential, denominator)
    }
}

/// Apply the stable sigmoid to BF16 gate values and multiply rounded sigmoid by attention.
///
/// `attention` and `gate` are BF16 vectors. The kernel records the FP32 sigmoid,
/// its BF16 round-to-nearest-even boundary, and the FP32 product of decoded rounded
/// sigmoid with the original BF16 attention value. No SiLU multiplication by the
/// gate is performed. FP32 and BF16 arithmetic preserve signed zero in the product.
///
/// # Safety
/// Launch `grid = [ceil(count / 256), 1, 1]` and `block = [256, 1, 1]`, with
/// `count` in `1..=67108864`. `attention` and `gate` must each cover `count`
/// readable BF16 values. `output`, `sigmoid_bf16` must each cover `count` writable
/// BF16 values; `sigmoid_f32` and `product_f32` must each cover `count` writable
/// FP32 values. All extents and index arithmetic must fit `usize`; pointers must
/// be aligned, pairwise disjoint, and live through completion. The host validates
/// finite BF16 inputs and rejects non-finite FP32 diagnostics or output values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_gate_bf16(
    attention: *const u16,
    gate: *const u16,
    output: *mut u16,
    sigmoid_f32: *mut f32,
    sigmoid_bf16: *mut u16,
    product_f32: *mut f32,
    count: u32,
) {
    if count == 0 || count > MAX_COUNT {
        return;
    }
    let (block, thread) = block_and_thread();
    let index = block * BLOCK_THREADS + thread;
    if index >= count {
        return;
    }

    let index = index as usize;
    // SAFETY: The elementwise bounds guard puts this lane's BF16 reads in range.
    let (attention_value, gate_value) = unsafe {
        (
            decode_bf16(*attention.add(index)),
            decode_bf16(*gate.add(index)),
        )
    };
    let sigmoid = stable_sigmoid(gate_value);
    let sigmoid_bits = encode_bf16_rne(sigmoid);
    let product = multiply_rn(decode_bf16(sigmoid_bits), attention_value);
    let output_bits = encode_bf16_rne(product);

    // SAFETY: Every in-range lane uniquely owns this index in all disjoint outputs.
    unsafe {
        sigmoid_f32.add(index).write(sigmoid);
        sigmoid_bf16.add(index).write(sigmoid_bits);
        product_f32.add(index).write(product);
        output.add(index).write(output_bits);
    }
}
