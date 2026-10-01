use core::arch::asm;

const BLOCK_THREADS: u32 = 256;
const LOG2_E: f32 = core::f32::consts::LOG2_E;

#[inline(always)]
fn block_and_thread() -> (u32, u32) {
    let block: u32;
    let thread: u32;
    // SAFETY: Reads thread coordinates without memory effects.
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
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline(always)]
fn encode_bf16(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Converts one FP32 register to BF16 with round-to-nearest-even.
    unsafe {
        asm!(
            "cvt.rn.bf16.f32 {bits}, {value};",
            bits = out(reg16) bits,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    bits
}

#[inline(always)]
fn source_exp(value: f32) -> f32 {
    let exponent: f32;
    let scaled = value * LOG2_E;
    // SAFETY: Approximate exponentiation reads and writes registers only.
    unsafe {
        asm!(
            "ex2.approx.ftz.f32 {exponent}, {scaled};",
            exponent = out(reg32) exponent,
            scaled = in(reg32) scaled,
            options(nomem, nostack),
        )
    };
    exponent
}

#[inline(always)]
fn source_sigmoid(value: f32) -> f32 {
    let denominator = 1.0_f32 + source_exp(-value);
    let result: f32;
    // SAFETY: The scalar division has no memory or stack effects.
    unsafe {
        asm!(
            "div.rn.f32 {result}, {one}, {denominator};",
            result = out(reg32) result,
            one = in(reg32) 1.0_f32,
            denominator = in(reg32) denominator,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn source_silu(value: f32) -> f32 {
    let denominator = 1.0_f32 + source_exp(-value);
    let result: f32;
    // SAFETY: The scalar division has no memory or stack effects.
    unsafe {
        asm!(
            "div.rn.f32 {result}, {value}, {denominator};",
            result = out(reg32) result,
            value = in(reg32) value,
            denominator = in(reg32) denominator,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn multiply_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: The scalar multiply has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f32 {result}, {left}, {right};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_attention_gate(
    attention: *const u16,
    gate: *const u16,
    output: *mut u16,
    count: u32,
) {
    let (block, thread) = block_and_thread();
    let index = block * BLOCK_THREADS + thread;
    if index >= count {
        return;
    }
    let index = index as usize;
    // SAFETY: The host-provided equal-sized BF16 vectors cover this guarded index.
    let (attention_value, gate_value) = unsafe {
        (
            decode_bf16(*attention.add(index)),
            decode_bf16(*gate.add(index)),
        )
    };
    let product = multiply_rn(attention_value, source_sigmoid(gate_value));
    // SAFETY: Each in-range lane uniquely owns this BF16 output element.
    unsafe { output.add(index).write(encode_bf16(product)) };
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_silu_mul(
    gate: *const u16,
    up: *const u16,
    output: *mut u16,
    count: u32,
) {
    let (block, thread) = block_and_thread();
    let index = block * BLOCK_THREADS + thread;
    if index >= count {
        return;
    }
    let index = index as usize;
    // SAFETY: The host-provided equal-sized BF16 vectors cover this guarded index.
    let (gate_value, up_value) =
        unsafe { (decode_bf16(*gate.add(index)), decode_bf16(*up.add(index))) };
    let product = multiply_rn(source_silu(gate_value), up_value);
    // SAFETY: Each in-range lane uniquely owns this BF16 output element.
    unsafe { output.add(index).write(encode_bf16(product)) };
}
