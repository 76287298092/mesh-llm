use core::arch::asm;

#[inline(always)]
fn value_head_and_column() -> (u32, u32) {
    let value_head: u32;
    let column: u32;
    // SAFETY: Reads the calling thread's CTA and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {value_head}, %ctaid.x;",
            "mov.u32 {column}, %tid.x;",
            value_head = out(reg32) value_head,
            column = out(reg32) column,
            options(nomem, nostack),
        )
    };
    (value_head, column)
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

/// Apply the Qwen GDN recurrent matrix update with one owner per state column.
///
/// A CTA owns one value head and each thread owns one value column for that head.
/// State is laid out row-major as `[value_heads, width, width]`; Q and K are
/// `[rows, key_heads, width]` with unrepeated key heads, while QKV is time-major
/// with Q, K, then V sections. Value head `h` maps to key head
/// `h / (value_heads / key_heads)`. Output and `unrounded` are time-major
/// `[rows, value_heads, width]`. Each row decays and predicts the entire owned
/// state column in increasing key order, computes the gated delta, then updates
/// state and accumulates the output in increasing key order. State is mutated in
/// place across rows, with FP32 output also recorded before BF16 rounding.
///
/// # Safety
/// Launch `grid = [value_heads, 1, 1]` and `block = [width, 1, 1]`. `rows` must be
/// 1..=2048, `key_heads` 1..=64, `value_heads` 1..=256 and divisible by
/// `key_heads`, and `width` must be a power of two in 1..=256. `q` and `k` must each
/// cover `rows * key_heads * width` readable `f32` values. `qkv` must cover
/// `rows * (2 * key_heads + value_heads) * width` readable BF16 values, with V
/// beginning after the Q and K sections. `beta` and `decay` must each cover
/// `rows * value_heads` readable BF16 and `f32` values. `state` must cover
/// `value_heads * width * width` readable and writable `f32` values; `out` and
/// `unrounded` must each cover `rows * value_heads * width` writable BF16 and `f32`
/// values. All extents and index arithmetic must fit `usize`; pointers must be
/// correctly aligned, mutually disjoint, and live until completion. The host must
/// validate all read values are finite and reject non-finite or overflowing outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_recurrent(
    q: *const f32,
    k: *const f32,
    qkv: *const u16,
    beta: *const u16,
    decay: *const f32,
    state: *mut f32,
    out: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    key_heads: u32,
    value_heads: u32,
    width: u32,
) {
    let (value_head, column) = value_head_and_column();
    if value_head >= value_heads || column >= width {
        return;
    }

    let value_head = value_head as usize;
    let column = column as usize;
    let key_heads = key_heads as usize;
    let value_heads = value_heads as usize;
    let width = width as usize;
    let rows = rows as usize;
    let key_head = value_head / (value_heads / key_heads);
    let state_head_start = value_head * width * width;
    let input_row_width = (2 * key_heads + value_heads) * width;
    let output_row_width = value_heads * width;

    let mut row = 0;
    while row < rows {
        let qk_head_start = (row * key_heads + key_head) * width;
        let qkv_row_start = row * input_row_width;
        let v_start = qkv_row_start + (2 * key_heads + value_head) * width + column;
        let gate_index = row * value_heads + value_head;
        // SAFETY: Row/head indices and host-validated tensor extents cover these inputs.
        let (value, beta_value, decay_value) = unsafe {
            (
                decode_bf16(*qkv.add(v_start)),
                decode_bf16(*beta.add(gate_index)),
                *decay.add(gate_index),
            )
        };

        let mut prediction = 0.0_f32;
        let mut key_index = 0;
        while key_index < width {
            let state_index = state_head_start + key_index * width + column;
            // SAFETY: This CTA/thread exclusively owns this state column.
            let decayed_state = unsafe { fp32_multiply_rn(*state.add(state_index), decay_value) };
            // SAFETY: Q/K tensors contain the mapped unrepeated key head for this row.
            let key_value = unsafe { *k.add(qk_head_start + key_index) };
            let prediction_term = fp32_multiply_rn(decayed_state, key_value);
            prediction = fp32_add_rn(prediction, prediction_term);
            // SAFETY: This CTA/thread exclusively owns this state column.
            unsafe { state.add(state_index).write(decayed_state) };
            key_index += 1;
        }

        let prediction_delta = fp32_subtract_rn(value, prediction);
        let delta = fp32_multiply_rn(prediction_delta, beta_value);
        let mut accumulator = 0.0_f32;
        key_index = 0;
        while key_index < width {
            let state_index = state_head_start + key_index * width + column;
            // SAFETY: This CTA/thread exclusively owns this state column.
            let current_state = unsafe { *state.add(state_index) };
            // SAFETY: Q/K tensors contain the mapped unrepeated key head for this row.
            let key_value = unsafe { *k.add(qk_head_start + key_index) };
            let update = fp32_multiply_rn(key_value, delta);
            let updated_state = fp32_add_rn(current_state, update);
            // SAFETY: This CTA/thread exclusively owns this state column.
            unsafe { state.add(state_index).write(updated_state) };

            // SAFETY: Q contains the mapped unrepeated key head for this row.
            let query_value = unsafe { *q.add(qk_head_start + key_index) };
            let output_term = fp32_multiply_rn(updated_state, query_value);
            accumulator = fp32_add_rn(accumulator, output_term);
            key_index += 1;
        }

        let output_index = row * output_row_width + value_head * width + column;
        // SAFETY: Each CTA/thread writes one distinct output element in each extent.
        unsafe {
            unrounded.add(output_index).write(accumulator);
            out.add(output_index).write(encode_bf16_rne(accumulator));
        }
        row += 1;
    }
}
