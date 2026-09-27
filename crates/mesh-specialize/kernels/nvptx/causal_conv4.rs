use core::arch::asm;

const BLOCK_THREADS: usize = 256;
const LOG2_E: f32 = core::f32::consts::LOG2_E;

#[inline(always)]
fn linear_thread_index() -> usize {
    let block: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's 1D block and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {block}, %ctaid.x;",
            "mov.u32 {thread}, %tid.x;",
            block = out(reg32) block,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    block as usize * BLOCK_THREADS + thread as usize
}

/// Load one BF16 value from the concatenation of the three history rows and input rows.
///
/// # Safety
/// `history` must cover three time-major rows and `input` must cover `rows` time-major
/// rows, each with `channels` values. `time` must be less than `rows + 3`, `channel`
/// less than `channels`, and all index arithmetic must fit `usize`.
#[inline(always)]
unsafe fn load_window_value(
    input: *const u16,
    history: *const u16,
    time: usize,
    channel: usize,
    _rows: usize,
    channels: usize,
) -> u16 {
    if time < 3 {
        // SAFETY: The caller provides the three-row history extent and valid channel index.
        unsafe { *history.add(time * channels + channel) }
    } else {
        // SAFETY: The virtual time maps to an input row in 0..rows.
        unsafe { *input.add((time - 3) * channels + channel) }
    }
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
fn fp32_exp2_approx_ftz(exponent: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar approximate exponential has no memory or stack effects.
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
fn stable_silu(value: f32) -> f32 {
    let absolute = f32::from_bits(value.to_bits() & 0x7fff_ffff);
    let negative_absolute = f32::from_bits(absolute.to_bits() | 0x8000_0000);
    let exponent = fp32_multiply_rn(negative_absolute, LOG2_E);
    let exponential = if exponent < -126.0_f32 {
        0.0_f32
    } else {
        fp32_exp2_approx_ftz(exponent)
    };
    let denominator = fp32_add_rn(1.0_f32, exponential);
    let sigmoid = if value >= 0.0_f32 {
        fp32_divide_rn(1.0_f32, denominator)
    } else {
        fp32_divide_rn(exponential, denominator)
    };
    fp32_multiply_rn(value, sigmoid)
}

/// Fuse a causal four-tap BF16 convolution with BF16-rounded SiLU and state update.
///
/// `input` and `output` are time-major `[rows, channels]`; `weight` is channel-major
/// `[channels, 4]`; and `history` and `next_history` are time-major `[3, channels]`,
/// ordered oldest to newest. For each row and channel, this computes the four-tap
/// convolution in ascending tap order with no bias, records its FP32 result, rounds
/// once to BF16, decodes that rounded value, then applies stable SiLU and records both
/// its FP32 and BF16 results. The last row's channel owners also shift the raw input
/// and history values into `next_history`.
///
/// # Safety
/// Launch a 1D grid with `ceil(rows * channels / 256)` blocks and `block = [256, 1, 1]`.
/// `rows` must be 1..=2048 and `channels` 1..=32768. `input` and `output` must each
/// cover `rows * channels` readable/writable BF16 values. `weight` must cover
/// `channels * 4` readable BF16 values. `history` and `next_history` must each cover
/// `3 * channels` readable/writable BF16 values. `conv_unrounded` and `silu_unrounded`
/// must each cover `rows * channels` writable `f32` values. All input, state, and
/// output pointers must be correctly aligned for their element types, mutually
/// disjoint, and live until completion. The host must validate all input and weight
/// BF16 values are finite and reject non-finite or overflowing outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn causal_conv4_bf16(
    input: *const u16,
    weight: *const u16,
    history: *const u16,
    next_history: *mut u16,
    output: *mut u16,
    conv_unrounded: *mut f32,
    silu_unrounded: *mut f32,
    rows: u32,
    channels: u32,
) {
    let flat_index = linear_thread_index();
    let rows_usize = rows as usize;
    let channels_usize = channels as usize;
    let output_count = rows_usize * channels_usize;
    if flat_index >= output_count {
        return;
    }

    let row = flat_index / channels_usize;
    let channel = flat_index % channels_usize;
    let mut convolution = 0.0_f32;
    let mut tap = 0;
    while tap < 4 {
        // SAFETY: Every output row's four-tap window lies within history + input.
        let input_value = unsafe {
            load_window_value(
                input,
                history,
                row + tap,
                channel,
                rows_usize,
                channels_usize,
            )
        };
        // SAFETY: The weight matrix contains four BF16 taps for every channel.
        let weight_value = unsafe { *weight.add(channel * 4 + tap) };
        let product = fp32_multiply_rn(decode_bf16(input_value), decode_bf16(weight_value));
        convolution = fp32_add_rn(convolution, product);
        tap += 1;
    }

    // SAFETY: Each flat output thread owns one distinct convolution and activation slot.
    unsafe { conv_unrounded.add(flat_index).write(convolution) };
    let rounded_convolution = encode_bf16_rne(convolution);
    let activation_input = decode_bf16(rounded_convolution);
    let silu = stable_silu(activation_input);
    // SAFETY: Each flat output thread owns one distinct activation slot and output value.
    unsafe {
        silu_unrounded.add(flat_index).write(silu);
        output.add(flat_index).write(encode_bf16_rne(silu));
    }

    if row + 1 == rows_usize {
        let mut slot = 0;
        while slot < 3 {
            // SAFETY: Virtual window positions rows..rows+2 are within history + input.
            let value = unsafe {
                load_window_value(
                    input,
                    history,
                    rows_usize + slot,
                    channel,
                    rows_usize,
                    channels_usize,
                )
            };
            // SAFETY: The last-row owner writes this channel's unique history slots.
            unsafe {
                next_history
                    .add(slot * channels_usize + channel)
                    .write(value)
            };
            slot += 1;
        }
    }
}
