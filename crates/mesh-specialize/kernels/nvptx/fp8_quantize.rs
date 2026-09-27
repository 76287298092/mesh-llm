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
fn partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one 256-element FP32 scratch array for this CTA.
    unsafe {
        asm!(
            ".shared .align 4 .b8 fp8_partials[1024];",
            "mov.u32 {base}, fp8_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_partial(base: u32, index: u32, value: f32) {
    // SAFETY: The caller indexes 0..256 in this CTA's 1024-byte scratch array.
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
    // SAFETY: The caller indexes 0..256 in this CTA's 1024-byte scratch array.
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
    // SAFETY: Every thread in the CTA reaches every reduction barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn fp32_abs_max(left: f32, right: f32) -> f32 {
    // Both inputs are finite and nonnegative, so their positive IEEE bit patterns
    // have the same ordering as their FP32 values, including subnormals.
    if left.to_bits() >= right.to_bits() {
        left
    } else {
        right
    }
}

#[inline(always)]
fn fp32_div_rn(numerator: f32, denominator: f32) -> f32 {
    let quotient: f32;
    // SAFETY: Scalar FP32 division has no memory or stack effects.
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
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn decode_e4m3fn(code: u8) -> f32 {
    let exponent = (code >> 3) as u32;
    let mantissa = (code & 7) as u32;
    if exponent == 0 {
        mantissa as f32 * (1.0_f32 / 512.0_f32)
    } else {
        f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
    }
}

#[inline(always)]
fn encode_e4m3fn_rne(magnitude: f32) -> u8 {
    if magnitude >= 448.0_f32 {
        return 126;
    }

    let mut low = 0_u32;
    let mut high = 126_u32;
    while low < high {
        let middle = low + (high - low) / 2;
        if decode_e4m3fn(middle as u8) < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    let upper = low as u8;
    if upper == 0 {
        return 0;
    }
    let lower = upper - 1;
    let lower_distance = magnitude - decode_e4m3fn(lower);
    let upper_distance = decode_e4m3fn(upper) - magnitude;
    if lower_distance < upper_distance || (lower_distance == upper_distance && lower & 1 == 0) {
        lower
    } else {
        upper
    }
}

/// Quantize one row-major BF16 row per block into finite E4M3 values.
///
/// The output scale is `max(abs(row)) / 448`, with scale one for an all-zero row.
/// FP8 conversion uses a software nearest-even search over finite E4M3 codes.
///
/// # Safety
/// Launch `grid = [rows, 1, 1]` with `rows > 0` and `block = [256, 1, 1]`.
/// `width` must be 1..=32768 and all `row * width + column` indices must fit
/// `usize`. `input` must cover `rows * width` readable BF16 values, `codes` must
/// cover exactly `rows * width` writable bytes, and `scales` must cover exactly
/// `rows` writable FP32 values. The host must reject non-finite BF16 inputs.
/// All three device allocations must be correctly aligned, disjoint, and live
/// until kernel completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_quantize_bf16(
    input: *const u16,
    codes: *mut u8,
    scales: *mut f32,
    width: u32,
) {
    let shared = partials_base();
    let (thread, row) = thread_and_row();
    let width_usize = width as usize;
    let row_start = row as usize * width_usize;

    let mut partial_max = 0.0_f32;
    let mut column = thread;
    while column < width {
        let index = row_start + column as usize;
        // SAFETY: The launch contract provides every BF16 value in each grid row.
        let value = unsafe { decode_bf16(*input.add(index)) };
        let magnitude = f32::from_bits(value.to_bits() & 0x7fff_ffff);
        partial_max = fp32_abs_max(partial_max, magnitude);
        column += BLOCK_THREADS;
    }
    store_partial(shared, thread, partial_max);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_partial(shared, thread);
            let right = load_partial(shared, thread + stride);
            store_partial(shared, thread, fp32_abs_max(left, right));
        }
        block_barrier();
        stride /= 2;
    }

    let amax = load_partial(shared, 0);
    let raw_scale = fp32_div_rn(amax, 448.0_f32);
    let scale = if raw_scale == 0.0 { 1.0_f32 } else { raw_scale };
    if thread == 0 {
        // SAFETY: CTA row indexes one of the `rows` output scales.
        unsafe { scales.add(row as usize).write(scale) };
    }

    column = thread;
    while column < width {
        let index = row_start + column as usize;
        // SAFETY: The launch contract provides this input and a distinct output byte.
        let input_bits = unsafe { *input.add(index) };
        let value = decode_bf16(input_bits);
        let scaled = fp32_div_rn(value, scale);
        let magnitude = f32::from_bits(scaled.to_bits() & 0x7fff_ffff);
        let sign = ((input_bits & 0x8000) >> 8) as u8;
        let code = sign | encode_e4m3fn_rne(magnitude);
        // SAFETY: CTA rows and thread columns map to distinct output bytes.
        unsafe { codes.add(index).write(code) };
        column += BLOCK_THREADS;
    }
}
