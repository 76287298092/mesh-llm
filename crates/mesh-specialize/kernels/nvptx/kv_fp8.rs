use core::arch::asm;

const ROW_WIDTH: u32 = 256;
const BLOCK_THREADS: u32 = 256;
const SCALE_ONE_F16: u16 = 0x3c00;
const STATUS_NONFINITE_BF16: u32 = 1;
const STATUS_SCALE_OUT_OF_RANGE: u32 = 2;
const STATUS_INVALID_E4M3: u32 = 4;
const STATUS_INVALID_F16_SCALE: u32 = 8;

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
fn encode_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares 256 u32 reduction slots for one encode CTA.
    unsafe {
        asm!(
            ".shared .align 4 .b8 kv_fp8_encode_partials[1024];",
            "mov.u32 {base}, kv_fp8_encode_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn decode_flags_base() -> u32 {
    let base: u32;
    // SAFETY: Declares 256 u32 validation slots for one decode CTA.
    unsafe {
        asm!(
            ".shared .align 4 .b8 kv_fp8_decode_flags[1024];",
            "mov.u32 {base}, kv_fp8_decode_flags;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_shared_u32(base: u32, index: u32, value: u32) {
    let address = base + index * 4;
    // SAFETY: Callers keep `index` inside the 256-element CTA-local array.
    unsafe {
        asm!(
            "st.shared.u32 [{address}], {value};",
            address = in(reg32) address,
            value = in(reg32) value,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_u32(base: u32, index: u32) -> u32 {
    let value: u32;
    let address = base + index * 4;
    // SAFETY: Callers keep `index` inside the 256-element CTA-local array.
    unsafe {
        asm!(
            "ld.shared.u32 {value}, [{address}];",
            value = out(reg32) value,
            address = in(reg32) address,
            options(nostack),
        )
    };
    value
}

#[inline(always)]
fn block_barrier() {
    // SAFETY: Every thread in the 256-thread CTA reaches each barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
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
fn fp32_mul_rn(left: f32, right: f32) -> f32 {
    let product: f32;
    // SAFETY: Scalar FP32 multiplication has no memory or stack effects.
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
fn fp32_to_f16_rn(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Converts one FP32 scale to IEEE binary16 using round-to-nearest-even.
    unsafe {
        asm!(
            "cvt.rn.f16.f32 {bits}, {value};",
            bits = out(reg16) bits,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    bits
}

#[inline(always)]
fn f16_to_fp32(bits: u16) -> f32 {
    let value: f32;
    // SAFETY: Converts one IEEE binary16 value to FP32 without changing its value.
    unsafe {
        asm!(
            "cvt.f32.f16 {value}, {bits};",
            value = out(reg32) value,
            bits = in(reg16) bits,
            options(nomem, nostack),
        )
    };
    value
}

#[inline(always)]
fn fp32_to_bf16_rn(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Converts one FP32 cache value to BF16 with round-to-nearest-even.
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
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn decode_e4m3fn_positive(code: u8) -> f32 {
    let exponent = (code >> 3) as u32;
    let fraction = (code & 7) as u32;
    if exponent == 0 {
        fraction as f32 * (1.0_f32 / 512.0_f32)
    } else {
        f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
    }
}

#[inline(always)]
fn encode_e4m3fn_rn(magnitude: f32) -> u8 {
    if magnitude >= 448.0_f32 {
        return 126;
    }

    let mut low = 0_u32;
    let mut high = 126_u32;
    while low < high {
        let middle = low + (high - low) / 2;
        if decode_e4m3fn_positive(middle as u8) < magnitude {
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
    let lower_distance = magnitude - decode_e4m3fn_positive(lower);
    let upper_distance = decode_e4m3fn_positive(upper) - magnitude;
    if lower_distance < upper_distance || (lower_distance == upper_distance && lower & 1 == 0) {
        lower
    } else {
        upper
    }
}

#[inline(always)]
fn choose_scale_f16(maximum: f32) -> Option<u16> {
    if maximum == 0.0 {
        return Some(SCALE_ONE_F16);
    }

    let mut scale_bits = fp32_to_f16_rn(fp32_div_rn(maximum, 448.0_f32));
    if scale_bits == 0 {
        // The smallest positive binary16 scale is 2^-24.
        scale_bits = 1;
    }
    if scale_bits >= 0x7c00 {
        return None;
    }

    let represented = f16_to_fp32(scale_bits);
    if fp32_mul_rn(represented, 448.0_f32) < maximum {
        if scale_bits == 0x7bff {
            return None;
        }
        scale_bits += 1;
        if scale_bits >= 0x7c00 {
            return None;
        }
    }

    Some(scale_bits)
}

/// Encode row-major BF16 cache rows as signed E4M3FN with one represented FP16 scale per row.
///
/// Each CTA owns one row of exactly 256 values. The scale is the smallest finite positive
/// binary16 value whose represented value times 448 covers the row maximum. A zero row uses
/// scale one. Nonfinite input and an unrepresentable scale set a per-row status and produce
/// deterministic zero codes; the scale output for either error row is one.
///
/// # Safety
/// Launch `grid.x >= rows` with `1 <= rows <= 262144` and `block = [256, 1, 1]`; padded grid
/// CTAs return before touching memory. `rows * 256` must fit `usize`. `input_bf16` must cover
/// that many readable BF16 values; `codes` must cover that many writable bytes; `scales_f16` and
/// `status` must each cover `rows` writable u16/u32 values. The four allocations must be aligned
/// as required by their element types, disjoint, and live through kernel completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kv_fp8_encode_bf16_width256(
    input_bf16: *const u16,
    codes: *mut u8,
    scales_f16: *mut u16,
    status: *mut u32,
    rows: u32,
) {
    let (thread, row) = thread_and_row();
    if row >= rows {
        return;
    }

    let shared = encode_partials_base();
    let row_start = row as usize * ROW_WIDTH as usize;
    let input_bits = unsafe { *input_bf16.add(row_start + thread as usize) };
    let absolute_bits = ((input_bits as u32) << 16) & 0x7fff_ffff;
    let nonfinite = (absolute_bits & 0x7f80_0000) == 0x7f80_0000;
    // Finite positive FP32 bit patterns sort as unsigned integers. Bit 31 carries
    // the row-wide nonfinite-input flag, above the finite FP32 range.
    let partial = (if nonfinite { 0x8000_0000 } else { 0 }) | absolute_bits;
    store_shared_u32(shared, thread, partial);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_shared_u32(shared, thread);
            let right = load_shared_u32(shared, thread + stride);
            store_shared_u32(shared, thread, left.max(right));
        }
        block_barrier();
        stride /= 2;
    }

    let row_max = load_shared_u32(shared, 0);
    let has_nonfinite = row_max & 0x8000_0000 != 0;
    let maximum = f32::from_bits(row_max & 0x7fff_ffff);
    let selected_scale = if has_nonfinite {
        None
    } else {
        choose_scale_f16(maximum)
    };

    if has_nonfinite || selected_scale.is_none() {
        let row_status = if has_nonfinite {
            STATUS_NONFINITE_BF16
        } else {
            STATUS_SCALE_OUT_OF_RANGE
        };
        // SAFETY: This CTA uniquely owns all 256 codes for its row.
        unsafe { codes.add(row_start + thread as usize).write(0) };
        if thread == 0 {
            // SAFETY: This CTA uniquely owns its scale and status entries.
            unsafe {
                scales_f16.add(row as usize).write(SCALE_ONE_F16);
                status.add(row as usize).write(row_status);
            }
        }
        return;
    }

    let scale_bits = selected_scale.unwrap_or(SCALE_ONE_F16);
    let scale = f16_to_fp32(scale_bits);
    if thread == 0 {
        // SAFETY: This CTA uniquely owns its scale and status entries.
        unsafe {
            scales_f16.add(row as usize).write(scale_bits);
            status.add(row as usize).write(0);
        }
    }

    let value = decode_bf16(input_bits);
    let scaled = fp32_div_rn(value, scale);
    let magnitude = f32::from_bits(scaled.to_bits() & 0x7fff_ffff);
    let sign = ((input_bits & 0x8000) >> 8) as u8;
    let code = sign | encode_e4m3fn_rn(magnitude);
    // SAFETY: This CTA uniquely owns one code byte in its row.
    unsafe { codes.add(row_start + thread as usize).write(code) };
}

#[inline(always)]
fn is_valid_positive_f16(bits: u16) -> bool {
    bits & 0x8000 == 0 && bits & 0x7fff != 0 && bits & 0x7c00 != 0x7c00
}

#[inline(always)]
fn decode_e4m3fn(code: u8) -> f32 {
    let magnitude = code & 0x7f;
    let positive = decode_e4m3fn_positive(magnitude);
    if code & 0x80 == 0 {
        positive
    } else {
        f32::from_bits(positive.to_bits() | 0x8000_0000)
    }
}

/// Decode finite signed E4M3FN cache rows through their represented FP16 scales into BF16 rows.
///
/// Each CTA owns one row of exactly 256 values. Invalid E4M3 NaN codes and invalid scales are
/// reported per row; an invalid row is written as zero BF16 values. Scale errors require a
/// finite, strictly positive IEEE binary16 value, including valid positive subnormals.
///
/// # Safety
/// Launch `grid.x >= rows` with `1 <= rows <= 262144` and `block = [256, 1, 1]`; padded grid
/// CTAs return before touching memory. `rows * 256` must fit `usize`. `codes` must cover that
/// many readable bytes, `scales_f16` must cover `rows` readable u16 values, `output_bf16` must
/// cover that many writable u16 values, and `status` must cover `rows` writable u32 values. The
/// four allocations must be aligned as required by their element types, disjoint, and live
/// through completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kv_fp8_decode_bf16_width256(
    codes: *const u8,
    scales_f16: *const u16,
    output_bf16: *mut u16,
    status: *mut u32,
    rows: u32,
) {
    let (thread, row) = thread_and_row();
    if row >= rows {
        return;
    }

    let shared = decode_flags_base();
    let row_start = row as usize * ROW_WIDTH as usize;
    let code = unsafe { *codes.add(row_start + thread as usize) };
    let scale_bits = unsafe { *scales_f16.add(row as usize) };
    let mut local_status = 0_u32;
    if code & 0x7f == 0x7f {
        local_status |= STATUS_INVALID_E4M3;
    }
    if !is_valid_positive_f16(scale_bits) {
        local_status |= STATUS_INVALID_F16_SCALE;
    }
    store_shared_u32(shared, thread, local_status);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_shared_u32(shared, thread);
            let right = load_shared_u32(shared, thread + stride);
            store_shared_u32(shared, thread, left | right);
        }
        block_barrier();
        stride /= 2;
    }

    let row_status = load_shared_u32(shared, 0);
    if row_status != 0 {
        // SAFETY: This CTA uniquely owns all 256 output values for its row.
        unsafe { output_bf16.add(row_start + thread as usize).write(0) };
        if thread == 0 {
            // SAFETY: This CTA uniquely owns its row status entry.
            unsafe { status.add(row as usize).write(row_status) };
        }
        return;
    }

    let scale = f16_to_fp32(scale_bits);
    let value = fp32_mul_rn(decode_e4m3fn(code), scale);
    let rounded = fp32_to_bf16_rn(value);
    // SAFETY: This CTA uniquely owns one BF16 output in its row.
    unsafe { output_bf16.add(row_start + thread as usize).write(rounded) };
    if thread == 0 {
        // SAFETY: This CTA uniquely owns its row status entry.
        unsafe { status.add(row as usize).write(0) };
    }
}
