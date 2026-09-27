use core::arch::asm;

const WARP_LANES: u32 = 32;
const GROUP_VALUES: u32 = 16;

#[inline(always)]
fn group_and_lane() -> (u32, u32) {
    let group: u32;
    let lane: u32;
    // SAFETY: Reads the calling thread's CTA and lane coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {group}, %ctaid.x;",
            "mov.u32 {lane}, %laneid;",
            group = out(reg32) group,
            lane = out(reg32) lane,
            options(nomem, nostack),
        )
    };
    (group, lane)
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
fn warp_max_xor(value: f32, lane_mask: u32) -> f32 {
    let result: f32;
    // SAFETY: All 32 warp lanes execute each full-mask butterfly shuffle.
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {result}, {value}, {lane_mask}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            value = in(reg32) value,
            lane_mask = in(reg32) lane_mask,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn warp_bfly_u32(value: u32, lane_mask: u32) -> u32 {
    let result: u32;
    // SAFETY: All 32 warp lanes execute the full-mask butterfly shuffle.
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {result}, {value}, {lane_mask}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            value = in(reg32) value,
            lane_mask = in(reg32) lane_mask,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn fp32_abs_max(left: f32, right: f32) -> f32 {
    // Inputs are finite and nonnegative, so positive IEEE bit order matches value order.
    if left.to_bits() >= right.to_bits() {
        left
    } else {
        right
    }
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

#[inline(always)]
fn encode_e2m1_magnitude_rne(magnitude: f32) -> u8 {
    let mut best_code = 0_u8;
    let mut best_distance = f32::INFINITY;
    let mut code = 0_u8;
    while code < 8 {
        let level = match code {
            0 => 0.0_f32,
            1 => 0.5_f32,
            2 => 1.0_f32,
            3 => 1.5_f32,
            4 => 2.0_f32,
            5 => 3.0_f32,
            6 => 4.0_f32,
            _ => 6.0_f32,
        };
        let distance = (magnitude - level).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0) {
            best_code = code;
            best_distance = distance;
        }
        code += 1;
    }
    best_code
}

/// Quantize row-major BF16 activations into packed signed NVFP4 E2M1 groups.
///
/// One warp owns 16 contiguous values. Its local scale is the nearest-even finite
/// E4M3FN encoding of `(amax / 6) * global_scale`, replacing a zero encoding with
/// `0.125`; `effective` stores the decoded scale divided by `global_scale`. Each
/// activation is divided by that effective scale, clamped to `[-6, 6]`, and encoded
/// to E2M1 with nearest-even ties and its original sign bit. Adjacent lane codes are
/// packed low-nibble first.
///
/// # Safety
/// Launch exactly `grid = [rows * (width / 16), 1, 1]` and `block = [32, 1, 1]`.
/// `rows` must be 1..=2048; `width` must be a multiple of 16 in 16..=32768; and
/// `global_scale` must be finite and positive. `input` must cover `rows * width`
/// readable BF16 values, `packed` must cover `rows * width / 2` writable bytes,
/// and `scales` and `effective` must each cover `rows * width / 16` writable
/// elements (`u8` and `f32`, respectively). Extents and index arithmetic must fit
/// `usize`; pointers must be correctly aligned, mutually disjoint, and live until
/// completion. The host must validate finite input values, finite nonnegative local
/// scale intermediates, and finite positive effective scales.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_quantize_bf16(
    input: *const u16,
    packed: *mut u8,
    scales: *mut u8,
    effective: *mut f32,
    _rows: u32,
    _width: u32,
    global_scale: f32,
) {
    let (group, lane) = group_and_lane();
    let group_start = group as usize * GROUP_VALUES as usize;
    let mut value = 0.0_f32;
    if lane < GROUP_VALUES {
        // SAFETY: Exact group grid and width multiple cover this input element.
        value = unsafe { decode_bf16(*input.add(group_start + lane as usize)) };
    }
    let magnitude = f32::from_bits(value.to_bits() & 0x7fff_ffff);

    let mut amax = magnitude;
    let mut offset = WARP_LANES / 2;
    while offset > 0 {
        let other = warp_max_xor(amax, offset);
        amax = fp32_abs_max(amax, other);
        offset /= 2;
    }

    let local_scale = fp32_multiply_rn(fp32_divide_rn(amax, 6.0_f32), global_scale);
    let encoded_scale = encode_e4m3fn_rne(local_scale);
    let scale_code = if encoded_scale == 0 {
        0x20_u8
    } else {
        encoded_scale
    };
    let decoded_scale = decode_e4m3fn(scale_code);
    let effective_scale = fp32_divide_rn(decoded_scale, global_scale);
    if lane == 0 {
        // SAFETY: One warp owns each group's unique scale and effective-scale slots.
        unsafe {
            scales.add(group as usize).write(scale_code);
            effective.add(group as usize).write(effective_scale);
        }
    }

    let mut packed_code = 0_u8;
    if lane < GROUP_VALUES {
        let input_bits = unsafe { *input.add(group_start + lane as usize) };
        let input_value = decode_bf16(input_bits);
        let scaled = fp32_divide_rn(input_value, effective_scale);
        let bounded = if scaled > 6.0_f32 {
            6.0_f32
        } else if scaled < -6.0_f32 {
            -6.0_f32
        } else {
            scaled
        };
        let scaled_magnitude = f32::from_bits(bounded.to_bits() & 0x7fff_ffff);
        let sign = ((input_bits >> 12) & 0x8) as u8;
        packed_code = sign | encode_e2m1_magnitude_rne(scaled_magnitude);
    }

    let paired_code = warp_bfly_u32(packed_code as u32, 1) as u8;
    if lane < GROUP_VALUES && lane & 1 == 0 {
        let byte = packed_code | (paired_code << 4);
        // SAFETY: Even lanes 0..14 uniquely write the eight bytes for this group.
        unsafe {
            packed
                .add(group as usize * 8 + lane as usize / 2)
                .write(byte)
        };
    }
}
