use core::arch::asm;

const WARP_LANES: usize = 32;
const WARPS_PER_BLOCK: usize = 8;
const VALUES_PER_LANE: usize = 16;

#[inline(always)]
fn coordinates() -> (usize, usize, usize) {
    let lane: u32;
    let thread: u32;
    let tile: u32;
    // SAFETY: Reads the calling thread's lane, thread, and CTA coordinates only.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile}, %ctaid.x;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile = out(reg32) tile,
            options(nomem, nostack),
        )
    };
    (lane as usize, thread as usize / WARP_LANES, tile as usize)
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline(always)]
fn decode_e2m1(code: u8) -> f32 {
    let magnitude = match code & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[inline(always)]
fn decode_e4m3(code: u8) -> f32 {
    let exponent = u32::from(code >> 3);
    let fraction = u32::from(code & 0x07);
    if exponent == 0 {
        fraction as f32 * (1.0_f32 / 512.0_f32)
    } else {
        f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
    }
}

#[inline(always)]
fn multiply_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: Scalar FP32 multiply has no memory or stack effects.
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

#[inline(always)]
fn fma_rn(accumulator: f32, left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: Scalar FP32 fused multiply-add has no memory or stack effects.
    unsafe {
        asm!(
            "fma.rn.f32 {result}, {left}, {right}, {accumulator};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            accumulator = in(reg32) accumulator,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn add_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: Scalar FP32 addition has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {result}, {left}, {right};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn warp_reduce_down(value: f32, lane: usize) -> f32 {
    let mut sum = value;
    let mut offset = WARP_LANES / 2;
    while offset > 0 {
        let partner: u32;
        // SAFETY: All 32 warp lanes execute each full-mask down-shuffle.
        unsafe {
            asm!(
                "shfl.sync.down.b32 {partner}, {value}, {offset}, 0x1f, 0xffffffff;",
                partner = out(reg32) partner,
                value = in(reg32) sum.to_bits(),
                offset = in(reg32) offset as u32,
                options(nomem, nostack),
            )
        };
        if lane + offset < WARP_LANES {
            sum = add_rn(sum, f32::from_bits(partner));
        }
        offset /= 2;
    }
    sum
}

#[inline(always)]
unsafe fn dot_row(
    input: *const u16,
    codes: *const u8,
    scales: *const u8,
    row: usize,
    lane: usize,
    width: usize,
    inverse_weight_divisor: f32,
) -> f32 {
    let mut accumulators = [0.0_f32; 4];
    let groups_per_row = width / VALUES_PER_LANE;
    let phases = groups_per_row.div_ceil(WARP_LANES);
    for phase in 0..phases {
        let group = phase * WARP_LANES + lane;
        if group < groups_per_row {
            // SAFETY: The in-range group owns 16 BF16 activations, 8 packed bytes, and one scale.
            let scale_code = unsafe { *scales.add(row * groups_per_row + group) };
            let coefficient = multiply_rn(decode_e4m3(scale_code), inverse_weight_divisor);
            let first = group * VALUES_PER_LANE;
            for value_index in 0..VALUES_PER_LANE {
                let position = first + value_index;
                // SAFETY: Group bounds provide the full 16-value input/code segment.
                let (activation, packed) = unsafe {
                    (
                        decode_bf16(*input.add(position)),
                        *codes.add(row * (width / 2) + position / 2),
                    )
                };
                let nibble = (packed >> ((position % 2) * 4)) & 0x0f;
                let weight = multiply_rn(decode_e2m1(nibble), coefficient);
                let chain = value_index & 3;
                accumulators[chain] = fma_rn(accumulators[chain], weight, activation);
            }
        }
    }
    let chains = add_rn(add_rn(add_rn(accumulators[0], accumulators[1]), accumulators[2]), accumulators[3]);
    warp_reduce_down(chains, lane)
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

/// Fuse direct BF16 NVFP4 gate/up GEMVs with one raw-FP32 SwiGLU product.
///
/// Weights use canonical row-major low-nibble-first E2M1 and one E4M3 scale per
/// K16 group. Eight warps process eight gate/up pairs per CTA. This is an A16
/// arithmetic profile, not the baseline activation-quantized integer profile.
///
/// # Safety
/// Launch `grid = [ceil(channels / 8), 1, 1]`, `block = [256, 1, 1]`. Require
/// `1 <= channels <= 32768` and `width` a multiple of 16 in `16..=32768`. `input`
/// covers `width` BF16 elements; each packed matrix covers `channels * width / 2`
/// bytes and each scale plane covers `channels * width / 16` bytes. Both inverse
/// weight divisors are finite and positive. The two BF16 outputs cover `channels`
/// elements, both raw outputs cover `channels` FP32 elements, and activation covers
/// `channels` BF16 elements. All buffers are aligned, pairwise disjoint, and remain
/// live through launch completion. The host validates finite inputs/weights and
/// outputs before accepting the report.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_swiglu_a16(
    input: *const u16,
    gate_codes: *const u8,
    gate_scales: *const u8,
    up_codes: *const u8,
    up_scales: *const u8,
    gate_output: *mut u16,
    gate_raw: *mut f32,
    up_output: *mut u16,
    up_raw: *mut f32,
    activation_raw: *mut f32,
    activation: *mut u16,
    channels: u32,
    width: u32,
    gate_inverse_divisor: f32,
    up_inverse_divisor: f32,
) {
    let (lane, warp, tile) = coordinates();
    let channel = tile * WARPS_PER_BLOCK + warp;
    let channels = channels as usize;
    let width = width as usize;
    let valid = channel < channels;
    let gate_dot = if valid {
        // SAFETY: Host launch contract validates the one-row BF16 input and gate planes.
        unsafe { dot_row(input, gate_codes, gate_scales, channel, lane, width, gate_inverse_divisor) }
    } else {
        0.0
    };
    let up_dot = if valid {
        // SAFETY: Host launch contract validates the one-row BF16 input and up planes.
        unsafe { dot_row(input, up_codes, up_scales, channel, lane, width, up_inverse_divisor) }
    } else {
        0.0
    };
    if lane == 0 && valid {
        let silu = super::silu::silu(gate_dot);
        let product = multiply_rn(silu, up_dot);
        // SAFETY: Lane zero owns one channel in every disjoint output allocation.
        unsafe {
            gate_raw.add(channel).write(gate_dot);
            gate_output.add(channel).write(encode_bf16_rne(gate_dot));
            up_raw.add(channel).write(up_dot);
            up_output.add(channel).write(encode_bf16_rne(up_dot));
            activation_raw.add(channel).write(product);
            activation.add(channel).write(encode_bf16_rne(product));
        }
    }
}
