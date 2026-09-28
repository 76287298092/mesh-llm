//! Unregistered exact-decode candidate. Only packed E2M1 expansion differs from control.
use core::arch::asm;

use super::{
    fp8_linear_exact::{
        add_s64, e4m3fn_units, fp32_multiply_rn, i64_to_f32_rn, multiply_wide_s32, warp_reduce_s64,
    },
    nvfp4_linear::store_scaled_output,
};

const WARP_SIZE: u32 = 32;
const OUTPUTS_PER_BLOCK: usize = 4;
const SCALE_GROUP_K: usize = 16;

#[inline(always)]
fn lane_warp_tile() -> (u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile: u32;
    // SAFETY: Reads the calling thread's lane, CTA thread, and block coordinates only.
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
    (lane, thread / WARP_SIZE, tile)
}

#[inline(always)]
fn pack_e2m1x4(word: u32, first_shift: u32) -> u32 {
    let ctrl = word >> first_shift;
    let selectors = ctrl & 0x7777;
    // Each nibble's low bit becomes OR(magnitude bits). Bits from adjoining
    // nibbles cannot enter these positions. Code 8 MUST NOT request negation:
    // complement(0) + 1 would otherwise carry into the following byte.
    let nonzero = (ctrl | (ctrl >> 1) | (ctrl >> 2)) & 0x1111;
    let effective_sign = ctrl & (nonzero << 3);
    let magnitudes: u32;
    let sign_bytes: u32;
    // SAFETY: Generic PRMT selectors 0..7 select LUT bytes; selectors 8..15
    // replicate the selected byte's MSB. The second LUT has every MSB set,
    // producing 0x80 for positive/zero and 0xff for negative nonzero values.
    unsafe {
        asm!(
            "prmt.b32 {magnitudes}, {low}, {high}, {selectors};",
            "prmt.b32 {sign_bytes}, {sign_lut}, {sign_lut}, {effective_sign};",
            magnitudes = out(reg32) magnitudes,
            sign_bytes = out(reg32) sign_bytes,
            low = in(reg32) 0x0302_0100_u32,
            high = in(reg32) 0x0c08_0604_u32,
            selectors = in(reg32) selectors,
            sign_lut = in(reg32) 0x8080_8080_u32,
            effective_sign = in(reg32) effective_sign,
            options(nomem, nostack),
        )
    };
    let sign_ones = sign_bytes & 0x0101_0101;
    let sign_mask = sign_ones.wrapping_mul(255);
    // No inter-byte carry: positive bytes add 0; negative magnitudes 1..12
    // produce (255-m)+1 in 244..255. Signed zero is canonical integer zero.
    (magnitudes ^ sign_mask).wrapping_add(sign_ones)
}

#[inline(always)]
fn dp4a_s32(accumulator: i32, left: u32, right: u32) -> i32 {
    let sum: i32;
    // SAFETY: The packed operands contain four signed i8 values in each 32-bit word.
    unsafe {
        asm!(
            "dp4a.s32.s32 {sum}, {left}, {right}, {accumulator};",
            sum = out(reg32) sum,
            left = in(reg32) left,
            right = in(reg32) right,
            accumulator = in(reg32) accumulator,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
fn dot_e2m1_group16(
    activation_word0: u32,
    activation_word1: u32,
    weight_word0: u32,
    weight_word1: u32,
) -> i32 {
    let activation0 = pack_e2m1x4(activation_word0, 0);
    let activation1 = pack_e2m1x4(activation_word0, 16);
    let activation2 = pack_e2m1x4(activation_word1, 0);
    let activation3 = pack_e2m1x4(activation_word1, 16);
    let weight0 = pack_e2m1x4(weight_word0, 0);
    let weight1 = pack_e2m1x4(weight_word0, 16);
    let weight2 = pack_e2m1x4(weight_word1, 0);
    let weight3 = pack_e2m1x4(weight_word1, 16);

    let partial = dp4a_s32(0, activation0, weight0);
    let partial = dp4a_s32(partial, activation1, weight1);
    let partial = dp4a_s32(partial, activation2, weight2);
    dp4a_s32(partial, activation3, weight3)
}

/// Load the two aligned words that hold one row's sixteen packed E2M1 values.
///
/// # Safety
/// `matrix` must cover `row_count * groups_per_row * 8` readable bytes and have a
/// four-byte-aligned base. Require `row < row_count`, `group < groups_per_row`, and
/// all offset arithmetic to fit `usize`.
#[inline(always)]
unsafe fn load_group16_words(
    matrix: *const u8,
    row: usize,
    row_count: usize,
    groups_per_row: usize,
    group: usize,
) -> (u32, u32) {
    if row >= row_count || group >= groups_per_row {
        return (0, 0);
    }
    let byte_offset = (row * groups_per_row + group) * 8;
    // SAFETY: The caller's extent/alignment contract covers both four-byte reads.
    unsafe {
        (
            u32::from_le(matrix.add(byte_offset).cast::<u32>().read()),
            u32::from_le(matrix.add(byte_offset + 4).cast::<u32>().read()),
        )
    }
}

/// Exact integer decode with two-PRMT E2M1 expansion. Not a new arithmetic profile.
///
/// # Safety
/// Identical to `nvfp4_decode_exact`: grid `[ceil(n/4),1,1]`, block `[128,1,1]`,
/// m=1, n in 1..=32768, k a multiple of 16 in 16..=32768. Disjoint a/w packed
/// low-nibble-first rows cover k/2 and n*k/2 bytes and are four-byte aligned.
/// Disjoint sa/sw cover k/16 and n*k/16 bytes with unsigned scale codes 0..=126.
/// Disjoint naturally aligned out/unrounded cover n BF16/FP32 elements. All
/// pointers remain live through completion and offsets fit the address space.
/// global_factor is finite and positive; host rejects nonfinite output. The
/// signed group dot is bounded by 2304, activation-scaled dot by 528482304, and
/// total i64 magnitude by less than 2^58. No dynamic shared memory is required.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_decode_exact_prmt(
    a: *const u8,
    w: *const u8,
    sa: *const u8,
    sw: *const u8,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    global_factor: f32,
) {
    // The shape check is block-uniform, so no warp can be partially removed before reduction.
    if m != 1 {
        return;
    }

    let (lane, warp, tile) = lane_warp_tile();
    let lane = lane as usize;
    let output_channel = tile as usize * OUTPUTS_PER_BLOCK + warp as usize;
    let n = n as usize;
    let k = k as usize;
    let groups_per_row = k / SCALE_GROUP_K;
    let channel_valid = output_channel < n;
    let mut partial = 0_i64;

    if channel_valid {
        let weight_scale_start = output_channel * groups_per_row;
        let mut group = lane;
        while group < groups_per_row {
            // SAFETY: A valid output channel and in-range group select complete,
            // four-byte-aligned activation and weight group words.
            let ((activation_word0, activation_word1), (weight_word0, weight_word1)) = unsafe {
                (
                    load_group16_words(a, 0, 1, groups_per_row, group),
                    load_group16_words(w, output_channel, n, groups_per_row, group),
                )
            };
            // SAFETY: The group bounds above cover both scale bytes.
            let (activation_scale, weight_scale) = unsafe {
                (
                    e4m3fn_units(*sa.add(group)),
                    e4m3fn_units(*sw.add(weight_scale_start + group)),
                )
            };
            let group_dot = dot_e2m1_group16(
                activation_word0,
                activation_word1,
                weight_word0,
                weight_word1,
            );
            // The documented bound keeps this i32 product in range.
            let activation_scaled_dot = group_dot * activation_scale;
            partial = add_s64(
                partial,
                multiply_wide_s32(activation_scaled_dot, weight_scale),
            );
            group += WARP_SIZE as usize;
        }
    }

    // All lanes of each warp, including warps past the N tail, take every full-mask shuffle.
    let total = warp_reduce_s64(partial);

    if lane == 0 && channel_valid {
        let dot = fp32_multiply_rn(i64_to_f32_rn(total), 1.0 / 1_048_576.0);
        // SAFETY: This valid warp uniquely owns output `[0, output_channel]`; the shared
        // store applies the unchanged global-factor multiply and BF16 RNE conversion.
        unsafe {
            store_scaled_output(out, unrounded, 0, output_channel, 1, n, dot, global_factor);
        }
    }
}
