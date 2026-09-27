use core::arch::asm;

use super::fp8_linear_exact::{
    add_s64, decode_bf16, e4m3fn_units, encode_bf16_rne, fp32_multiply_rn, i64_to_f32_rn,
    multiply_wide_s32, warp_reduce_s64,
};

const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: usize = 4;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile_channel: u32;
    let row: u32;
    // SAFETY: Reads the calling thread's lane, CTA, and thread coordinates only.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile_channel}, %ctaid.x;",
            "mov.u32 {row}, %ctaid.y;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile_channel = out(reg32) tile_channel,
            row = out(reg32) row,
            options(nomem, nostack),
        )
    };
    (lane, thread, tile_channel, row)
}

/// Accumulate gate and up dots while decoding the shared activation byte once.
///
/// # Safety
/// The caller must provide readable row-major E4M3FN extents for the selected
/// activation row and gate/up channels, and `k` must fit the validated extent.
#[inline(always)]
unsafe fn exact_dot_pair(
    input_codes: *const u8,
    gate_weight: *const u8,
    up_weight: *const u8,
    row_start: usize,
    gate_start: usize,
    up_start: usize,
    k: usize,
    lane: usize,
) -> (i64, i64) {
    let mut gate_partial = 0_i64;
    let mut up_partial = 0_i64;
    let mut index = lane;
    while index < k {
        // SAFETY: The launch contract covers every selected row and channel, and
        // `index < k` keeps all three reads within their row extents.
        let (input_code, gate_code, up_code) = unsafe {
            (
                *input_codes.add(row_start + index),
                *gate_weight.add(gate_start + index),
                *up_weight.add(up_start + index),
            )
        };
        let input_units = e4m3fn_units(input_code);
        gate_partial = add_s64(
            gate_partial,
            multiply_wide_s32(input_units, e4m3fn_units(gate_code)),
        );
        up_partial = add_s64(
            up_partial,
            multiply_wide_s32(input_units, e4m3fn_units(up_code)),
        );
        index += WARP_SIZE as usize;
    }
    (gate_partial, up_partial)
}

#[inline(always)]
fn scaled_projection(total: i64, input_scale: f32, weight_scale: u16) -> u16 {
    let dot = fp32_multiply_rn(i64_to_f32_rn(total), 1.0 / 262_144.0);
    let scaled = fp32_multiply_rn(
        fp32_multiply_rn(dot, input_scale),
        decode_bf16(weight_scale),
    );
    encode_bf16_rne(scaled)
}

#[inline(always)]
fn swiglu(gate: u16, up: u16) -> (u16, f32) {
    let activated = encode_bf16_rne(super::silu::silu(decode_bf16(gate)));
    let product = fp32_multiply_rn(decode_bf16(activated), decode_bf16(up));
    (encode_bf16_rne(product), product)
}

/// Fuse exact E4M3FN gate/up projections with the existing BF16 SwiGLU profile.
///
/// Input codes and row scales are shared by both projections. Gate and up each
/// retain their own exact dot, ordered FP32 scaling, and BF16 RNE boundary. SiLU
/// consumes decoded BF16 gate, rounds to BF16 before multiplication, and the
/// final product is stored as both FP32 diagnostics and BF16 RNE output.
///
/// # Safety
/// Launch `grid = [ceil(n / 4), m, 1]` and `block = [128, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `input_codes` must
/// cover row-major `[m, k]`; `gate_weight` and `up_weight` must each cover
/// row-major `[n, k]`. All codes must be finite E4M3FN values, excluding `0x7f`
/// and `0xff`. `input_scales` must cover `m` finite positive FP32 scales, and
/// `gate_scales` and `up_scales` must each cover `n` finite positive BF16 scales.
/// `output` and `unrounded` must each cover row-major `[m, n]` writable BF16 and
/// FP32 values. All index arithmetic must fit the device address space. Pointers
/// must be aligned for their element types, pairwise disjoint, and live through
/// kernel completion. The exact signed integer dots fit in i64 for this K range.
/// The host must reject nonfinite projection, activation, or product results.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_swiglu_exact(
    input_codes: *const u8,
    gate_weight: *const u8,
    up_weight: *const u8,
    input_scales: *const f32,
    gate_scales: *const u16,
    up_scales: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (lane, thread, tile_channel, row) = coordinates();
    let channel = tile_channel as usize * WARPS_PER_BLOCK + (thread / WARP_SIZE) as usize;
    let row = row as usize;
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let row_valid = row < m;
    let channel_valid = channel < n;

    let (gate_partial, up_partial) = if row_valid && channel_valid {
        let row_start = row * k;
        let gate_start = channel * k;
        // Both projections read the same activation row and output channel.
        // SAFETY: Valid row/channel coordinates and the documented extents make
        // every row start and K-strided load fit its respective array.
        unsafe {
            exact_dot_pair(
                input_codes,
                gate_weight,
                up_weight,
                row_start,
                gate_start,
                gate_start,
                k,
                lane as usize,
            )
        }
    } else {
        (0, 0)
    };

    // Every lane reaches both full-mask reductions, including M/N tail warps.
    let gate_total = warp_reduce_s64(gate_partial);
    let up_total = warp_reduce_s64(up_partial);

    if lane == 0 && row_valid && channel_valid {
        // SAFETY: This valid row/channel pair has one entry in each scale array.
        let (input_scale, gate_scale, up_scale) = unsafe {
            (
                *input_scales.add(row),
                *gate_scales.add(channel),
                *up_scales.add(channel),
            )
        };
        let gate = scaled_projection(gate_total, input_scale, gate_scale);
        let up = scaled_projection(up_total, input_scale, up_scale);
        let (result, diagnostic) = swiglu(gate, up);
        let output_index = row * n + channel;
        // SAFETY: Lane zero uniquely owns this in-range row/channel output.
        unsafe {
            output.add(output_index).write(result);
            unrounded.add(output_index).write(diagnostic);
        }
    }
}
