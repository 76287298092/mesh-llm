//! M=1 schedule only. Arithmetic and epilogue helpers are the unchanged control's.
use super::fp8_linear_exact::{
    add_s64, decode_bf16, e4m3fn_units, encode_bf16_rne, fp32_multiply_rn, i64_to_f32_rn,
    multiply_wide_s32, warp_reduce_s64,
};
use core::arch::asm;

#[inline(always)]
fn coordinates() -> (u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile: u32;
    // SAFETY: Reads only the calling thread's special registers.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile}, %ctaid.x;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile = out(reg32) tile,
            options(nomem, nostack),
        );
    }
    (lane, thread, tile)
}

#[inline(always)]
unsafe fn load16(pointer: *const u8) -> [u32; 4] {
    let x: u32;
    let y: u32;
    let z: u32;
    let w: u32;
    // SAFETY: Caller guarantees a 16-byte aligned, fully readable global span.
    unsafe {
        asm!(
            "ld.global.v4.u32 {{{x}, {y}, {z}, {w}}}, [{pointer}];",
            x = out(reg32) x,
            y = out(reg32) y,
            z = out(reg32) z,
            w = out(reg32) w,
            pointer = in(reg64) pointer as u64,
            options(readonly, nostack),
        );
    }
    [x, y, z, w]
}

#[inline(always)]
fn accumulate(sum: i64, a: u32, w: u32, shift: u32) -> i64 {
    add_s64(
        sum,
        multiply_wide_s32(
            e4m3fn_units((a >> shift) as u8),
            e4m3fn_units((w >> shift) as u8),
        ),
    )
}

/// Exact finite E4M3FN dot with four independent integer chains per lane.
///
/// # Safety
/// Same nine-argument ABI and scale/output contracts as `fp8_linear_exact`, except
/// `m == 1`, `k in 16..=32768` must be a multiple of 16, and A/W must be 16-byte
/// aligned. Launch grid `[ceil(n/4), 1, 1]`, block `[128, 1, 1]`; `n in 1..=262144`.
/// A covers K bytes, W covers N*K bytes; every code is finite (not 0x7f/0xff).
/// All six allocations are disjoint and live through completion. Output extents
/// are N BF16 / N FP32. Scales are finite positive FP32 / BF16 as in the control.
/// Every integer subset sum is bounded by K*229376^2 < 2^51, so reassociation is
/// exact. K alignment makes each last vector fully readable, even below 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_linear_exact_vector16(
    a: *const u8,
    w: *const u8,
    sa: *const f32,
    sw: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (lane, thread, tile) = coordinates();
    let column = tile as usize * 4 + (thread / 32) as usize;
    let valid = m == 1 && column < n as usize;
    let (mut s0, mut s1, mut s2, mut s3) = (0_i64, 0_i64, 0_i64, 0_i64);
    // The N-tail branch is warp-uniform. Inactive warps still join the reduction.
    if valid {
        let mut index = lane as usize * 16;
        while index < k as usize {
            // SAFETY: K and both bases are 16-byte aligned. index+16 <= K.
            let (av, wv) = unsafe {
                (
                    load16(a.add(index)),
                    load16(w.add(column * k as usize + index)),
                )
            };
            // Each word feeds a separate dependency chain, four code pairs each.
            // Fixed shifts expose all 16 products without a variable-index array.
            s0 = accumulate(s0, av[0], wv[0], 0);
            s1 = accumulate(s1, av[1], wv[1], 0);
            s2 = accumulate(s2, av[2], wv[2], 0);
            s3 = accumulate(s3, av[3], wv[3], 0);
            s0 = accumulate(s0, av[0], wv[0], 8);
            s1 = accumulate(s1, av[1], wv[1], 8);
            s2 = accumulate(s2, av[2], wv[2], 8);
            s3 = accumulate(s3, av[3], wv[3], 8);
            s0 = accumulate(s0, av[0], wv[0], 16);
            s1 = accumulate(s1, av[1], wv[1], 16);
            s2 = accumulate(s2, av[2], wv[2], 16);
            s3 = accumulate(s3, av[3], wv[3], 16);
            s0 = accumulate(s0, av[0], wv[0], 24);
            s1 = accumulate(s1, av[1], wv[1], 24);
            s2 = accumulate(s2, av[2], wv[2], 24);
            s3 = accumulate(s3, av[3], wv[3], 24);
            index += 32 * 16;
        }
    }
    let total = warp_reduce_s64(add_s64(add_s64(s0, s1), add_s64(s2, s3)));
    if lane == 0 && valid {
        let dot_units = i64_to_f32_rn(total);
        let dot = fp32_multiply_rn(dot_units, 1.0 / 262_144.0);
        // SAFETY: This valid output column has its full scale and output extents.
        let (row_scale, channel_scale) = unsafe { (*sa, decode_bf16(*sw.add(column))) };
        let scaled = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
        // SAFETY: Exactly lane zero of this warp owns this in-range output.
        unsafe {
            unrounded.add(column).write(scaled);
            out.add(column).write(encode_bf16_rne(scaled));
        }
    }
}
