use core::arch::asm;

use super::fp8_linear_exact::{
    add_s64, decode_bf16, e4m3fn_units, encode_bf16_rne, fp32_multiply_rn, i64_to_f32_rn,
    multiply_wide_s32, warp_reduce_s64,
};

const WARP_SIZE: u32 = 32;

#[inline(always)]
fn coordinates() -> (u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile_n: u32;
    let tile_m: u32;
    // SAFETY: Reads the calling thread's lane, thread, and CTA coordinates only.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {tile_m}, %ctaid.y;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile_n = out(reg32) tile_n,
            tile_m = out(reg32) tile_m,
            options(nomem, nostack),
        )
    };
    let column = tile_n * 4 + thread / WARP_SIZE;
    let row_base = tile_m * 4;
    (lane, column, row_base)
}

/// Exact E4M3FN linear projection variant that reuses each decoded weight across four rows.
///
/// Each warp owns one output column and four adjacent output rows. The exact signed
/// integer dot, scale order, and BF16 rounding match `fp8_linear_exact`.
///
/// # Safety
/// Launch `grid = [ceil(n / 4), ceil(m / 4), 1]`, `block = [128, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `a` and `w` must each
/// cover row-major matrices of `m * k` and `n * k` readable E4M3FN bytes. Codes
/// must be finite (not `0x7f` or `0xff`). `sa` must cover `m` readable finite
/// positive FP32 scales, and `sw` must cover `n` readable finite positive BF16
/// scales. `out` and `unrounded` must cover `m * n` writable BF16 and FP32 values.
/// All products and index arithmetic must fit the device address space. Pointers
/// must have their element alignment, be pairwise disjoint, and remain live until
/// kernel completion. The exact integer dot fits signed 64-bit for the stated K.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_linear_exact4(
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
    let (lane, column, row_base) = coordinates();
    let lane_index = lane as usize;
    let column = column as usize;
    let row0 = row_base as usize;
    let row1 = row0 + 1;
    let row2 = row0 + 2;
    let row3 = row0 + 3;
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let row0_valid = row0 < m;
    let row1_valid = row1 < m;
    let row2_valid = row2 < m;
    let row3_valid = row3 < m;
    let column_valid = column < n;

    let mut partial0 = 0_i64;
    let mut partial1 = 0_i64;
    let mut partial2 = 0_i64;
    let mut partial3 = 0_i64;

    if column_valid {
        let weight_start = column * k;
        let row0_start = row0 * k;
        let row1_start = row1 * k;
        let row2_start = row2 * k;
        let row3_start = row3 * k;
        let mut index = lane_index;
        while index < k {
            // SAFETY: `column_valid` and `index < k` select a byte in this weight row.
            let weight_units = unsafe { e4m3fn_units(*w.add(weight_start + index)) };
            if row0_valid {
                // SAFETY: `row0_valid` and `index < k` select a byte in this activation row.
                let activation_units = unsafe { e4m3fn_units(*a.add(row0_start + index)) };
                partial0 = add_s64(partial0, multiply_wide_s32(activation_units, weight_units));
            }
            if row1_valid {
                // SAFETY: `row1_valid` and `index < k` select a byte in this activation row.
                let activation_units = unsafe { e4m3fn_units(*a.add(row1_start + index)) };
                partial1 = add_s64(partial1, multiply_wide_s32(activation_units, weight_units));
            }
            if row2_valid {
                // SAFETY: `row2_valid` and `index < k` select a byte in this activation row.
                let activation_units = unsafe { e4m3fn_units(*a.add(row2_start + index)) };
                partial2 = add_s64(partial2, multiply_wide_s32(activation_units, weight_units));
            }
            if row3_valid {
                // SAFETY: `row3_valid` and `index < k` select a byte in this activation row.
                let activation_units = unsafe { e4m3fn_units(*a.add(row3_start + index)) };
                partial3 = add_s64(partial3, multiply_wide_s32(activation_units, weight_units));
            }
            index += WARP_SIZE as usize;
        }
    }

    // Every lane in the warp reaches every reduction, including row and column tails.
    let total0 = warp_reduce_s64(partial0);
    let total1 = warp_reduce_s64(partial1);
    let total2 = warp_reduce_s64(partial2);
    let total3 = warp_reduce_s64(partial3);

    if lane == 0 && column_valid {
        // SAFETY: The valid output column has one readable BF16 scale entry.
        let channel_scale = unsafe { decode_bf16(*sw.add(column)) };
        if row0_valid {
            // SAFETY: The valid output row has one readable FP32 scale entry.
            let row_scale = unsafe { *sa.add(row0) };
            let dot = fp32_multiply_rn(i64_to_f32_rn(total0), 1.0 / 262_144.0);
            let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
            let output_index = row0 * n + column;
            // SAFETY: Lane zero uniquely owns this valid row/column output pair.
            unsafe {
                unrounded.add(output_index).write(value);
                out.add(output_index).write(encode_bf16_rne(value));
            }
        }
        if row1_valid {
            // SAFETY: The valid output row has one readable FP32 scale entry.
            let row_scale = unsafe { *sa.add(row1) };
            let dot = fp32_multiply_rn(i64_to_f32_rn(total1), 1.0 / 262_144.0);
            let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
            let output_index = row1 * n + column;
            // SAFETY: Lane zero uniquely owns this valid row/column output pair.
            unsafe {
                unrounded.add(output_index).write(value);
                out.add(output_index).write(encode_bf16_rne(value));
            }
        }
        if row2_valid {
            // SAFETY: The valid output row has one readable FP32 scale entry.
            let row_scale = unsafe { *sa.add(row2) };
            let dot = fp32_multiply_rn(i64_to_f32_rn(total2), 1.0 / 262_144.0);
            let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
            let output_index = row2 * n + column;
            // SAFETY: Lane zero uniquely owns this valid row/column output pair.
            unsafe {
                unrounded.add(output_index).write(value);
                out.add(output_index).write(encode_bf16_rne(value));
            }
        }
        if row3_valid {
            // SAFETY: The valid output row has one readable FP32 scale entry.
            let row_scale = unsafe { *sa.add(row3) };
            let dot = fp32_multiply_rn(i64_to_f32_rn(total3), 1.0 / 262_144.0);
            let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
            let output_index = row3 * n + column;
            // SAFETY: Lane zero uniquely owns this valid row/column output pair.
            unsafe {
                unrounded.add(output_index).write(value);
                out.add(output_index).write(encode_bf16_rne(value));
            }
        }
    }
}
