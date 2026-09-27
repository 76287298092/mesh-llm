use core::arch::asm;

use super::fp8_linear_exact::{
    add_s64, decode_bf16, e4m3fn_units, encode_bf16_rne, fp32_multiply_rn, i64_to_f32_rn,
};

const TILE_M: usize = 16;
const TILE_N: usize = 8;
const TILE_K: usize = 32;

type AFragment = (u32, u32, u32, u32);
type BFragment = (u32, u32);
type Totals = (i64, i64, i64, i64);

#[inline(always)]
fn warp_and_tile() -> (u32, u32, u32) {
    let lane: u32;
    let tile_n: u32;
    let tile_m: u32;
    // SAFETY: Reads the calling thread's lane and CTA coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {tile_m}, %ctaid.y;",
            lane = out(reg32) lane,
            tile_n = out(reg32) tile_n,
            tile_m = out(reg32) tile_m,
            options(nomem, nostack),
        )
    };
    (lane, tile_n, tile_m)
}

/// Pack four in-range row-major E4M3 bytes into one register, zero-padding K tails.
///
/// # Safety
/// If `row < row_count`, `matrix` must cover `row_count * k` readable bytes and
/// all row and column offset arithmetic must fit `usize`.
#[inline(always)]
unsafe fn load_e4m3x4(
    matrix: *const u8,
    row: usize,
    row_count: usize,
    k: usize,
    column: usize,
) -> u32 {
    if row >= row_count {
        return 0;
    }

    let row_start = row * k;
    let mut packed = 0_u32;
    let mut byte = 0_usize;
    while byte < 4 {
        let input_column = column + byte;
        if input_column < k {
            // SAFETY: The caller supplies row-major storage for row_count rows of k bytes.
            let value = unsafe { *matrix.add(row_start + input_column) };
            packed |= (value as u32) << (byte * 8);
        }
        byte += 1;
    }
    packed
}

/// Split four E4M3 values into exact signed base-128 digit fragments.
///
/// Each finite code is represented in units of 1/512 by three signed digits:
/// `d0 + 128*d1 + 16384*d2`.
#[inline(always)]
fn split_e4m3x4(codes: u32) -> (u32, u32, u32) {
    let mut packed_d0 = 0_u32;
    let mut packed_d1 = 0_u32;
    let mut packed_d2 = 0_u32;
    let mut byte = 0_u32;
    while byte < 4 {
        let code = (codes >> (byte * 8)) as u8;
        let units = e4m3fn_units(code);
        let sign = if units < 0 { -1_i32 } else { 1_i32 };
        let magnitude = units.abs() as u32;
        let d0 = (magnitude & 0x7f) as i32 * sign;
        let d1 = ((magnitude >> 7) & 0x7f) as i32 * sign;
        let d2 = (magnitude >> 14) as i32 * sign;
        packed_d0 |= (d0 as i8 as u8 as u32) << (byte * 8);
        packed_d1 |= (d1 as i8 as u8 as u32) << (byte * 8);
        packed_d2 |= (d2 as i8 as u8 as u32) << (byte * 8);
        byte += 1;
    }
    (packed_d0, packed_d1, packed_d2)
}

#[inline(always)]
fn mma_s8(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32) -> (i32, i32, i32, i32) {
    let (mut d0, mut d1, mut d2, mut d3) = (0_i32, 0_i32, 0_i32, 0_i32);
    // SAFETY: All lanes execute the same m16n8k32 signed-INT8 MMA with valid fragments;
    // the 32-term signed-byte partial sum fits in i32.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d0,
            d1 = inout(reg32) d1,
            d2 = inout(reg32) d2,
            d3 = inout(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            options(nomem, nostack),
        );
    }
    (d0, d1, d2, d3)
}

#[inline(always)]
fn accumulate_digit_pair(totals: Totals, a: AFragment, b: BFragment, shift: u32) -> Totals {
    let (p0, p1, p2, p3) = mma_s8(a.0, a.1, a.2, a.3, b.0, b.1);
    (
        add_s64(totals.0, (p0 as i64) << shift),
        add_s64(totals.1, (p1 as i64) << shift),
        add_s64(totals.2, (p2 as i64) << shift),
        add_s64(totals.3, (p3 as i64) << shift),
    )
}

/// Store one exact signed-integer dot using the reference kernel's scale and BF16 order.
///
/// # Safety
/// For an in-range row and column, `sa`, `sw`, `out`, and `unrounded` must have
/// the corresponding readable scale and writable output elements; all offsets fit `usize`.
#[inline(always)]
unsafe fn store_scaled_output(
    sa: *const f32,
    sw: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    row: usize,
    column: usize,
    m: usize,
    n: usize,
    total: i64,
) {
    if row >= m || column >= n {
        return;
    }

    // SAFETY: This valid output coordinate has its corresponding scale entries.
    let (row_scale, channel_scale) = unsafe { (*sa.add(row), decode_bf16(*sw.add(column))) };
    let dot_units = i64_to_f32_rn(total);
    let dot = fp32_multiply_rn(dot_units, 1.0 / 262_144.0);
    let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
    let output_index = row * n + column;
    // SAFETY: The in-range coordinate uniquely owns these row-major output slots.
    unsafe {
        unrounded.add(output_index).write(value);
        out.add(output_index).write(encode_bf16_rne(value));
    }
}

/// Compute an exact E4M3FN row-major projection for prefill-sized row tiles.
///
/// Each warp computes one 16x8 tile. Three signed base-128 operand digits are
/// combined through nine exact signed-INT8 MMAs per K32 tile, retaining the exact
/// integer dot before the same FP32 scaling and BF16 RNE as `fp8_linear_exact`.
///
/// # Safety
/// Launch `grid = [ceil(n / 8), ceil(m / 16), 1]`, `block = [32, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `a` and `w` must each
/// cover row-major matrices of `m * k` and `n * k` readable E4M3FN bytes. Codes
/// must be finite (not `0x7f` or `0xff`). `sa` must cover `m` readable finite
/// positive FP32 scales, and `sw` must cover `n` readable finite positive BF16
/// scales. `out` and `unrounded` must cover `m * n` writable BF16 and FP32 values.
/// All products and index arithmetic must fit the device address space. Pointers
/// must have their element alignment, be pairwise disjoint, and remain live until
/// kernel completion. The exact integer dot fits signed 64-bit for the stated K.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_prefill_exact(
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
    let (lane, tile_n, tile_m) = warp_and_tile();
    let lane_group = (lane >> 2) as usize;
    let thread_in_group = (lane & 3) as usize;
    let row_start = tile_m as usize * TILE_M;
    let column_start = tile_n as usize * TILE_N;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let mut totals = (0_i64, 0_i64, 0_i64, 0_i64);

    for k_tile in 0..k.div_ceil(TILE_K as u32) {
        let k_start = k_tile as usize * TILE_K;
        let column = k_start + thread_in_group * 4;
        let a_row0 = row_start + lane_group;
        let a_row1 = a_row0 + 8;
        let b_row = column_start + lane_group;
        // SAFETY: The launch contract provides row-major extents. The loader returns zero
        // for out-of-range rows and K columns so every lane can issue all nine MMA calls.
        let (a0, a1, a2, a3, b0, b1) = unsafe {
            (
                load_e4m3x4(a, a_row0, m_usize, k_usize, column),
                load_e4m3x4(a, a_row1, m_usize, k_usize, column),
                load_e4m3x4(a, a_row0, m_usize, k_usize, column + 16),
                load_e4m3x4(a, a_row1, m_usize, k_usize, column + 16),
                load_e4m3x4(w, b_row, n_usize, k_usize, column),
                load_e4m3x4(w, b_row, n_usize, k_usize, column + 16),
            )
        };
        let (a0d0, a0d1, a0d2) = split_e4m3x4(a0);
        let (a1d0, a1d1, a1d2) = split_e4m3x4(a1);
        let (a2d0, a2d1, a2d2) = split_e4m3x4(a2);
        let (a3d0, a3d1, a3d2) = split_e4m3x4(a3);
        let (b0d0, b0d1, b0d2) = split_e4m3x4(b0);
        let (b1d0, b1d1, b1d2) = split_e4m3x4(b1);
        let a_d0 = (a0d0, a1d0, a2d0, a3d0);
        let a_d1 = (a0d1, a1d1, a2d1, a3d1);
        let a_d2 = (a0d2, a1d2, a2d2, a3d2);
        let b_d0 = (b0d0, b1d0);
        let b_d1 = (b0d1, b1d1);
        let b_d2 = (b0d2, b1d2);

        totals = accumulate_digit_pair(totals, a_d0, b_d0, 0);
        totals = accumulate_digit_pair(totals, a_d0, b_d1, 7);
        totals = accumulate_digit_pair(totals, a_d0, b_d2, 14);
        totals = accumulate_digit_pair(totals, a_d1, b_d0, 7);
        totals = accumulate_digit_pair(totals, a_d1, b_d1, 14);
        totals = accumulate_digit_pair(totals, a_d1, b_d2, 21);
        totals = accumulate_digit_pair(totals, a_d2, b_d0, 14);
        totals = accumulate_digit_pair(totals, a_d2, b_d1, 21);
        totals = accumulate_digit_pair(totals, a_d2, b_d2, 28);
    }

    let lane_column_start = column_start + 2 * thread_in_group;
    // SAFETY: The helper guards tail rows and columns before reading scales or storing.
    unsafe {
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            row_start + lane_group,
            lane_column_start,
            m_usize,
            n_usize,
            totals.0,
        );
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            row_start + lane_group,
            lane_column_start + 1,
            m_usize,
            n_usize,
            totals.1,
        );
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            row_start + lane_group + 8,
            lane_column_start,
            m_usize,
            n_usize,
            totals.2,
        );
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            row_start + lane_group + 8,
            lane_column_start + 1,
            m_usize,
            n_usize,
            totals.3,
        );
    }
}
