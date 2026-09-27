use core::arch::asm;

use super::fp8_linear_exact::{
    add_s64, decode_bf16, e4m3fn_units, encode_bf16_rne, fp32_multiply_rn, i64_to_f32_rn,
};

const TILE_CHANNELS: usize = 16;
const TILE_ROWS: usize = 8;
const TILE_K: usize = 32;

type AFragment = (u32, u32, u32, u32);
type BFragment = (u32, u32);
type Totals = (i64, i64, i64, i64);

#[inline(always)]
fn lane_and_tiles() -> (u32, u32, u32) {
    let lane: u32;
    let tile_channel: u32;
    let tile_row: u32;
    // SAFETY: Reads the calling thread's lane and CTA coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {tile_channel}, %ctaid.x;",
            "mov.u32 {tile_row}, %ctaid.y;",
            lane = out(reg32) lane,
            tile_channel = out(reg32) tile_channel,
            tile_row = out(reg32) tile_row,
            options(nomem, nostack),
        )
    };
    (lane, tile_channel, tile_row)
}

/// Pack four row-major E4M3 bytes, zero-padding K columns and out-of-range rows.
///
/// # Safety
/// If `row < row_count`, `matrix` must cover `row_count * k` readable bytes, and
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
            // SAFETY: The launch contract guarantees row-major storage for all input rows.
            let value = unsafe { *matrix.add(row_start + input_column) };
            packed |= (value as u32) << (byte * 8);
        }
        byte += 1;
    }
    packed
}

/// Split four E4M3 codes into exact signed base-128 digit fragments.
///
/// Each finite code is represented in units of `1/512` as
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
    // SAFETY: Every lane issues this same m16n8k32 signed-INT8 MMA with valid fragments;
    // its 32-term signed-byte partial sums fit in i32.
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

/// Store one exact dot after the reference FP32 scale sequence and BF16 RNE.
///
/// # Safety
/// For an in-range token row and output channel, the scale pointers must contain
/// those elements and both output pointers must contain the corresponding
/// row-major slots; all offsets must fit `usize`.
#[inline(always)]
unsafe fn store_scaled_output(
    sa: *const f32,
    sw: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    row: usize,
    channel: usize,
    m: usize,
    n: usize,
    total: i64,
) {
    if row >= m || channel >= n {
        return;
    }

    // SAFETY: This valid coordinate has its corresponding row and channel scales.
    let (row_scale, channel_scale) = unsafe { (*sa.add(row), decode_bf16(*sw.add(channel))) };
    let dot_units = i64_to_f32_rn(total);
    let dot = fp32_multiply_rn(dot_units, 1.0 / 262_144.0);
    let value = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
    let output_index = row * n + channel;
    // SAFETY: The in-range coordinate uniquely owns these row-major output slots.
    unsafe {
        unrounded.add(output_index).write(value);
        out.add(output_index).write(encode_bf16_rne(value));
    }
}

/// Compute the exact E4M3 projection with weights as MMA rows and input rows as MMA columns.
///
/// Each warp computes a 16-channel by 8-token tile. Three signed base-128 digits
/// and nine signed-INT8 MMAs per K32 tile reconstruct each exact dot before the
/// existing scale order and BF16 RNE. The result is stored in token-major order.
///
/// # Safety
/// Launch `grid = [ceil(n / 16), ceil(m / 8), 1]`, `block = [32, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `a` and `w` must
/// cover row-major matrices of `m * k` and `n * k` readable E4M3FN bytes. Codes
/// must be finite (not `0x7f` or `0xff`). `sa` must cover `m` readable finite
/// positive FP32 scales; `sw` must cover `n` readable finite positive BF16
/// scales. `out` and `unrounded` must cover `m * n` writable BF16 and FP32
/// values. All products and index arithmetic must fit the device address space.
/// Pointers must have element alignment, be pairwise disjoint, and remain live
/// until kernel completion. Every warp lane must execute the same K loop and all
/// nine MMA instructions for each K tile. The exact dot fits signed 64-bit.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_verify_exact(
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
    let (lane, tile_channel, tile_row) = lane_and_tiles();
    let lane_group = (lane >> 2) as usize;
    let thread_in_group = (lane & 3) as usize;
    let channel_start = tile_channel as usize * TILE_CHANNELS;
    let row_start = tile_row as usize * TILE_ROWS;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let mut totals = (0_i64, 0_i64, 0_i64, 0_i64);

    for k_tile in 0..k.div_ceil(TILE_K as u32) {
        let k_start = k_tile as usize * TILE_K;
        let k_column = k_start + thread_in_group * 4;
        let a_channel0 = channel_start + lane_group;
        let a_channel1 = a_channel0 + 8;
        let b_row = row_start + lane_group;
        // SAFETY: The launch contract provides row-major extents. The loader zero-fills
        // channel/K tails so every lane can participate in all nine MMA instructions.
        let (a0, a1, a2, a3, b0, b1) = unsafe {
            (
                load_e4m3x4(w, a_channel0, n_usize, k_usize, k_column),
                load_e4m3x4(w, a_channel1, n_usize, k_usize, k_column),
                load_e4m3x4(w, a_channel0, n_usize, k_usize, k_column + 16),
                load_e4m3x4(w, a_channel1, n_usize, k_usize, k_column + 16),
                load_e4m3x4(a, b_row, m_usize, k_usize, k_column),
                load_e4m3x4(a, b_row, m_usize, k_usize, k_column + 16),
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

    let token_row0 = row_start + thread_in_group * 2;
    let channel0 = channel_start + lane_group;
    let channel1 = channel0 + 8;
    // SAFETY: The helper guards token-row and channel tails before loading scales or storing.
    unsafe {
        store_scaled_output(
            sa, sw, out, unrounded, token_row0, channel0, m_usize, n_usize, totals.0,
        );
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            token_row0 + 1,
            channel0,
            m_usize,
            n_usize,
            totals.1,
        );
        store_scaled_output(
            sa, sw, out, unrounded, token_row0, channel1, m_usize, n_usize, totals.2,
        );
        store_scaled_output(
            sa,
            sw,
            out,
            unrounded,
            token_row0 + 1,
            channel1,
            m_usize,
            n_usize,
            totals.3,
        );
    }
}
