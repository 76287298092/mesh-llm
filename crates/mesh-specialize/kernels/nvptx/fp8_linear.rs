use core::arch::asm;

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

/// Pack four in-range row-major E4M3 bytes into one warp-MMA register.
///
/// # Safety
/// If `row < row_count`, `matrix` must point to `row_count * k` readable bytes, and
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
fn fp32_to_fp64_rn(value: f32) -> f64 {
    let converted: f64;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.rn.f64.f32 {converted}, {value};",
            converted = out(reg64) converted,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
fn fp64_add_rn(left: f64, right: f64) -> f64 {
    let sum: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f64 {sum}, {left}, {right};",
            sum = out(reg64) sum,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
fn fp64_to_fp32_rn(value: f64) -> f32 {
    let converted: f32;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.rn.f32.f64 {converted}, {value};",
            converted = out(reg32) converted,
            value = in(reg64) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
fn mma_e4m3(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    accumulators: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let (mut d0, mut d1, mut d2, mut d3) = accumulators;
    // SAFETY: All lanes execute the same verified-shape E4M3 MMA with the documented
    // four A and two B registers arranged by the warp fragment mapping.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ",
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
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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

/// Store one scaled matrix result when its row and column are in range.
///
/// # Safety
/// `sa`, `sw`, `out`, and `unrounded` must point to valid allocations covering the
/// declared dimensions, remain live, and be disjoint. The output index arithmetic
/// must fit `usize`.
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
    accumulator: f32,
) {
    if row >= m || column >= n {
        return;
    }

    // SAFETY: The caller provides scales for every in-range row and output column.
    let (row_scale, column_scale) = unsafe { (*sa.add(row), decode_bf16(*sw.add(column))) };
    let scaled = fp32_multiply_rn(fp32_multiply_rn(accumulator, row_scale), column_scale);
    let index = row * n + column;
    // SAFETY: The caller provides disjoint output allocations covering m * n elements.
    unsafe {
        unrounded.add(index).write(scaled);
        out.add(index).write(encode_bf16_rne(scaled));
    }
}

/// Compute a tiled row-major E4M3FN linear layer with row and output-channel scales.
///
/// The kernel evaluates `out[m, n] = sum_k(a[m, k] * w[n, k]) * sa[m] * sw[n]`.
/// `a` and `w` are consumed directly as packed E4M3 fragments by warp MMA; no packed
/// staging matrix or shared-memory transpose is used. It stores both the FP32 result
/// and the same result rounded to BF16 with round-to-nearest-even.
///
/// # Safety
/// Launch `grid = [ceil(n / 8), ceil(m / 16), 1]` and `block = [32, 1, 1]`, with
/// nonzero `m`, `n`, and `k`. All dimension products and launch dimensions must fit
/// the device address space and hardware limits, including the padded K-index arithmetic.
/// `a` must point to at least `m * k`
/// readable E4M3FN bytes and `w` to at least `n * k` readable E4M3FN bytes in row-major
/// order. `sa` must contain at least `m` readable `f32` row scales, and `sw` at least
/// `n` readable BF16 output-channel scales. The host must validate all FP8 values and
/// scales are finite. `out` and `unrounded` must each cover `m * n` writable BF16 and
/// `f32` values, respectively. Pointers must be correctly aligned for their element
/// types, mutually disjoint, and live until kernel completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_linear(
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
    // SAFETY: The public launch contract is passed unchanged to the shared kernel body.
    unsafe { fp8_linear_impl::<false>(a, w, sa, sw, out, unrounded, m, n, k) };
}

/// Compute the FP8 linear layer with an experimental FP64 tile-total accumulator.
///
/// Each K=32 warp-MMA tile still accumulates and rounds in FP32. The four rounded
/// tile results are then added to FP64 running totals using round-to-nearest-even;
/// each total is rounded to FP32 before the existing row/channel scaling and BF16
/// output boundary. This reduces drift between FP32 tile accumulation and the
/// independent FP64-dot reference, but is not an exact FP64 dot product because
/// the MMA instructions round each tile in FP32.
///
/// # Safety
/// Use the same launch geometry, dimensions, extents, alignment, disjointness, and
/// lifetime requirements documented for [`fp8_linear`].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_linear_wide(
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
    // SAFETY: The public launch contract is passed unchanged to the shared kernel body.
    unsafe { fp8_linear_impl::<true>(a, w, sa, sw, out, unrounded, m, n, k) };
}

/// Shared layout and output implementation for the regular and wide profiles.
///
/// # Safety
/// The caller must uphold the pointer, extent, alignment, disjointness, launch,
/// and lifetime requirements documented on the exported kernels.
#[inline(always)]
unsafe fn fp8_linear_impl<const WIDE: bool>(
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
    let row_start = tile_m as usize * 16;
    let column_start = tile_n as usize * 8;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let mut accumulators = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut wide_totals = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    for k_tile in 0..k.div_ceil(32) {
        let k_start = k_tile as usize * 32 + thread_in_group * 4;
        let a_row0 = row_start + lane_group;
        let a_row1 = a_row0 + 8;
        let b_row = column_start + lane_group;
        // For lane=(4*g+t), A registers 0/1 use rows g/g+8 at K+4t..K+4t+3,
        // A registers 2/3 use those rows at K+16+4t..K+19+4t; B uses row g
        // at K+4t..K+4t+3 and K+16+4t..K+19+4t.
        // SAFETY: The matrix extents and offset arithmetic satisfy the public launch
        // contract; out-of-range rows and padded K columns are returned as zero.
        let (a0, a1, a2, a3, b0, b1) = unsafe {
            (
                load_e4m3x4(a, a_row0, m_usize, k_usize, k_start),
                load_e4m3x4(a, a_row1, m_usize, k_usize, k_start),
                load_e4m3x4(a, a_row0, m_usize, k_usize, k_start + 16),
                load_e4m3x4(a, a_row1, m_usize, k_usize, k_start + 16),
                load_e4m3x4(w, b_row, n_usize, k_usize, k_start),
                load_e4m3x4(w, b_row, n_usize, k_usize, k_start + 16),
            )
        };
        let tile_accumulators = mma_e4m3(
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            if WIDE {
                (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32)
            } else {
                accumulators
            },
        );
        if WIDE {
            wide_totals = (
                fp64_add_rn(wide_totals.0, fp32_to_fp64_rn(tile_accumulators.0)),
                fp64_add_rn(wide_totals.1, fp32_to_fp64_rn(tile_accumulators.1)),
                fp64_add_rn(wide_totals.2, fp32_to_fp64_rn(tile_accumulators.2)),
                fp64_add_rn(wide_totals.3, fp32_to_fp64_rn(tile_accumulators.3)),
            );
        } else {
            accumulators = tile_accumulators;
        }
    }

    let accumulators = if WIDE {
        (
            fp64_to_fp32_rn(wide_totals.0),
            fp64_to_fp32_rn(wide_totals.1),
            fp64_to_fp32_rn(wide_totals.2),
            fp64_to_fp32_rn(wide_totals.3),
        )
    } else {
        accumulators
    };
    let lane_column_start = column_start + 2 * thread_in_group;
    // SAFETY: The per-tile output mapping assigns each lane four distinct in-range
    // coordinates when they pass the helper's dimension guards.
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
            accumulators.0,
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
            accumulators.1,
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
            accumulators.2,
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
            accumulators.3,
        );
    }
}
