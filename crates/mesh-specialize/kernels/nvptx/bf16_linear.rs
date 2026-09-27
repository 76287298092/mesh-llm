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

/// Pack two in-range row-major BF16 values into one warp-MMA register.
///
/// # Safety
/// If `row < row_count`, `matrix` must point to `row_count * k` readable `u16` values,
/// and all row and column offset arithmetic must fit `usize`.
#[inline(always)]
unsafe fn load_bf16x2(
    matrix: *const u16,
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
    if column < k {
        // SAFETY: The caller supplies row-major storage for row_count rows of k values.
        packed = unsafe { *matrix.add(row_start + column) } as u32;
    }
    if column + 1 < k {
        // SAFETY: The second BF16 value is within the same validated row extent.
        let high = unsafe { *matrix.add(row_start + column + 1) } as u32;
        packed |= high << 16;
    }
    packed
}

#[inline(always)]
fn mma_bf16(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    accumulators: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let (mut d0, mut d1, mut d2, mut d3) = accumulators;
    // SAFETY: All lanes execute the qualified 16x8x16 BF16 warp MMA with the documented
    // four A and two B registers arranged by the row/column fragment mapping.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
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
fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Store one in-range BF16 linear result in both FP32 and rounded BF16 form.
///
/// # Safety
/// `out` and `unrounded` must cover disjoint `m * n` allocations, remain live, and
/// have index arithmetic that fits `usize`.
#[inline(always)]
unsafe fn store_output(
    out: *mut u16,
    unrounded: *mut f32,
    row: usize,
    column: usize,
    n: usize,
    accumulator: f32,
) {
    let index = row * n + column;
    // SAFETY: The caller supplies valid, disjoint output extents and an in-range index.
    unsafe {
        unrounded.add(index).write(accumulator);
        out.add(index).write(encode_bf16_rne(accumulator));
    }
}

/// Compute one row-major BF16 linear layer using warp-level tensor-core MMA.
///
/// The kernel computes `out[m, n] = sum_k(a[m, k] * w[n, k])` for matrices stored
/// as `A[m, k]` and `W[n, k]`. It performs no scaling or activation quantization and
/// writes the FP32 accumulator alongside its BF16 round-to-nearest-even encoding.
///
/// # Safety
/// Launch `grid = [ceil(n / 8), ceil(m / 16), 1]` and `block = [32, 1, 1]`, with
/// nonzero `m`, `n`, and `k`. All dimension products, padded K offsets, and launch
/// dimensions must fit the device address space and hardware limits. `a` must point
/// to at least `m * k` readable BF16 values and `w` to at least `n * k` readable BF16
/// values in row-major order. The host must validate every input value is finite.
/// `out` and `unrounded` must each cover `m * n` writable BF16 and `f32` values,
/// respectively. Pointers must be correctly aligned for their element types,
/// mutually disjoint, and live until kernel completion. The host must reject any
/// non-finite or overflowing output.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bf16_linear(
    a: *const u16,
    w: *const u16,
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

    for k_tile in 0..k.div_ceil(16) {
        let k_start = k_tile as usize * 16 + thread_in_group * 2;
        let a_row0 = row_start + lane_group;
        let a_row1 = a_row0 + 8;
        let b_row = column_start + lane_group;
        // For lane=(4*g+t), A registers 0/1 use rows g/g+8 at K+2t..K+2t+1,
        // A registers 2/3 use those rows at K+8+2t..K+9+2t; B uses row g at
        // K+2t..K+2t+1 and K+8+2t..K+9+2t.
        // SAFETY: Matrix extents and offset arithmetic satisfy the launch contract;
        // out-of-range rows and padded K values are returned as zero.
        let (a0, a1, a2, a3, b0, b1) = unsafe {
            (
                load_bf16x2(a, a_row0, m_usize, k_usize, k_start),
                load_bf16x2(a, a_row1, m_usize, k_usize, k_start),
                load_bf16x2(a, a_row0, m_usize, k_usize, k_start + 8),
                load_bf16x2(a, a_row1, m_usize, k_usize, k_start + 8),
                load_bf16x2(w, b_row, n_usize, k_usize, k_start),
                load_bf16x2(w, b_row, n_usize, k_usize, k_start + 8),
            )
        };
        accumulators = mma_bf16(a0, a1, a2, a3, b0, b1, accumulators);
    }

    let lane_column_start = column_start + 2 * thread_in_group;
    // SAFETY: This lane owns these four distinct tile outputs; the dimension guards
    // ensure each write maps to valid input-independent scale-free output extents.
    unsafe {
        if row_start + lane_group < m_usize && lane_column_start < n_usize {
            store_output(
                out,
                unrounded,
                row_start + lane_group,
                lane_column_start,
                n_usize,
                accumulators.0,
            );
        }
        if row_start + lane_group < m_usize && lane_column_start + 1 < n_usize {
            store_output(
                out,
                unrounded,
                row_start + lane_group,
                lane_column_start + 1,
                n_usize,
                accumulators.1,
            );
        }
        if row_start + lane_group + 8 < m_usize && lane_column_start < n_usize {
            store_output(
                out,
                unrounded,
                row_start + lane_group + 8,
                lane_column_start,
                n_usize,
                accumulators.2,
            );
        }
        if row_start + lane_group + 8 < m_usize && lane_column_start + 1 < n_usize {
            store_output(
                out,
                unrounded,
                row_start + lane_group + 8,
                lane_column_start + 1,
                n_usize,
                accumulators.3,
            );
        }
    }
}
