use core::arch::asm;

const TILE_M: usize = 16;
const TILE_N: usize = 8;
const TILE_K: usize = 64;
const K_GROUP: usize = 16;

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

/// Load eight low-nibble-first FP4 values into one warp-MMA register.
///
/// # Safety
/// If `row < row_count`, `matrix` must cover `row_count * (k / 2)` readable bytes;
/// row, column, and byte offset arithmetic must fit `usize`.
#[inline(always)]
unsafe fn load_e2m1x8(
    matrix: *const u8,
    row: usize,
    row_count: usize,
    k: usize,
    column: usize,
) -> u32 {
    if row >= row_count {
        return 0;
    }

    let row_start = row * (k / 2);
    let mut packed = 0_u32;
    let mut pair = 0_usize;
    while pair < 4 {
        let input_column = column + pair * 2;
        if input_column < k {
            // SAFETY: The caller supplies row-major low-first packed FP4 storage.
            let byte = unsafe { *matrix.add(row_start + input_column / 2) };
            packed |= (byte as u32) << (pair * 8);
        }
        pair += 1;
    }
    packed
}

/// Pack four consecutive UE4M3 group scales in little-endian byte order.
///
/// # Safety
/// If `row < row_count`, `scales` must cover `row_count * groups_per_row` readable
/// bytes; row and group offset arithmetic must fit `usize`.
#[inline(always)]
unsafe fn load_ue4m3x4(
    scales: *const u8,
    row: usize,
    row_count: usize,
    groups_per_row: usize,
    first_group: usize,
) -> u32 {
    let mut packed = 0_u32;
    let mut group = 0_usize;
    while group < 4 {
        let index = first_group + group;
        let code = if row < row_count && index < groups_per_row {
            // SAFETY: The caller supplies one byte for each row/group scale.
            unsafe { *scales.add(row * groups_per_row + index) }
        } else {
            0x38
        };
        packed |= (code as u32) << (group * 8);
        group += 1;
    }
    packed
}

#[inline(always)]
fn mma_nvfp4(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    scale_a: u32,
    scale_b: u32,
    accumulators: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let (mut d0, mut d1, mut d2, mut d3) = accumulators;
    // SAFETY: Every lane executes the qualified SM120a warp-level block-scaled MMA.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}}, {scale_a}, {{0, 0}}, {scale_b}, {{0, 0}};",
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
            scale_a = in(reg32) scale_a,
            scale_b = in(reg32) scale_b,
            options(nomem, nostack),
        );
    }
    (d0, d1, d2, d3)
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
fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Scale and store one in-range NVFP4 GEMM result.
///
/// # Safety
/// `out` and `unrounded` must each cover `m * n` writable elements. Their pointers
/// must be disjoint and output index arithmetic must fit `usize`.
#[inline(always)]
unsafe fn store_scaled_output(
    out: *mut u16,
    unrounded: *mut f32,
    row: usize,
    column: usize,
    m: usize,
    n: usize,
    accumulator: f32,
    global_factor: f32,
) {
    if row >= m || column >= n {
        return;
    }

    let value = fp32_multiply_rn(accumulator, global_factor);
    let index = row * n + column;
    // SAFETY: The caller supplies disjoint row-major outputs covering m * n items.
    unsafe {
        unrounded.add(index).write(value);
        out.add(index).write(encode_bf16_rne(value));
    }
}

/// Compute row-major `out = A * W^T * global_factor` from block-scaled NVFP4 data.
///
/// A and W store two low-nibble-first E2M1 values per byte, with shapes `[m, k]`
/// and `[n, k]`. `sa` and `sw` store one unsigned E4M3 scale per 16 values along K.
/// The kernel reads the logical matrices directly, pads incomplete 64-value MMA
/// tiles with zero data and unit scales, then stores FP32 diagnostics and BF16 RNE.
///
/// # Safety
/// Launch `grid = [ceil(n / 8), ceil(m / 16), 1]` and `block = [32, 1, 1]`.
/// Require `1 <= m <= 2048`, `1 <= n <= 32768`, and `k` a multiple of 16 in
/// `16..=32768`; dimension products, padded K offsets, and launch dimensions must
/// fit the device address space and hardware limits. `a` and `w` must each cover
/// respectively `m * (k / 2)` and `n * (k / 2)` readable bytes of low-first packed
/// E2M1 values. `sa` and `sw` must cover respectively `m * (k / 16)` and
/// `n * (k / 16)` readable unsigned E4M3 bytes with codes in `0..=126`.
/// `global_factor` must be finite and positive. `out` and `unrounded` must each
/// cover `m * n` writable BF16 and FP32 elements. All pointers must be correctly
/// aligned, mutually disjoint, and live until completion. The host must reject
/// nonfinite FP32 results before accepting the output.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_linear(
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
    let (lane, tile_n, tile_m) = warp_and_tile();
    let lane_group = (lane >> 2) as usize;
    let thread_in_group = (lane & 3) as usize;
    let row_start = tile_m as usize * TILE_M;
    let column_start = tile_n as usize * TILE_N;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let groups_per_a_row = k_usize / K_GROUP;
    let groups_per_b_row = groups_per_a_row;
    let mut accumulators = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

    for k_tile in 0..k.div_ceil(TILE_K as u32) {
        let k_start = k_tile as usize * TILE_K;
        let a_row0 = row_start + lane_group;
        let a_row1 = a_row0 + 8;
        let b_row = column_start + lane_group;
        let a_column0 = k_start + thread_in_group * 8;
        let a_column1 = a_column0 + 32;
        let first_scale_group = k_start / K_GROUP;

        // SAFETY: Logical dimensions define the matrix extents; helpers zero-pad rows
        // and K columns and supply neutral scales beyond the declared K extent.
        let (a0, a1, a2, a3, b0, b1, scale_a, scale_b) = unsafe {
            let a0 = load_e2m1x8(a, a_row0, m_usize, k_usize, a_column0);
            let a1 = load_e2m1x8(a, a_row1, m_usize, k_usize, a_column0);
            let a2 = load_e2m1x8(a, a_row0, m_usize, k_usize, a_column1);
            let a3 = load_e2m1x8(a, a_row1, m_usize, k_usize, a_column1);
            let b0 = load_e2m1x8(w, b_row, n_usize, k_usize, a_column0);
            let b1 = load_e2m1x8(w, b_row, n_usize, k_usize, a_column1);
            let scale_a = match thread_in_group {
                0 => load_ue4m3x4(sa, a_row0, m_usize, groups_per_a_row, first_scale_group),
                1 => load_ue4m3x4(sa, a_row1, m_usize, groups_per_a_row, first_scale_group),
                _ => 0,
            };
            let scale_b = if thread_in_group == 0 {
                load_ue4m3x4(sw, b_row, n_usize, groups_per_b_row, first_scale_group)
            } else {
                0
            };
            (a0, a1, a2, a3, b0, b1, scale_a, scale_b)
        };
        accumulators = mma_nvfp4(a0, a1, a2, a3, b0, b1, scale_a, scale_b, accumulators);
    }

    let row0 = row_start + lane_group;
    let row1 = row0 + 8;
    let column0 = column_start + thread_in_group * 2;
    // SAFETY: The lane mapping assigns four distinct output coordinates; the store
    // helper guards partial M and N tiles before indexing the disjoint outputs.
    unsafe {
        store_scaled_output(
            out,
            unrounded,
            row0,
            column0,
            m_usize,
            n_usize,
            accumulators.0,
            global_factor,
        );
        store_scaled_output(
            out,
            unrounded,
            row0,
            column0 + 1,
            m_usize,
            n_usize,
            accumulators.1,
            global_factor,
        );
        store_scaled_output(
            out,
            unrounded,
            row1,
            column0,
            m_usize,
            n_usize,
            accumulators.2,
            global_factor,
        );
        store_scaled_output(
            out,
            unrounded,
            row1,
            column0 + 1,
            m_usize,
            n_usize,
            accumulators.3,
            global_factor,
        );
    }
}
