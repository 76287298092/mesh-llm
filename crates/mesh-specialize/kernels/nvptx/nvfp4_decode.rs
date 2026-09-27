use super::nvfp4_linear::{
    load_e2m1x8, load_ue4m3x4, mma_nvfp4, store_scaled_output, warp_and_tile,
};

const OUTPUT_ROWS_PER_TILE: usize = 16;
const MMA_K: usize = 64;
const SCALE_GROUP_K: usize = 16;

/// Compute one NVFP4 projection token by transposing the existing 16x8 MMA operands.
///
/// Weight rows occupy logical MMA A. The single activation token occupies logical
/// MMA B column zero; the other seven B columns contain zero data. The kernel stores
/// only that token column, in the same `[m, n]` output order as `nvfp4_linear`.
///
/// # Safety
/// Launch `grid = [ceil(n / 16), 1, 1]` and `block = [32, 1, 1]`. Require `m == 1`,
/// `1 <= n <= 32768`, and `k` a multiple of 16 in `16..=32768`; dimension products,
/// padded K offsets, and launch dimensions must fit the device address space and
/// hardware limits. `a` and `w` must cover respectively `m * (k / 2)` and
/// `n * (k / 2)` readable bytes of low-first packed E2M1 values. `sa` and `sw` must
/// cover respectively `m * (k / 16)` and `n * (k / 16)` readable unsigned E4M3
/// bytes with codes in `0..=126`. The base addresses of `a`, `w`, `sa`, and `sw`
/// must be four-byte aligned for packed loads. `global_factor` must be finite and
/// positive. `out` and `unrounded` must cover `m * n` writable BF16 and FP32
/// elements. All pointers must be correctly aligned, mutually disjoint, and live
/// until completion. The host must reject nonfinite FP32 results before accepting
/// the output.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_decode(
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
    // This shape guard is uniform across the CTA, so every lane either reaches every
    // warp-level MMA or returns together.
    if m != 1 {
        return;
    }

    let (lane, output_tile, _) = warp_and_tile();
    let lane_group = (lane >> 2) as usize;
    let thread_in_group = (lane & 3) as usize;
    let output_start = output_tile as usize * OUTPUT_ROWS_PER_TILE;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let groups_per_row = k_usize / SCALE_GROUP_K;
    let mut accumulators = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

    for k_tile in 0..k.div_ceil(MMA_K as u32) {
        let k_start = k_tile as usize * MMA_K;
        let weight_row0 = output_start + lane_group;
        let weight_row1 = weight_row0 + 8;
        let k_column0 = k_start + thread_in_group * 8;
        let k_column1 = k_column0 + 32;
        let first_scale_group = k_start / SCALE_GROUP_K;

        // SAFETY: The launch contract supplies packed `[1, k]` activations and
        // `[n, k]` weights. The shared helpers zero-pad inactive rows/K values and
        // use unit scale codes for groups outside the logical K extent.
        let (a0, a1, a2, a3, b0, b1, scale_a, scale_b) = unsafe {
            let a0 = load_e2m1x8(w, weight_row0, n_usize, k_usize, k_column0);
            let a1 = load_e2m1x8(w, weight_row1, n_usize, k_usize, k_column0);
            let a2 = load_e2m1x8(w, weight_row0, n_usize, k_usize, k_column1);
            let a3 = load_e2m1x8(w, weight_row1, n_usize, k_usize, k_column1);
            let b0 = load_e2m1x8(a, lane_group, 1, k_usize, k_column0);
            let b1 = load_e2m1x8(a, lane_group, 1, k_usize, k_column1);
            let scale_a = match thread_in_group {
                0 => load_ue4m3x4(sw, weight_row0, n_usize, groups_per_row, first_scale_group),
                1 => load_ue4m3x4(sw, weight_row1, n_usize, groups_per_row, first_scale_group),
                _ => 0,
            };
            let scale_b = if thread_in_group == 0 {
                load_ue4m3x4(sa, lane_group, 1, groups_per_row, first_scale_group)
            } else {
                0
            };
            (a0, a1, a2, a3, b0, b1, scale_a, scale_b)
        };
        accumulators = mma_nvfp4(a0, a1, a2, a3, b0, b1, scale_a, scale_b, accumulators);
    }

    // In the existing fragment mapping, logical output column zero belongs to
    // thread_in_group zero and uses accumulators d0/d2 for the two A rows.
    if thread_in_group == 0 {
        let weight_row0 = output_start + lane_group;
        let weight_row1 = weight_row0 + 8;
        // SAFETY: All sixteen rows execute the MMA. The shared store helper rejects
        // rows beyond n and maps the sole input row to row-major `[1, n]` outputs.
        unsafe {
            store_scaled_output(
                out,
                unrounded,
                0,
                weight_row0,
                1,
                n_usize,
                accumulators.0,
                global_factor,
            );
            store_scaled_output(
                out,
                unrounded,
                0,
                weight_row1,
                1,
                n_usize,
                accumulators.2,
                global_factor,
            );
        }
    }
}
