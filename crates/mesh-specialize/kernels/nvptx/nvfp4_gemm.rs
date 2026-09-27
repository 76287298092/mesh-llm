use core::arch::asm;

/// Computes one tiled NVFP4 GEMM output tile per block.
///
/// # Safety
/// Launch exactly one 32-thread block per grid tile, with grid.x equal to
/// n_tiles and grid.y equal to m_tiles. All tile counts must be nonzero, and
/// all index arithmetic must fit usize. A uses [m_tile][k_tile][lane][word4]
/// and contains m_tiles*k_tiles*128 readable u32 words. B uses
/// [n_tile][k_tile][lane][word2] and contains n_tiles*k_tiles*64 words. SA
/// contains m_tiles*k_tiles*32 words and SB contains n_tiles*k_tiles*32 words,
/// each in tile/lane order. out contains m_tiles*n_tiles*128 writable f32
/// values in [m_tile][n_tile][lane][four outputs] order. All device pointers
/// must be aligned for their element types, disjoint, and live through completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_gemm_packed(
    a: *const u32,
    b: *const u32,
    sa: *const u32,
    sb: *const u32,
    out: *mut f32,
    m_tiles: u32,
    n_tiles: u32,
    k_tiles: u32,
) {
    let lane: u32;
    let tile_n: u32;
    let tile_m: u32;
    // SAFETY: these special registers are available to every thread and this
    // assembly does not access memory.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {tile_m}, %ctaid.y;",
            lane = out(reg32) lane,
            tile_n = out(reg32) tile_n,
            tile_m = out(reg32) tile_m,
            options(nomem, nostack),
        );
    }
    if tile_m >= m_tiles {
        return;
    }

    let lane = lane as usize;
    let tile_m = tile_m as usize;
    let tile_n = tile_n as usize;
    let n_tiles = n_tiles as usize;
    let k_tiles = k_tiles as usize;
    let (mut d0, mut d1, mut d2, mut d3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

    for k_tile in 0..k_tiles {
        let a_tile = (tile_m * k_tiles + k_tile) * 128;
        let b_tile = (tile_n * k_tiles + k_tile) * 64;
        let scale_a_tile = (tile_m * k_tiles + k_tile) * 32;
        let scale_b_tile = (tile_n * k_tiles + k_tile) * 32;
        // SAFETY: the launch contract provides each packed tile and the lane's
        // four A words, two B words, and one scale word in those allocations.
        let (a0, a1, a2, a3, b0, b1, scale_a, scale_b) = unsafe {
            let a_lane = a.add(a_tile + lane * 4);
            let b_lane = b.add(b_tile + lane * 2);
            (
                a_lane.read(),
                a_lane.add(1).read(),
                a_lane.add(2).read(),
                a_lane.add(3).read(),
                b_lane.read(),
                b_lane.add(1).read(),
                sa.add(scale_a_tile + lane).read(),
                sb.add(scale_b_tile + lane).read(),
            )
        };
        // SAFETY: every warp executes the verified SM120a block-scaled MMA.
        // Accumulators stay live across k tiles; both byte selectors are zero.
        unsafe {
            asm!(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 ",
                "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, ",
                "{{{d0}, {d1}, {d2}, {d3}}}, {scale_a}, {{0, 0}}, {scale_b}, {{0, 0}};",
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
    }

    let output_tile = (tile_m * n_tiles + tile_n) * 128 + lane * 4;
    // SAFETY: each lane writes four distinct outputs in its assigned tile.
    unsafe {
        out.add(output_tile).write(d0);
        out.add(output_tile + 1).write(d1);
        out.add(output_tile + 2).write(d2);
        out.add(output_tile + 3).write(d3);
    }
}
