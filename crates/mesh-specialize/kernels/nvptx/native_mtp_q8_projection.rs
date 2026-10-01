// SPDX-License-Identifier: Apache-2.0
// Derived from NInfer contributors' Q8 sliced-K implementation at
// e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d, src/ops/linear/q8/.
// Modification notice: Rust NVPTX identity-row resident projections for MTP;
// fixed T1/T5 entry points retain the shape-selected split counts and scale access.
#[path = "native_mtp_q8_projection/instructions.rs"]
mod instructions;
#[path = "native_mtp_q8_projection/staging.rs"]
mod staging;

use instructions as ptx;

struct Operands {
    codes: *const u8,
    scales: *const u16,
    input: *const u16,
    output: *mut u16,
    tokens: u32,
}

struct Tile<'a> {
    operands: &'a Operands,
    shared: u32,
    row0: u32,
    thread: u32,
}

#[inline(always)]
fn code_pair<const K_WARPS: u32>(tile: &Tile<'_>, row: u32, column: u32) -> u32 {
    let block_k = K_WARPS * 64;
    let split_column = (tile.thread >> 5) * 64 + column;
    let chunk = split_column >> 4;
    let offset = row * block_k + (chunk ^ (row & 7)) * 16 + (split_column & 15);
    ptx::load_code(tile.shared + offset)
}

#[inline(always)]
fn consume<const K: u32, const K_WARPS: u32, const SHARED_SCALES: bool>(
    tile: &Tile<'_>,
    iteration: u32,
    acc: [f32; 4],
) -> [f32; 4] {
    let lane = tile.thread & 31;
    let warp = tile.thread >> 5;
    let gid = lane >> 2;
    let lid = lane & 3;
    let block_k = K_WARPS * 64;
    let activation_base = 16 * block_k;
    let scale_base = activation_base + K_WARPS * 8 * 64 * 2;
    let warp_koff = warp * 64;
    let mut pair = 0;
    if lid < 2 {
        let row = gid + lid * 8;
        pair = if SHARED_SCALES {
            ptx::load_scale(tile.shared + scale_base + row * (block_k / 16) + warp * 4)
        } else {
            let scale_index =
                (tile.row0 + row) * (K / 32) + (iteration * block_k + warp_koff) / 32;
            // SAFETY: Static K divisibility and CTA row geometry keep this aligned pair in the
            // full parent FP16 scale plane supplied to the entry point.
            unsafe { ptx::global_scale(tile.operands.scales.add(scale_index as usize)) }
        };
    }
    let top = ptx::shuffle(pair, lane & !3);
    let bottom = ptx::shuffle(pair, (lane & !3) + 1);
    let mut accumulator = acc;
    for group in 0..2 {
        let mut dot = [0.0; 4];
        for ki in 0..2 {
            let ks = group * 2 + ki;
            let code_column = ks * 16 + lid * 2;
            let weights = [
                ptx::signed_pair(code_pair::<K_WARPS>(tile, gid, code_column)),
                ptx::signed_pair(code_pair::<K_WARPS>(tile, gid + 8, code_column)),
                ptx::signed_pair(code_pair::<K_WARPS>(tile, gid, code_column + 8)),
                ptx::signed_pair(code_pair::<K_WARPS>(tile, gid + 8, code_column + 8)),
            ];
            let br = lane & 7;
            let bk = ks * 16 + ((lane >> 3) & 1) * 8;
            let activation_offset =
                activation_base + (warp * 512 + br * 64 + staging::swizzle(br, bk)) * 2;
            dot = ptx::mma(weights, ptx::matrix(tile.shared + activation_offset), dot);
        }
        let top_scale = ptx::half(top >> (group * 16));
        let bottom_scale = ptx::half(bottom >> (group * 16));
        for index in 0..4 {
            let scale = if index < 2 { top_scale } else { bottom_scale };
            accumulator[index] = ptx::fma(dot[index], scale, accumulator[index]);
        }
    }
    accumulator
}

#[inline(always)]
fn reduce_store<const ROWS: u32, const K_WARPS: u32>(tile: &Tile<'_>, mut acc: [f32; 4]) {
    let lane = tile.thread & 31;
    let warp = tile.thread >> 5;
    ptx::barrier();
    ptx::store_odd_partial_barrier(tile.shared + (warp * 32 + lane) * 16, warp, acc);
    if warp & 1 == 0 {
        let partner = ptx::partial(tile.shared + ((warp + 1) * 32 + lane) * 16);
        for index in 0..4 {
            acc[index] = ptx::add(acc[index], partner[index]);
        }
        if warp != 0 {
            ptx::store_partial(tile.shared + (warp * 32 + lane) * 16, acc);
        }
    }
    ptx::barrier();
    if warp == 0 {
        let mut split = 2;
        while split < K_WARPS {
            let pair = ptx::partial(tile.shared + (split * 32 + lane) * 16);
            for index in 0..4 {
                acc[index] = ptx::add(acc[index], pair[index]);
            }
            split += 2;
        }
        for index in 0..4 {
            let token = (lane & 3) * 2 + (index as u32 & 1);
            let row = tile.row0 + (lane >> 2) + (index as u32 / 2) * 8;
            if token < tile.operands.tokens {
                let output_index = token * ROWS + row;
                // SAFETY: Warp zero maps each live token/row pair to one unique output word.
                unsafe {
                    tile.operands
                        .output
                        .add(output_index as usize)
                        .write(ptx::bf16(acc[index]))
                };
            }
        }
    }
}

#[inline(always)]
unsafe fn contract<const ROWS: u32, const K: u32, const K_WARPS: u32, const SHARED_SCALES: bool>(
    operands: Operands,
) {
    let (thread, block) = ptx::coordinates();
    let tile = Tile {
        operands: &operands,
        shared: staging::base::<K_WARPS, SHARED_SCALES>(),
        row0: block * 16,
        thread,
    };
    let mut acc = [0.0; 4];
    let iterations = K / (K_WARPS * 64);
    // SAFETY: The entry contract supplies complete aligned Q8 parent planes and input rows.
    unsafe { staging::issue::<K, K_WARPS, SHARED_SCALES>(&tile, 0) };
    ptx::wait();
    ptx::barrier();
    for iteration in 0..iterations {
        acc = consume::<K, K_WARPS, SHARED_SCALES>(&tile, iteration, acc);
        if iteration + 1 < iterations {
            ptx::barrier();
            // SAFETY: The prior tile is consumed by every warp before the staging slot is reused.
            unsafe { staging::issue::<K, K_WARPS, SHARED_SCALES>(&tile, iteration + 1) };
            ptx::wait();
            ptx::barrier();
        }
    }
    reduce_store::<ROWS, K_WARPS>(&tile, acc);
}

macro_rules! projection_entry {
    ($name:ident, $rows:literal, $k:literal, $warps:literal, $shared:literal) => {
        #[doc = concat!("Resident identity-row Q8 projection entry, rows=", stringify!($rows), ", K=", stringify!($k), ".")]
        ///
        /// # Safety
        /// Launch the fixed K-split CTA grid with complete disjoint aligned Q8 and BF16 planes.
        /// The token count is one or five and all activation values must be finite.
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            codes: *const u8,
            scales: *const u16,
            input: *const u16,
            output: *mut u16,
            tokens: u32,
        ) {
            // SAFETY: Forward the caller's fixed geometry and complete-plane contract.
            unsafe {
                contract::<$rows, $k, $warps, $shared>(Operands {
                    codes,
                    scales,
                    input,
                    output,
                    tokens,
                })
            };
        }
    };
}

projection_entry!(native_mtp_q8_projection_qkv_c4, 14_336, 5_120, 8, false);
projection_entry!(native_mtp_q8_projection_qkv_c8, 14_336, 5_120, 4, true);
projection_entry!(native_mtp_q8_projection_mlp_c4, 34_816, 5_120, 4, false);
projection_entry!(native_mtp_q8_projection_mlp_c8, 34_816, 5_120, 4, true);
projection_entry!(native_mtp_q8_projection_attention_output_c4, 5_120, 6_144, 8, false);
projection_entry!(native_mtp_q8_projection_attention_output_c8, 5_120, 6_144, 8, true);
projection_entry!(native_mtp_q8_projection_mlp_down_c4, 5_120, 17_408, 8, false);
projection_entry!(native_mtp_q8_projection_mlp_down_c8, 5_120, 17_408, 8, true);
