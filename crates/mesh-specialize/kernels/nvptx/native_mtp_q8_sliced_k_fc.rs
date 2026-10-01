// SPDX-License-Identifier: Apache-2.0
// Derived from NInfer contributors' Q8 sliced-K implementation at
// e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d, src/ops/linear/q8/.
// Modification notice: Rust NVPTX port restricted to N5120/K10240, T1/T5;
// separate unregistered C4/C8 entries, explicit PTX conversion and reduction.
#[path = "native_mtp_q8_sliced_k_fc/instructions.rs"]
mod instructions;
#[path = "native_mtp_q8_sliced_k_fc/staging.rs"]
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
fn code_pair(tile: &Tile<'_>, row: u32, column: u32) -> u32 {
    let chunk = column >> 4;
    let offset = row * 512 + (chunk ^ (row & 7)) * 16 + (column & 15);
    ptx::signed_pair(ptx::load_code(tile.shared + offset))
}

#[inline(always)]
fn consume<const SHARED_SCALES: bool>(tile: &Tile<'_>, iteration: u32, acc: [f32; 4]) -> [f32; 4] {
    let lane = tile.thread & 31;
    let warp = tile.thread >> 5;
    let gid = lane >> 2;
    let lid = lane & 3;
    let mut pair = 0;
    if lid < 2 {
        let row = gid + lid * 8;
        pair = if SHARED_SCALES {
            ptx::load_scale(tile.shared + 16384 + row * 32 + warp * 4)
        } else {
            // SAFETY: Complete aligned FP16 scale plane, 320 scales per row.
            unsafe {
                ptx::global_scale(
                    tile.operands
                        .scales
                        .add(((tile.row0 + row) * 320 + iteration * 16 + warp * 2) as usize),
                )
            }
        };
    }
    let top = ptx::shuffle(pair, lane & !3);
    let bottom = ptx::shuffle(pair, (lane & !3) + 1);
    let mut accumulator = acc;
    for group in 0..2 {
        let mut dot = [0.0; 4];
        for ki in 0..2 {
            let ks = group * 2 + ki;
            let col = warp * 64 + ks * 16 + lid * 2;
            let weights = [
                code_pair(tile, gid, col),
                code_pair(tile, gid + 8, col),
                code_pair(tile, gid, col + 8),
                code_pair(tile, gid + 8, col + 8),
            ];
            let br = lane & 7;
            let bk = ks * 16 + ((lane >> 3) & 1) * 8;
            let address =
                tile.shared + 8192 + (warp * 512 + br * 64 + staging::swizzle(br, bk)) * 2;
            dot = ptx::mma(weights, ptx::matrix(address), dot);
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
fn reduce_store(tile: &Tile<'_>, mut acc: [f32; 4]) {
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
        for split in [2, 4, 6] {
            let pair = ptx::partial(tile.shared + (split * 32 + lane) * 16);
            for index in 0..4 {
                acc[index] = ptx::add(acc[index], pair[index]);
            }
        }
        for index in 0..4 {
            let token = (lane & 3) * 2 + (index as u32 & 1);
            let row = tile.row0 + (lane >> 2) + (index as u32 / 2) * 8;
            if token < tile.operands.tokens {
                // SAFETY: Warp zero exclusively owns each live token/row output.
                unsafe {
                    tile.operands
                        .output
                        .add((token * 5120 + row) as usize)
                        .write(ptx::bf16(acc[index]))
                };
            }
        }
    }
}

#[inline(always)]
unsafe fn contract<const SHARED_SCALES: bool>(operands: Operands) {
    let (thread, block) = ptx::coordinates();
    let tile = Tile {
        operands: &operands,
        shared: staging::base::<SHARED_SCALES>(),
        row0: block * 16,
        thread,
    };
    let mut acc = [0.0; 4];
    // SAFETY: Entry contracts guarantee complete, aligned, disjoint planes.
    unsafe { staging::issue::<SHARED_SCALES>(&tile, 0) };
    ptx::wait();
    ptx::barrier();
    for iteration in 0..20 {
        acc = consume::<SHARED_SCALES>(&tile, iteration, acc);
        if iteration + 1 < 20 {
            ptx::barrier();
            // SAFETY: All consumers finished before this single staging slot is reused.
            unsafe { staging::issue::<SHARED_SCALES>(&tile, iteration + 1) };
            ptx::wait();
            ptx::barrier();
        }
    }
    reduce_store(&tile, acc);
}

/// Unregistered C4 FC candidate, static K10240, capacity four, ExactTokens=false.
///
/// # Safety
/// Launch grid [320,1,1], block [256,1,1], tokens=1, on an sm80+ device.
/// Provide disjoint, live, 16-byte-aligned codes[5120*10240] signed-byte bits,
/// scales[5120*320] IEEE FP16 words, input[tokens*10240] BF16 words and writable
/// output[tokens*5120] BF16 words. Input/scales and arithmetic results must be
/// finite. Inactive shared columns deliberately remain uninitialized, as upstream.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_q8_sliced_k_fc_c4(
    codes: *const u8,
    scales: *const u16,
    input: *const u16,
    output: *mut u16,
    tokens: u32,
) {
    // SAFETY: Forward the caller's fixed-shape C4 launch contract.
    unsafe {
        contract::<false>(Operands {
            codes,
            scales,
            input,
            output,
            tokens,
        })
    };
}

/// Unregistered C8 FC candidate, static K10240, capacity eight, ExactTokens=false.
///
/// # Safety
/// Same allocation, alignment and launch contract as C4, except tokens=5.
/// Shared scale staging occupies 16896 bytes; C4 uses 16400 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_q8_sliced_k_fc_c8(
    codes: *const u8,
    scales: *const u16,
    input: *const u16,
    output: *mut u16,
    tokens: u32,
) {
    // SAFETY: Forward the caller's fixed-shape C8 launch contract.
    unsafe {
        contract::<true>(Operands {
            codes,
            scales,
            input,
            output,
            tokens,
        })
    };
}
