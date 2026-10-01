// SPDX-License-Identifier: Apache-2.0
// NInfer contributors, pin e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d.
// Modified: Rust NVPTX staging for full-parent identity-row MTP projections.
use super::{Tile, instructions as ptx};
use core::arch::asm;

#[inline(always)]
pub(super) fn base<const K_WARPS: u32, const SHARED_SCALES: bool>() -> u32 {
    let shared: u32;
    // SAFETY: The static CTA allocation is 16-byte aligned and covers the larger staging/reduction union.
    unsafe {
        asm!("{{ .shared .align 16 .b8 q8_mtp_projection[{bytes}];",
            "mov.u32 {shared}, q8_mtp_projection; }}",
            bytes = const if 16 * K_WARPS * 64 + K_WARPS * 8 * 64 * 2
                + if SHARED_SCALES { 16 * (K_WARPS * 64 / 16) } else { 16 }
                > K_WARPS * 32 * 16
            {
                16 * K_WARPS * 64 + K_WARPS * 8 * 64 * 2
                    + if SHARED_SCALES { 16 * (K_WARPS * 64 / 16) } else { 16 }
            } else {
                K_WARPS * 32 * 16
            },
            shared = out(reg32) shared, options(nostack))
    };
    shared
}

#[inline(always)]
pub(super) fn swizzle(row: u32, col: u32) -> u32 {
    (((col >> 3) ^ (row & 7)) << 3) | (col & 7)
}

#[inline(always)]
unsafe fn copy<const CACHE_GLOBAL: bool>(destination: u32, source: *const u8) {
    // SAFETY: Callers prove aligned, readable 16-byte global and writable shared ranges.
    unsafe {
        if CACHE_GLOBAL {
            asm!("{{ .reg .b64 global; cvta.to.global.u64 global, {source};",
                "cp.async.ca.shared.global [{destination}], [global], 16; }}",
                destination = in(reg32) destination, source = in(reg64) source,
                options(nostack));
        } else {
            asm!("{{ .reg .b64 global; cvta.to.global.u64 global, {source};",
                "cp.async.cg.shared.global [{destination}], [global], 16; }}",
                destination = in(reg32) destination, source = in(reg64) source,
                options(nostack));
        }
    }
}

#[inline(always)]
unsafe fn zero_fill(destination: u32, source: *const u8) {
    // SAFETY: The dummy source is valid and aligned; src-size zero initializes the whole shared copy.
    unsafe {
        asm!("{{ .reg .b64 global; cvta.to.global.u64 global, {source};",
            "cp.async.ca.shared.global [{destination}], [global], 16, 0; }}",
            destination = in(reg32) destination, source = in(reg64) source,
            options(nostack));
    }
}

#[inline(always)]
pub(super) unsafe fn issue<const K: u32, const K_WARPS: u32, const SHARED_SCALES: bool>(
    tile: &Tile<'_>,
    iteration: u32,
) {
    let warp = tile.thread >> 5;
    let lane = tile.thread & 31;
    let block_k = K_WARPS * 64;
    let k0 = iteration * block_k;
    let code_bytes = 16 * block_k;
    let activation_bytes = K_WARPS * 8 * 64 * 2;
    let scale_base = code_bytes + activation_bytes;
    let rows_per_loader_warp = 16 / K_WARPS;
    for row_item in 0..rows_per_loader_warp {
        let row = warp * rows_per_loader_warp + row_item;
        let mut chunk = lane;
        while chunk < block_k / 16 {
            let destination = tile.shared + row * block_k + (chunk ^ (row & 7)) * 16;
            let parent_row = tile.row0 + row;
            let source_index = parent_row * K + k0 + chunk * 16;
            // SAFETY: Rows and K are bounded by the full physical parent projection contract.
            unsafe {
                copy::<false>(
                    destination,
                    tile.operands.codes.add(source_index as usize),
                )
            };
            chunk += 32;
        }
    }
    if SHARED_SCALES {
        let scale_bytes_per_row = block_k / 16;
        let scale_chunks = scale_bytes_per_row / 16;
        let mut item = tile.thread;
        while item < 16 * scale_chunks {
            let row = item / scale_chunks;
            let chunk = item % scale_chunks;
            let destination = tile.shared + scale_base + row * scale_bytes_per_row + chunk * 16;
            let parent_row = tile.row0 + row;
            let scale_index = parent_row * (K / 32) + k0 / 32 + chunk * 8;
            // SAFETY: Each copy reads eight adjacent FP16 scales from one complete parent row.
            unsafe {
                copy::<false>(destination, tile.operands.scales.add(scale_index as usize).cast())
            };
            item += K_WARPS * 32;
        }
    }
    let activation_base = code_bytes;
    let mut item = lane;
    while item < 8 * 8 {
        let col = item / 8;
        let k8 = (item & 7) * 8;
        let destination = tile.shared
            + activation_base
            + (warp * 512 + col * 64 + swizzle(col, k8)) * 2;
        if col < tile.operands.tokens {
            let source_index =
                col * K + k0 + warp * 64 + k8;
            // SAFETY: The token, K tile and eight BF16 values fit the live input allocation.
            unsafe {
                copy::<true>(
                    destination,
                    tile.operands.input.add(source_index as usize).cast(),
                )
            };
        } else {
            // SAFETY: The valid first-token input word is only an aligned dummy source for zero-fill.
            unsafe { zero_fill(destination, tile.operands.input.cast()) };
        }
        item += 32;
    }
    ptx::commit();
}
