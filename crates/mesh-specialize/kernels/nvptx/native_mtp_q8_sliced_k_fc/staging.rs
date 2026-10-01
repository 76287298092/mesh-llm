// SPDX-License-Identifier: Apache-2.0
// NInfer contributors, pin e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d.
// Modified: Rust NVPTX fixed-shape single-slot staging port.
use super::{Tile, instructions as ptx};
use core::arch::asm;

#[inline(always)]
pub(super) fn base<const SHARED_SCALES: bool>() -> u32 {
    let base: u32;
    // SAFETY: A CTA-local aligned union of staging and reduction scratch.
    unsafe {
        asm!(
            "{{ .shared .align 16 .b8 fc_tile[{bytes}];",
            "mov.u32 {base}, fc_tile; }}",
            bytes = const if SHARED_SCALES { 16896 } else { 16400 },
            base = out(reg32) base, options(nostack),
        );
    }
    base
}

#[inline(always)]
pub(super) fn swizzle(row: u32, col: u32) -> u32 {
    (((col >> 3) ^ (row & 7)) << 3) | (col & 7)
}

#[inline(always)]
unsafe fn copy<const CA: bool>(destination: u32, source: *const u8) {
    // SAFETY: Internal callers supply valid aligned 16-byte shared/global ranges.
    unsafe {
        if CA {
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
pub(super) unsafe fn issue<const SHARED_SCALES: bool>(tile: &Tile<'_>, iteration: u32) {
    let warp = tile.thread >> 5;
    let lane = tile.thread & 31;
    let k0 = iteration * 512;
    for row_item in 0..2 {
        let row = warp * 2 + row_item;
        let destination = tile.shared + row * 512 + (lane ^ (row & 7)) * 16;
        // SAFETY: Two loader rows per warp and 32 aligned chunks exactly cover 16x512.
        unsafe {
            copy::<false>(
                destination,
                tile.operands
                    .codes
                    .add(((tile.row0 + row) * 10240 + k0 + lane * 16) as usize),
            )
        };
    }
    if SHARED_SCALES && tile.thread < 32 {
        let row = tile.thread / 2;
        let chunk = tile.thread & 1;
        let destination = tile.shared + 16384 + row * 32 + chunk * 16;
        // SAFETY: These 32 copies cover 16 rows of 16 FP16 scales for this K512 tile.
        unsafe {
            copy::<false>(
                destination,
                tile.operands
                    .scales
                    .add(((tile.row0 + row) * 320 + iteration * 16 + chunk * 8) as usize)
                    .cast(),
            )
        };
    }
    let mut item = lane;
    while item < tile.operands.tokens * 8 {
        let col = item / 8;
        let k8 = (item & 7) * 8;
        let destination = tile.shared + 8192 + (warp * 512 + col * 64 + swizzle(col, k8)) * 2;
        // SAFETY: RuntimeActive copies only live columns, each with eight BF16 words.
        unsafe {
            copy::<true>(
                destination,
                tile.operands
                    .input
                    .add((col * 10240 + k0 + warp * 64 + k8) as usize)
                    .cast(),
            )
        };
        item += 32;
    }
    ptx::commit();
}
