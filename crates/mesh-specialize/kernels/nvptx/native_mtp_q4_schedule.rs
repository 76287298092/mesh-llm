#[path = "native_mtp_q4_schedule/tile.rs"]
mod tile;

use core::arch::asm;

pub(super) struct WarpRow {
    pub(super) activations: *const u16,
    pub(super) logical_k: usize,
    pub(super) code_row: *const u8,
    pub(super) scale_row: *const u8,
    pub(super) groups_per_row: usize,
    pub(super) shared_base: u32,
}

pub(super) fn shared_base() -> u32 {
    let base: u32;
    // SAFETY: Static 2176-byte allocation is a valid warp-local scratch region.
    unsafe {
        asm!(
            ".shared .align 16 .b8 native_mtp_q4_tiles[2176];",
            "mov.u32 {base}, native_mtp_q4_tiles;",
            base = out(reg32) base,
            options(nostack),
        );
    }
    base
}

#[inline(always)]
fn warp_sync() {
    // SAFETY: All 32 lanes in each participating warp execute this uniformly.
    unsafe { asm!("bar.warp.sync 0xffffffff;", options(nostack)) };
}

#[inline(always)]
pub(super) fn dot_row(row: &WarpRow, lane: usize) -> f32 {
    let mut accumulator = 0.0_f32;
    let mut group_begin = 0;
    while group_begin < row.groups_per_row {
        let active_groups = (row.groups_per_row - group_begin).min(16);
        let tile = tile::Tile { row, group_begin, active_groups };
        tile::issue(&tile, lane);
        // SAFETY: Each warp waits for its own committed code/scale copies.
        unsafe { asm!("cp.async.wait_group 0;", options(nostack)) };
        warp_sync();
        accumulator = tile::consume(&tile, lane, accumulator);
        warp_sync();
        group_begin += 16;
    }
    accumulator
}

#[inline(always)]
pub(super) fn warp_reduce_sum(mut value: f32) -> f32 {
    for offset in [16, 8, 4, 2, 1] {
        let partner: f32;
        // SAFETY: Every lane in the full warp executes the same shuffle instruction.
        unsafe {
            asm!(
                "shfl.sync.down.b32 {partner}, {value}, {offset}, 0x1f, 0xffffffff;",
                partner = out(reg32) partner,
                value = in(reg32) value,
                offset = in(reg32) offset as u32,
                options(nomem, nostack),
            );
        }
        value = super::add_rn(value, partner);
    }
    value
}
