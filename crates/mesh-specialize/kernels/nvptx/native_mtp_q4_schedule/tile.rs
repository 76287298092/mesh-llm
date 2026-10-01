#[path = "tile/decode.rs"]
mod decode;

use super::WarpRow;
use core::arch::asm;

pub(super) struct Tile<'a> {
    pub(super) row: &'a WarpRow,
    pub(super) group_begin: usize,
    pub(super) active_groups: usize,
}

#[inline(always)]
fn store_zero_vector(address: u32) {
    let zero = 0_u32;
    // SAFETY: The caller assigns each aligned 16-byte shared destination once.
    unsafe {
        asm!(
            "st.shared.v4.u32 [{address}], {{{zero}, {zero}, {zero}, {zero}}};",
            address = in(reg32) address,
            zero = in(reg32) zero,
            options(nostack),
        );
    }
}

#[inline(always)]
pub(super) fn issue(tile: &Tile<'_>, lane: usize) {
    if lane < tile.active_groups * 2 {
        // SAFETY: Tile coordinates are within validated padded K.
        let source = unsafe { tile.row.code_row.add(tile.group_begin * 32 + lane * 16) };
        // SAFETY: Validated row extents provide aligned complete code vectors.
        unsafe {
            asm!(
                "cvta.to.global.u64 {global}, {source};",
                "cp.async.ca.shared.global [{destination}], [{global}], 16;",
                global = out(reg64) _,
                source = in(reg64) source as u64,
                destination = in(reg32) tile.row.shared_base + lane as u32 * 16,
                options(nostack),
            );
        }
    } else {
        store_zero_vector(tile.row.shared_base + lane as u32 * 16);
    }
    if lane < tile.active_groups.div_ceil(2) {
        // SAFETY: Tile coordinates select a complete validated scale pair.
        let source = unsafe { tile.row.scale_row.add(tile.group_begin * 2 + lane * 4) };
        // SAFETY: Validated padded K covers the paired scale transfer.
        unsafe {
            asm!(
                "cvta.to.global.u64 {global}, {source};",
                "cp.async.ca.shared.global [{destination}], [{global}], 4;",
                global = out(reg64) _,
                source = in(reg64) source as u64,
                destination = in(reg32) tile.row.shared_base + 512 + lane as u32 * 4,
                options(nostack),
            );
        }
    } else if lane < 8 {
        let zero = 0_u32;
        // SAFETY: This lane owns the aligned inactive word in its warp's scale tile.
        unsafe {
            asm!(
                "st.shared.u32 [{destination}], {zero};",
                destination = in(reg32) tile.row.shared_base + 512 + lane as u32 * 4,
                zero = in(reg32) zero,
                options(nostack),
            );
        }
    }
    // SAFETY: Each lane commits its own async-copy group before waiting.
    unsafe { asm!("cp.async.commit_group;", options(nostack)) };
}

#[inline(always)]
fn shared_word(address: u32) -> u32 {
    let value: u32;
    // SAFETY: The consumed word is aligned and its tile copy has completed.
    unsafe {
        asm!(
            "ld.shared.b32 {value}, [{address}];",
            value = out(reg32) value,
            address = in(reg32) address,
            options(nostack),
        );
    }
    value
}

#[inline(always)]
fn activation_words(row: &WarpRow, k_begin: usize) -> [u32; 4] {
    if k_begin + 8 <= row.logical_k {
        // SAFETY: The following vector load is within the complete K range.
        let pointer = unsafe { row.activations.add(k_begin) };
        let (first, second, third, fourth): (u32, u32, u32, u32);
        // SAFETY: K bounds the full aligned 16-byte vector read.
        unsafe {
            asm!(
                "ld.global.v4.u32 {{{first}, {second}, {third}, {fourth}}}, [{pointer}];",
                first = out(reg32) first,
                second = out(reg32) second,
                third = out(reg32) third,
                fourth = out(reg32) fourth,
                pointer = in(reg64) pointer as u64,
                options(readonly, nostack),
            );
        }
        [first, second, third, fourth]
    } else {
        let mut words = [0_u32; 4];
        for offset in 0..8 {
            let k = k_begin + offset;
            if k < row.logical_k {
                // SAFETY: The element index is bounded by the activation length.
                let activation = unsafe { *row.activations.add(k) };
                words[offset / 2] |= u32::from(activation) << ((offset % 2) * 16);
            }
        }
        words
    }
}

#[inline(always)]
pub(super) fn consume(tile: &Tile<'_>, lane: usize, mut accumulator: f32) -> f32 {
    let lane_group = lane >> 3;
    let lane_in_group = lane & 7;
    let mut group_base = 0;
    while group_base < 16 {
        let local_group = group_base + lane_group;
        if local_group < tile.active_groups {
            let packed = shared_word(
                tile.row.shared_base + (group_base * 8 + lane) as u32 * 4,
            );
            let scale_pair = shared_word(
                tile.row.shared_base + 512 + (local_group >> 1) as u32 * 4,
            );
            let scale_bits = (scale_pair >> ((local_group & 1) * 16)) as u16;
            let weights = decode::decode_eight(packed, scale_bits);
            let k_begin = (tile.group_begin + local_group) * 64 + lane_in_group * 8;
            let packed_activations = activation_words(tile.row, k_begin);
            let activations = [
                f32::from_bits(packed_activations[0] << 16),
                f32::from_bits(packed_activations[0] & 0xffff_0000),
                f32::from_bits(packed_activations[1] << 16),
                f32::from_bits(packed_activations[1] & 0xffff_0000),
                f32::from_bits(packed_activations[2] << 16),
                f32::from_bits(packed_activations[2] & 0xffff_0000),
                f32::from_bits(packed_activations[3] << 16),
                f32::from_bits(packed_activations[3] & 0xffff_0000),
            ];
            for index in 0..8 {
                accumulator = super::super::multiply_add_rn(
                    weights[index],
                    activations[index],
                    accumulator,
                );
            }
        }
        group_base += 4;
    }
    accumulator
}
