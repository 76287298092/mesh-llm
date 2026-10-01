#[path = "native_mtp_q4_schedule.rs"]
mod schedule;

use core::arch::asm;

// Ported from Ninfer Apache-2.0 Q4 GemvR4W1, pinned at
// e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d; modified for this ABI and safe K tails.

#[inline(always)]
fn coordinates() -> (usize, usize) {
    let thread: u32;
    let block: u32;
    // SAFETY: Special-register reads have no memory effects.
    unsafe {
        asm!(
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {block}, %ctaid.x;",
            thread = out(reg32) thread,
            block = out(reg32) block,
            options(nomem, nostack),
        );
    }
    (thread as usize, block as usize)
}

#[inline(always)]
pub(super) fn half_to_f32(bits: u16) -> f32 {
    let value: f32;
    // SAFETY: Converts the input register from IEEE binary16 to FP32.
    unsafe {
        asm!(
            "cvt.f32.f16 {value}, {bits};",
            value = out(reg32) value,
            bits = in(reg16) bits,
            options(nomem, nostack),
        );
    }
    value
}

#[inline(always)]
fn multiply_rn(left: f32, right: f32) -> f32 {
    let value: f32;
    // SAFETY: Rounded scalar multiplication reads and writes registers only.
    unsafe {
        asm!(
            "mul.rn.f32 {value}, {left}, {right};",
            value = out(reg32) value,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        );
    }
    value
}

#[inline(always)]
pub(super) fn add_rn(left: f32, right: f32) -> f32 {
    let value: f32;
    // SAFETY: Rounded scalar addition reads and writes registers only.
    unsafe {
        asm!(
            "add.rn.f32 {value}, {left}, {right};",
            value = out(reg32) value,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        );
    }
    value
}

#[inline(always)]
pub(super) fn multiply_add_rn(left: f32, right: f32, accumulator: f32) -> f32 {
    let value: f32;
    // SAFETY: Rounded scalar fused multiply-add reads and writes registers only.
    unsafe {
        asm!(
            "fma.rn.f32 {value}, {left}, {right}, {accumulator};",
            value = out(reg32) value,
            left = in(reg32) left,
            right = in(reg32) right,
            accumulator = in(reg32) accumulator,
            options(nomem, nostack),
        );
    }
    value
}

#[inline(always)]
fn round_bf16(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Scalar RNE conversion reads and writes registers only.
    unsafe {
        asm!(
            "cvt.rn.bf16.f32 {bits}, {value};",
            bits = out(reg16) bits,
            value = in(reg32) value,
            options(nomem, nostack),
        );
    }
    bits
}

/// # Safety
/// Launch `[ceil(selected_rows/4), 1, 1]` blocks of `[128, 1, 1]` threads.
/// Inputs must be finite BF16 activations and validated, aligned row-split Q4
/// planes; `source_rows` contains `selected_rows` valid parent indices. Outputs
/// each contain `selected_rows` entries. Allocations are disjoint and live until
/// completion. `padded_k` is a multiple of 128 and covers every code/scale tile.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_q4_head_gemv(
    input_bf16: *const u16,
    packed_object: *const u8,
    source_rows: *const u32,
    output_f32: *mut f32,
    output_bf16: *mut u16,
    selected_rows: u32,
    logical_k: u32,
    padded_k: u32,
    scale_offset: u32,
) {
    let (thread, block) = coordinates();
    let lane = thread % 32;
    let warp = thread / 32;
    let selected_row = block * 4 + warp;
    if selected_row >= selected_rows as usize {
        return;
    }
    // SAFETY: The selected-row bound matches the validated row-map allocation.
    let parent_row = unsafe { *source_rows.add(selected_row) } as usize;
    let padded_k = padded_k as usize;
    let groups_per_row = padded_k / 64;
    // SAFETY: Host validation bounds parent_row and the complete padded planes.
    let code_row = unsafe { packed_object.add(parent_row * (padded_k / 2)) };
    // SAFETY: Host validation proves the scale plane includes all parent rows.
    let scale_row = unsafe {
        packed_object.add(scale_offset as usize + parent_row * groups_per_row * 2)
    };
    let row = schedule::WarpRow {
        activations: input_bf16,
        logical_k: logical_k as usize,
        code_row,
        scale_row,
        groups_per_row: (logical_k as usize).div_ceil(64),
        shared_base: schedule::shared_base() + warp as u32 * 544,
    };
    let total = schedule::warp_reduce_sum(schedule::dot_row(&row, lane));
    if lane == 0 {
        unsafe {
            output_f32.add(selected_row).write(total);
            output_bf16.add(selected_row).write(round_bf16(total));
        }
    }
}
