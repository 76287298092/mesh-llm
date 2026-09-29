use core::arch::asm;

#[inline(always)]
fn coordinates() -> (usize, usize) {
    let lane: u32;
    let row: u32;
    // SAFETY: These PTX special-register reads have no memory effects.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {row}, %ctaid.x;",
            lane = out(reg32) lane,
            row = out(reg32) row,
            options(nomem, nostack),
        );
    }
    (lane as usize, row as usize)
}

#[inline(always)]
fn half_to_f32(bits: u16) -> f32 {
    let value: f32;
    // SAFETY: Converts the scalar register from IEEE binary16 to FP32.
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
fn multiply_add_rn(left: f32, right: f32, accumulator: f32) -> f32 {
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
fn add_rn(left: f32, right: f32) -> f32 {
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
fn shuffle_xor(value: f32, mask: u32) -> f32 {
    let shuffled: f32;
    // SAFETY: The complete 32-lane warp executes this register shuffle uniformly.
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {shuffled}, {value}, {mask}, 0x1f, 0xffffffff;",
            shuffled = out(reg32) shuffled,
            value = in(reg32) value,
            mask = in(reg32) mask,
            options(nomem, nostack),
        );
    }
    shuffled
}

/// One warp computes one selected output row from the original parent Q8 planes.
///
/// # Safety
/// Launch grid `[selected_rows, 1, 1]` and block `[32, 1, 1]`. `input_bf16`
/// contains `logical_k` finite BF16 words; `packed_object` contains complete,
/// validated row-split code and scale planes; `source_rows` contains one valid
/// parent row per selected row; and `output` contains `selected_rows` FP32 values.
/// All allocations are disjoint, correctly aligned, and live through completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn native_mtp_q8_gemv(
    input_bf16: *const u16,
    packed_object: *const u8,
    source_rows: *const u32,
    output: *mut f32,
    selected_rows: u32,
    logical_k: u32,
    padded_k: u32,
    scale_offset: u32,
) {
    let (lane, row) = coordinates();
    if row >= selected_rows as usize {
        return;
    }
    // SAFETY: The launch contract bounds row to selected_rows, whose row-map allocation is exact.
    let parent_row = unsafe { *source_rows.add(row) } as usize;
    let logical_k = logical_k as usize;
    let padded_k = padded_k as usize;
    let groups_per_row = padded_k / 32;
    let logical_groups = logical_k.div_ceil(32);
    let row_code_start = parent_row * padded_k;
    let row_scale_start = scale_offset as usize + parent_row * groups_per_row * 2;
    let mut sum = 0.0_f32;
    let mut group = 0;
    while group < logical_groups {
        let k = group * 32 + lane;
        if k < logical_k {
            // SAFETY: Host validation checked mapped parent rows and complete padded code rows.
            let code = unsafe { *packed_object.cast::<i8>().add(row_code_start + k) };
            let scale_index = row_scale_start + group * 2;
            // SAFETY: Host validation checked the aligned scale plane and every selected row extent.
            let scale_bits = unsafe {
                u16::from_le_bytes([
                    *packed_object.add(scale_index),
                    *packed_object.add(scale_index + 1),
                ])
            };
            let scale = half_to_f32(scale_bits);
            let weight = multiply_rn(scale, f32::from(code));
            // SAFETY: Host validation requires exactly logical_k input BF16 words.
            let activation = unsafe {
                f32::from_bits(u32::from(*input_bf16.add(k)) << 16)
            };
            sum = multiply_add_rn(weight, activation, sum);
        }
        group += 1;
    }
    for mask in [16_u32, 8, 4, 2, 1] {
        sum = add_rn(sum, shuffle_xor(sum, mask));
    }
    if lane == 0 {
        // SAFETY: Exactly lane zero owns this in-range selected-row output slot.
        unsafe { output.add(row).write(sum) };
    }
}
