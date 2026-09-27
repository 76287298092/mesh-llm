use core::arch::asm;

#[inline(always)]
fn lane_id() -> usize {
    let lane: u32;
    // SAFETY: Reads the calling thread's lane identifier and has no memory effects.
    unsafe { asm!("mov.u32 {}, %laneid;", out(reg32) lane, options(nomem, nostack)) };
    lane as usize
}

/// Probe the ordinary BF16 16x8x16 warp MMA instruction.
///
/// # Safety
/// Launch exactly one 32-thread block. `a`, `b`, and `output` must be disjoint,
/// 4-byte-aligned device pointers to 128 readable, 64 readable, and 128 writable
/// `u32` words, respectively, and remain live until the kernel completes. Lane `i`
/// reads `a[4*i..4*i+4]` and `b[2*i..2*i+2]`; it writes four output words.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_bf16_mma(a: *const u32, b: *const u32, output: *mut u32) {
    let lane = lane_id();
    // SAFETY: The caller provides the full aligned fragments for all 32 lanes.
    let (a0, a1, a2, a3, b0, b1) = unsafe {
        (
            *a.add(lane * 4),
            *a.add(lane * 4 + 1),
            *a.add(lane * 4 + 2),
            *a.add(lane * 4 + 3),
            *b.add(lane * 2),
            *b.add(lane * 2 + 1),
        )
    };
    let (mut d0, mut d1, mut d2, mut d3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    // SAFETY: All lanes execute the same warp MMA with valid BF16 register fragments.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
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
            options(nomem, nostack),
        );
    }
    // SAFETY: Each lane owns four distinct output words within the 128-word allocation.
    unsafe {
        output.add(lane * 4).write(d0.to_bits());
        output.add(lane * 4 + 1).write(d1.to_bits());
        output.add(lane * 4 + 2).write(d2.to_bits());
        output.add(lane * 4 + 3).write(d3.to_bits());
    }
}

/// Probe the ordinary FP16 16x8x16 warp MMA instruction.
///
/// # Safety
/// Launch exactly one 32-thread block. `a`, `b`, and `output` must be disjoint,
/// 4-byte-aligned device pointers to 128 readable, 64 readable, and 128 writable
/// `u32` words, respectively, and remain live until the kernel completes. Lane `i`
/// reads `a[4*i..4*i+4]` and `b[2*i..2*i+2]`; it writes four output words.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_fp16_mma(a: *const u32, b: *const u32, output: *mut u32) {
    let lane = lane_id();
    // SAFETY: The caller provides the full aligned fragments for all 32 lanes.
    let (a0, a1, a2, a3, b0, b1) = unsafe {
        (
            *a.add(lane * 4),
            *a.add(lane * 4 + 1),
            *a.add(lane * 4 + 2),
            *a.add(lane * 4 + 3),
            *b.add(lane * 2),
            *b.add(lane * 2 + 1),
        )
    };
    let (mut d0, mut d1, mut d2, mut d3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    // SAFETY: All lanes execute the same warp MMA with valid FP16 register fragments.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
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
            options(nomem, nostack),
        );
    }
    // SAFETY: Each lane owns four distinct output words within the 128-word allocation.
    unsafe {
        output.add(lane * 4).write(d0.to_bits());
        output.add(lane * 4 + 1).write(d1.to_bits());
        output.add(lane * 4 + 2).write(d2.to_bits());
        output.add(lane * 4 + 3).write(d3.to_bits());
    }
}

/// Probe the ordinary signed INT8 16x8x32 warp MMA instruction.
///
/// # Safety
/// Launch exactly one 32-thread block. `a`, `b`, and `output` must be disjoint,
/// 4-byte-aligned device pointers to 128 readable, 64 readable, and 128 writable
/// `u32` words, respectively, and remain live until the kernel completes. Lane `i`
/// reads `a[4*i..4*i+4]` and `b[2*i..2*i+2]`; it writes four output words.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_int8_mma(a: *const u32, b: *const u32, output: *mut u32) {
    let lane = lane_id();
    // SAFETY: The caller provides the full aligned fragments for all 32 lanes.
    let (a0, a1, a2, a3, b0, b1) = unsafe {
        (
            *a.add(lane * 4),
            *a.add(lane * 4 + 1),
            *a.add(lane * 4 + 2),
            *a.add(lane * 4 + 3),
            *b.add(lane * 2),
            *b.add(lane * 2 + 1),
        )
    };
    let (mut d0, mut d1, mut d2, mut d3) = (0_i32, 0_i32, 0_i32, 0_i32);
    // SAFETY: All lanes execute the same warp MMA with valid signed INT8 fragments.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
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
            options(nomem, nostack),
        );
    }
    // SAFETY: Each lane owns four distinct output words within the 128-word allocation.
    unsafe {
        output.add(lane * 4).write(d0 as u32);
        output.add(lane * 4 + 1).write(d1 as u32);
        output.add(lane * 4 + 2).write(d2 as u32);
        output.add(lane * 4 + 3).write(d3 as u32);
    }
}
