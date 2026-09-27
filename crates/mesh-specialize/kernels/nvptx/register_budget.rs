use core::arch::asm;

/// Exercise a balanced register release/reacquisition in one complete warpgroup.
///
/// # Safety
/// Launch one block of 128 threads. The JIT must reserve at least 64 registers
/// per thread, verified through the driver function attribute before launch.
/// Input/output are disjoint allocations of 128 aligned u32 elements. All four
/// warps must execute both barriers and both register instructions together.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_register_budget(input: *const u32, output: *mut u32) {
    let thread: u32;
    // SAFETY: caller supplies a full warpgroup and a verified initial register
    // budget. Released registers are reacquired before computing output; no
    // values in the released tail are relied upon across the transition.
    unsafe {
        asm!(
            "setmaxnreg.dec.sync.aligned.u32 24;",
            "bar.sync 0;",
            "setmaxnreg.inc.sync.aligned.u32 64;",
            "bar.sync 0;",
            "mov.u32 {}, %tid.x;",
            out(reg32) thread,
            options(nostack),
        );
        output.add(thread as usize).write(*input.add(thread as usize) ^ 0x1393_5090);
    }
}
