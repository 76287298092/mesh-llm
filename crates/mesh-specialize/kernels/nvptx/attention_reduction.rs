/// Reduces 256 FP64 partials with the original shared-tree addition order.
///
/// Keep the complete convergent region inside one inline assembly block. LLVM
/// must not thread a later lane-zero branch through the reduction's barriers.
/// A device function call avoids that transformation but grows CUDA stack storage.
///
/// # Safety
/// Every one of the CTA's 256 threads must call this function uniformly with
/// its `%tid.x` in `0..256` and the same CTA-local shared-memory base. `shared`
/// must name an aligned 2,048-byte array of 256 FP64 values, and each thread
/// must own the corresponding slot. All callers must reach both block barriers.
// PTX branch targets use identifiers, scoped by the assembly block braces.
#[allow(named_asm_labels)]
#[inline(always)]
pub(super) fn reduce_dot(shared: u32, thread: u32, partial: f64) -> f64 {
    let result: f64;
    // SAFETY: The caller supplies uniform CTA participation and shared storage.
    // Only the complete first warp enters the shuffle loop. Its predicated store
    // does not branch lane zero away from peers before the final CTA barrier.
    unsafe {
        core::arch::asm!(
            "{{",
            ".reg .pred active, leader, repeat;",
            ".reg .b32 address, offset, low, high, other_low, other_high;",
            ".reg .f64 p0, p1, p2, p3, p4, p5, p6, p7, sum, partner;",
            "mad.lo.u32 address, {thread}, 8, {shared};",
            "st.shared.f64 [address], {partial};",
            "bar.sync 0;",
            "setp.lt.u32 active, {thread}, 32;",
            "@!active bra REDUCE_DONE;",
            "ld.shared.f64 p0, [address];",
            "ld.shared.f64 p1, [address+256];",
            "ld.shared.f64 p2, [address+512];",
            "ld.shared.f64 p3, [address+768];",
            "ld.shared.f64 p4, [address+1024];",
            "ld.shared.f64 p5, [address+1280];",
            "ld.shared.f64 p6, [address+1536];",
            "ld.shared.f64 p7, [address+1792];",
            // Original shared-tree strides 128, 64, then 32.
            "add.rn.f64 p0, p0, p4;",
            "add.rn.f64 p2, p2, p6;",
            "add.rn.f64 p0, p0, p2;",
            "add.rn.f64 p1, p1, p5;",
            "add.rn.f64 p3, p3, p7;",
            "add.rn.f64 p1, p1, p3;",
            "add.rn.f64 sum, p0, p1;",
            "mov.u32 offset, 16;",
            "REDUCE_SHUFFLE:",
            "mov.b64 {{low, high}}, sum;",
            "shfl.sync.down.b32 other_low, low, offset, 0x1f, 0xffffffff;",
            "shfl.sync.down.b32 other_high, high, offset, 0x1f, 0xffffffff;",
            "mov.b64 partner, {{other_low, other_high}};",
            "add.rn.f64 sum, sum, partner;",
            "shr.u32 offset, offset, 1;",
            "setp.ne.u32 repeat, offset, 0;",
            "@repeat bra REDUCE_SHUFFLE;",
            "setp.eq.u32 leader, {thread}, 0;",
            "@leader st.shared.f64 [{shared}], sum;",
            "REDUCE_DONE:",
            "bar.sync 0;",
            "ld.shared.f64 {result}, [{shared}];",
            "}}",
            shared = in(reg32) shared,
            thread = in(reg32) thread,
            partial = in(reg64) partial,
            result = out(reg64) result,
            options(nostack),
        )
    };
    result
}
