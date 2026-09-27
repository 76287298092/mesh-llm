use core::arch::asm;

/// Copies a warp-sized input tile through shared memory and loads its matrix fragment.
///
/// # Safety
/// Launch exactly one block of 32 threads. input must point to 128 readable
/// u32 values and output to 128 writable u32 values; both device pointers
/// must be 16-byte aligned, disjoint, and remain valid through kernel completion.
/// mode must be identical in every lane and in 0..=7, so every lane executes
/// the same ldmatrix.sync.aligned instruction.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_shared_load(
    input: *const u32,
    output: *mut u32,
    mode: u32,
) {
    let lane: u32;
    let shared_base: u32;
    // SAFETY: the launch contract provides one 32-thread warp, a 16-byte-aligned
    // input tile, and enough shared memory for every lane's disjoint 16-byte copy.
    unsafe {
        asm!(
            ".shared .align 16 .b8 tile[512];",
            ".reg .b64 %memory_global_base;",
            ".reg .b64 %memory_byte_offset;",
            ".reg .b64 %memory_global_source;",
            ".reg .b32 %memory_shared_offset;",
            ".reg .b32 %memory_shared_destination;",
            ".reg .b32 %memory_cache_bit;",
            ".reg .pred %memory_use_ca;",
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {shared_base}, tile;",
            "cvta.to.global.u64 %memory_global_base, {input};",
            "mul.wide.u32 %memory_byte_offset, {lane}, 16;",
            "add.u64 %memory_global_source, %memory_global_base, %memory_byte_offset;",
            "mul.lo.u32 %memory_shared_offset, {lane}, 16;",
            "add.u32 %memory_shared_destination, {shared_base}, %memory_shared_offset;",
            "and.b32 %memory_cache_bit, {mode}, 1;",
            "setp.eq.u32 %memory_use_ca, %memory_cache_bit, 0;",
            "@%memory_use_ca cp.async.ca.shared.global [%memory_shared_destination], [%memory_global_source], 16;",
            "@!%memory_use_ca cp.async.cg.shared.global [%memory_shared_destination], [%memory_global_source], 16;",
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            "bar.sync 0;",
            lane = out(reg32) lane,
            shared_base = out(reg32) shared_base,
            input = in(reg64) input as u64,
            mode = in(reg32) mode,
            options(nostack),
        );
    }

    let shared_address = shared_base + lane * 16;
    let transposed = mode & 0b100 != 0;
    let (d0, d1, d2, d3) = if mode & 0b010 == 0 {
        load_x2(shared_address, transposed)
    } else {
        load_x4(shared_address, transposed)
    };

    // SAFETY: the launch contract supplies 128 writable, 16-byte-aligned output
    // words disjoint from input; each lane owns its four-word output segment.
    unsafe {
        let lane_output = output.add(lane as usize * 4);
        lane_output.write(d0);
        lane_output.add(1).write(d1);
        lane_output.add(2).write(d2);
        lane_output.add(3).write(d3);
    }
}

#[inline(always)]
fn load_x2(shared_address: u32, transposed: bool) -> (u32, u32, u32, u32) {
    let d0: u32;
    let d1: u32;
    // SAFETY: callers pass this lane's valid address in the shared tile, and the
    // kernel contract keeps all 32 lanes on the same x2/transposition variant.
    unsafe {
        if transposed {
            asm!(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{{d0}, {d1}}}, [{address}];",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                address = in(reg32) shared_address,
                options(nostack),
            );
        } else {
            asm!(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{d0}, {d1}}}, [{address}];",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                address = in(reg32) shared_address,
                options(nostack),
            );
        }
    }
    (d0, d1, 0, 0)
}

#[inline(always)]
fn load_x4(shared_address: u32, transposed: bool) -> (u32, u32, u32, u32) {
    let d0: u32;
    let d1: u32;
    let d2: u32;
    let d3: u32;
    // SAFETY: callers pass this lane's valid address in the shared tile, and the
    // kernel contract keeps all 32 lanes on the same x4/transposition variant.
    unsafe {
        if transposed {
            asm!(
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{d0}, {d1}, {d2}, {d3}}}, [{address}];",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                address = in(reg32) shared_address,
                options(nostack),
            );
        } else {
            asm!(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{d0}, {d1}, {d2}, {d3}}}, [{address}];",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                address = in(reg32) shared_address,
                options(nostack),
            );
        }
    }
    (d0, d1, d2, d3)
}
