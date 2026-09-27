use core::arch::asm;

/// Evaluate every BF16 encoding through the production SiLU arithmetic.
///
/// # Safety
/// Launch 256 blocks of 256 threads. `output` addresses 65,536 writable f32
/// elements and remains live until synchronization. Each thread owns one value.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn silu_bf16_probe(output: *mut f32) {
    let block: u32;
    let thread: u32;
    // SAFETY: Coordinate register reads have no memory or stack effects.
    unsafe {
        asm!("mov.u32 {block}, %ctaid.x;", "mov.u32 {thread}, %tid.x;",
            block=out(reg32) block,thread=out(reg32) thread,options(nomem,nostack));
    }
    let index = block * 256 + thread;
    if index < 65536 {
        let input = f32::from_bits(index << 16);
        // SAFETY: The bounded thread index selects its exclusive output element.
        unsafe {
            output.add(index as usize).write(super::silu::silu(input));
        }
    }
}
