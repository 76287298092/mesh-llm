use core::arch::asm;

/// BF16 residual plus branch, with FP32 addition and BF16 nearest-even rounding.
///
/// # Safety
/// Launch ceil(count/256) blocks of 256 threads. Count must be 1..=67108864.
/// Each pointer spans count aligned BF16 values, is disjoint from the others,
/// and remains live through completion. Inputs and rounded outputs must be finite.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn residual_add_bf16(
    residual: *const u16,
    branch: *const u16,
    output: *mut u16,
    count: u32,
) {
    let block: u32;
    let thread: u32;
    // SAFETY: Reads device coordinates without memory effects.
    unsafe {
        asm!("mov.u32 {b}, %ctaid.x;", "mov.u32 {t}, %tid.x;",
            b=out(reg32) block, t=out(reg32) thread, options(nomem,nostack));
    }
    let index = block * 256 + thread;
    if index >= count {
        return;
    }
    let index = index as usize;
    // SAFETY: Guarded unique element lies inside each declared allocation.
    let (left, right) = unsafe { (*residual.add(index), *branch.add(index)) };
    let left = f32::from_bits((left as u32) << 16);
    let right = f32::from_bits((right as u32) << 16);
    let sum: f32;
    // SAFETY: Rounded scalar addition has no memory effects.
    unsafe {
        asm!("add.rn.f32 {sum}, {left}, {right};", sum=out(reg32) sum,
            left=in(reg32) left,right=in(reg32) right, options(nomem,nostack));
    }
    let bits = sum.to_bits();
    let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
    // SAFETY: This thread exclusively owns this in-range output element.
    unsafe {
        output.add(index).write(rounded);
    }
}
