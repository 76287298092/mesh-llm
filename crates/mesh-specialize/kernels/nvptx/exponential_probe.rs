//! Raw-u64 control/candidate exponential outputs, including NaN payloads and signed zero.
use core::arch::asm;

#[inline(always)]
fn index() -> usize {
    let block: u32;
    let thread: u32;
    // SAFETY: Coordinates only; no memory or stack effects.
    unsafe {
        asm!("mov.u32 {block}, %ctaid.x;", "mov.u32 {thread}, %tid.x;",
            block = out(reg32) block, thread = out(reg32) thread, options(nomem, nostack));
    }
    block as usize * 256 + thread as usize
}

/// # Safety
/// Launch ceil(count/256) CTAs of 256 threads. Input/output cover count aligned
/// u64 elements, are disjoint, and remain live until completion. All input bits
/// are accepted; this is the unchanged helper's behavior, not an exp domain claim.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exponential_control_bits(
    input: *const u64,
    output: *mut u64,
    count: u32,
) {
    // SAFETY: The entry and shared probe body have identical contracts.
    unsafe {
        probe::<false>(input, output, count);
    }
}

/// # Safety
/// Identical launch, extent, disjointness and lifetime contract to exponential_control_bits.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exponential_unrolled_bits(
    input: *const u64,
    output: *mut u64,
    count: u32,
) {
    // SAFETY: The entry and shared probe body have identical contracts.
    unsafe {
        probe::<true>(input, output, count);
    }
}

#[inline(always)]
unsafe fn probe<const UNROLLED: bool>(input: *const u64, output: *mut u64, count: u32) {
    let index = index();
    if index >= count as usize {
        return;
    }
    // SAFETY: The count check bounds both independent arrays.
    let value = f64::from_bits(unsafe { *input.add(index) });
    let result = if UNROLLED {
        super::exponential_unrolled::exp_nonpositive(value)
    } else {
        super::exponential::exp_nonpositive(value)
    };
    // SAFETY: Each active thread exclusively owns this output word.
    unsafe {
        output.add(index).write(result.to_bits());
    }
}
