//! Existing GDN gate arithmetic with full-precision per-head parameter loads.

use crate::gdn_prepare::{block_and_thread, gate_values};

/// Compute BF16 beta and FP32 log-decay/decay, retaining FP32 A_log and dt_bias.
///
/// # Safety
/// Same launch and disjoint allocation contract as `gdn_gates`, except `a_log`
/// and `dt_bias` each hold `heads` aligned readable f32 values, not BF16 words.
/// Launch ceil(rows * heads / 256) blocks of 256 threads. A/B and beta remain
/// BF16 [rows, heads]; g/decay remain f32 [rows, heads]. All arithmetic/extents
/// must fit usize and u32, rows/heads must be nonzero, and pointers must remain
/// live until completion. Finite inputs and A_log in [-80, 80] are required.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_gates_f32_params(
    a: *const u16,
    b: *const u16,
    a_log: *const f32,
    dt_bias: *const f32,
    beta: *mut u16,
    g: *mut f32,
    decay: *mut f32,
    rows: u32,
    heads: u32,
) {
    let (block, thread) = block_and_thread();
    let index = block as usize * 256 + thread as usize;
    if index >= rows as usize * heads as usize {
        return;
    }
    let head = index % heads as usize;
    // SAFETY: Validated time-major input and per-head parameter extents cover these loads.
    let (a_value, b_value, log_value, bias_value) = unsafe {
        (
            f32::from_bits(u32::from(*a.add(index)) << 16),
            f32::from_bits(u32::from(*b.add(index)) << 16),
            *a_log.add(head),
            *dt_bias.add(head),
        )
    };
    let (beta_value, gate, decay_value) = gate_values(a_value, b_value, log_value, bias_value);
    // SAFETY: Every active thread owns one element of each output.
    unsafe {
        beta.add(index).write(beta_value);
        g.add(index).write(gate);
        decay.add(index).write(decay_value);
    }
}
