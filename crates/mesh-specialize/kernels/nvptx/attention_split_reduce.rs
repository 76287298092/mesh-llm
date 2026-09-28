//! Stable FP32 merge of split-attention partials, independent of current past.
use super::attention_split_math as math;

/// Merge all workspace slots, ignoring neutral/empty records (sum==0).
///
/// # Safety
/// Grid [rows*24,1,1], block [256,1,1], dynamic shared 0. Rows=1..8, slots=1..85.
/// Workspace is FP32 [rows,24,slots,258], fully written by the partial kernel on
/// the same stream. Each record is [max,sum,acc[256]]; empty is [-inf,0,0...].
/// At least one split per query has a positive sum. Outputs are disjoint BF16
/// and FP32 [rows,24,256]; all allocations are aligned and live to completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_split_reduce_bf16(
    partial: *const f32,
    output: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    split_slots: u32,
) {
    if !(1..=8).contains(&rows) || !(1..=85).contains(&split_slots) {
        return;
    }
    let (channel, query, _) = math::coordinates();
    if query >= rows * 24 || channel >= 256 {
        return;
    }
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0.0;
    let mut accumulator = 0.0;
    for split in 0..split_slots {
        let offset = ((query * split_slots + split) * 258) as usize;
        // SAFETY: Producer owns and initialized every slot; index bounded by ABI.
        let denominator = unsafe { partial.add(offset + 1).read() };
        if denominator != 0.0 {
            // SAFETY: This is a nonempty, finite producer record.
            let (m, a) = unsafe {
                (
                    partial.add(offset).read(),
                    partial.add(offset + 2 + channel as usize).read(),
                )
            };
            let next = maximum.max(m);
            let alpha = if sum == 0.0 {
                0.0
            } else {
                math::exp(maximum - next)
            };
            let beta = math::exp(m - next);
            accumulator = accumulator * alpha + a * beta;
            sum = sum * alpha + denominator * beta;
            maximum = next;
        }
    }
    let value = math::divide(accumulator, sum);
    let index = (query * 256 + channel) as usize;
    // SAFETY: Exactly one thread owns each output channel.
    unsafe {
        output.add(index).write(math::round_bf16(value));
        unrounded.add(index).write(value);
    }
}
