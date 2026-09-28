//! Three ordered M=1 stages. Workspace layout is capacity-strided, not length-strided.
//! No stage reads or writes the unused suffix of any score/coefficient head plane.
use super::{
    attention_warp_fp64::{local_tree, warp_dot},
    causal_attention::{
        add_rn, decode_bf16, divide_rn, encode_bf16_rn, fp32_to_fp64_exact, fp64_to_fp32_rn,
        multiply_rn, subtract_rn,
    },
    exponential_unrolled::exp_nonpositive,
};
use core::arch::asm;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let x: u32;
    let y: u32;
    let thread: u32;
    let threads: u32;
    // SAFETY: Coordinate reads only, with no memory effects.
    unsafe {
        asm!("mov.u32 {x}, %ctaid.x;", "mov.u32 {y}, %ctaid.y;",
        "mov.u32 {thread}, %tid.x;", "mov.u32 {threads}, %ntid.x;",
        x=out(reg32)x,y=out(reg32)y,thread=out(reg32)thread,threads=out(reg32)threads,
        options(nomem,nostack));
    }
    (x, y, thread, threads)
}
#[inline(always)]
fn valid(length: u32, capacity: u32) -> bool {
    length > 0 && length <= capacity && capacity <= 262_144
}

/// # Safety
/// Fixed M1/24Q/4KV/D256. Grid[length,24,1], block[32,1,1]. Q covers6144 BF16,
/// K covers capacity*1024 BF16; initialized prefix [0,length) and Q are finite.
/// Workspace covers (3*24*capacity+24) aligned f64 values. Q/K/workspace are
/// disjoint and live through completion. Scale is positive finite, length=past+1.
/// Only score[head*capacity+key] for key<length is written. This launch must
/// complete before the coefficient stage through ordering on the SAME stream.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_staged_scores_fp64(
    q: *const u16,
    cache_k: *const u16,
    workspace: *mut f64,
    length: u32,
    capacity: u32,
    scale: f32,
) {
    let (key, head, lane, threads) = coordinates();
    if !valid(length, capacity) || key >= length || head >= 24 || threads != 32 {
        return;
    }
    let mut products = [0.0_f64; 8];
    for c in 0..8 {
        let channel = lane as usize + 32 * c;
        // SAFETY: Fixed-width stripes cover one query/key head in the initialized prefix.
        let query = unsafe { decode_bf16(*q.add(head as usize * 256 + channel)) };
        let key_value = unsafe {
            decode_bf16(*cache_k.add(key as usize * 1024 + (head as usize / 6) * 256 + channel))
        };
        products[c] = multiply_rn(query, key_value);
    }
    let dot = warp_dot(local_tree(products));
    if lane == 0 {
        let score = multiply_rn(dot, fp32_to_fp64_exact(scale));
        // SAFETY: Each CTA uniquely owns one score in the capacity-strided prefix.
        unsafe {
            workspace
                .add(head as usize * capacity as usize + key as usize)
                .write(score);
        }
    }
}

/// # Safety
/// Grid[24,1,1], block[32,1,1]. Same live workspace/capacity as scores. Scores for
/// EVERY head's [0,length) have completed on this stream. Only lane0 scans each
/// head strictly ascending. Planes: scores=0, alpha=24*C, beta=48*C, norm=72*C.
/// Every initialized coefficient and all24 normalizers are overwritten, then this
/// entire launch must complete on the SAME stream before values may consume them.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_staged_coefficients_fp64(
    workspace: *mut f64,
    length: u32,
    capacity: u32,
) {
    let (head, y, lane, threads) = coordinates();
    if !valid(length, capacity) || head >= 24 || y != 0 || lane != 0 || threads != 32 {
        return;
    }
    let plane = 24 * capacity as usize;
    let base = head as usize * capacity as usize;
    let mut maximum = f64::NEG_INFINITY;
    let mut normalizer = 0.0_f64;
    for key in 0..length {
        let index = base + key as usize;
        // SAFETY: The preceding same-stream score launch initialized exactly this prefix.
        let score = unsafe { *workspace.add(index) };
        let next_maximum = if score > maximum { score } else { maximum };
        let alpha = if normalizer == 0.0 {
            0.0_f64
        } else {
            exp_nonpositive(subtract_rn(maximum, next_maximum))
        };
        let beta = exp_nonpositive(subtract_rn(score, next_maximum));
        normalizer = add_rn(multiply_rn(normalizer, alpha), beta);
        maximum = next_maximum;
        // SAFETY: Coefficient planes are disjoint from score and each other; each
        // head has exactly one writer. The unused suffix is deliberately untouched.
        unsafe {
            workspace.add(plane + index).write(alpha);
            workspace.add(2 * plane + index).write(beta);
        }
    }
    // SAFETY: Normalizers are the final24 separate workspace elements, one per head.
    unsafe {
        workspace.add(3 * plane + head as usize).write(normalizer);
    }
}

/// # Safety
/// Grid[24,4,1], block[64,1,1]. V covers capacity*1024 finite-prefix BF16 values;
/// workspace is the completed same-stream score/coefficient result for THIS head
/// and length, not a previous layer/longer invocation. Outputs cover6144 BF16 and
/// FP32 elements. All four allocations are aligned, disjoint, and live through
/// completion. Each thread owns one output channel and visits only keys<length,
/// in ascending order. Normalizer is read only after all coefficient CTAs finish.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_staged_values_fp64(
    cache_v: *const u16,
    workspace: *const f64,
    output: *mut u16,
    raw: *mut f32,
    length: u32,
    capacity: u32,
) {
    let (head, shard, thread, threads) = coordinates();
    if !valid(length, capacity) || head >= 24 || shard >= 4 || threads != 64 {
        return;
    }
    let channel = shard as usize * 64 + thread as usize;
    let plane = 24 * capacity as usize;
    let base = head as usize * capacity as usize;
    let mut accumulator = 0.0_f64;
    for key in 0..length {
        let index = base + key as usize;
        // SAFETY: Same-stream predecessor initialized every coefficient used here.
        let (alpha, beta, value) = unsafe {
            (
                *workspace.add(plane + index),
                *workspace.add(2 * plane + index),
                decode_bf16(
                    *cache_v.add(key as usize * 1024 + (head as usize / 6) * 256 + channel),
                ),
            )
        };
        accumulator = add_rn(multiply_rn(accumulator, alpha), multiply_rn(beta, value));
    }
    // SAFETY: The preceding scan initialized this head's final normalizer.
    let normalizer = unsafe { *workspace.add(3 * plane + head as usize) };
    let normalized = fp64_to_fp32_rn(divide_rn(accumulator, normalizer));
    let index = head as usize * 256 + channel;
    // SAFETY: Each head/shard/thread exclusively owns one in-range output pair.
    unsafe {
        output.add(index).write(encode_bf16_rn(normalized));
        raw.add(index).write(normalized);
    }
}
