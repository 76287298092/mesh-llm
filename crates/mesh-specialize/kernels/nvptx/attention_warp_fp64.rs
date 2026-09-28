//! M=1 exact-order attention: one warp per query head / 64-channel output shard.
//! The four shards duplicate the complete QK dot, never partition the key sequence.
use core::arch::asm;

use super::causal_attention::{
    add_rn, decode_bf16, divide_rn, encode_bf16_rn, fp32_to_fp64_exact, fp64_to_fp32_rn,
    multiply_rn, subtract_rn,
};

#[inline(always)]
fn coordinates() -> (u32, u32, u32) {
    let block: u32;
    let lane: u32;
    let threads: u32;
    // SAFETY: Reads one-dimensional launch coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {block}, %ctaid.x;",
            "mov.u32 {lane}, %tid.x;",
            "mov.u32 {threads}, %ntid.x;",
            block = out(reg32) block, lane = out(reg32) lane,
            threads = out(reg32) threads, options(nomem, nostack),
        );
    }
    (block, lane, threads)
}

/// Original tree strides 128, 64, 32, then shuffle strides 16 through 1.
#[inline(always)]
pub(super) fn local_tree(p: [f64; 8]) -> f64 {
    add_rn(
        add_rn(add_rn(p[0], p[4]), add_rn(p[2], p[6])),
        add_rn(add_rn(p[1], p[5]), add_rn(p[3], p[7])),
    )
}

#[allow(named_asm_labels)]
#[inline(always)]
pub(super) fn warp_dot(local: f64) -> f64 {
    let result: f64;
    // SAFETY: All 32 lanes execute both word shuffles uniformly. The full-warp
    // mask and clamp match the control reduction. Broadcast lane zero's result.
    unsafe {
        asm!(
            "{{",
            ".reg .pred repeat;",
            ".reg .b32 offset, low, high, other_low, other_high;",
            ".reg .f64 sum, partner;",
            "mov.f64 sum, {local};",
            "mov.u32 offset, 16;",
            "WARP_DOT_LOOP:",
            "mov.b64 {{low, high}}, sum;",
            "shfl.sync.down.b32 other_low, low, offset, 0x1f, 0xffffffff;",
            "shfl.sync.down.b32 other_high, high, offset, 0x1f, 0xffffffff;",
            "mov.b64 partner, {{other_low, other_high}};",
            "add.rn.f64 sum, sum, partner;",
            "shr.u32 offset, offset, 1;",
            "setp.ne.u32 repeat, offset, 0;",
            "@repeat bra WARP_DOT_LOOP;",
            "mov.b64 {{low, high}}, sum;",
            "shfl.sync.idx.b32 other_low, low, 0, 0x1f, 0xffffffff;",
            "shfl.sync.idx.b32 other_high, high, 0, 0x1f, 0xffffffff;",
            "mov.b64 {result}, {{other_low, other_high}};",
            "}}",
            local = in(reg64) local, result = out(reg64) result,
            options(nomem, nostack),
        );
    }
    result
}

struct Buffers {
    q: *const u16,
    k: *const u16,
    v: *const u16,
    output: *mut u16,
    raw: *mut f32,
}

/// # Safety
/// Fixed M=1, 24 query heads, 4 KV heads, D=256. Launch exactly grid [96,1,1],
/// block [32,1,1]. This retains causal_attention_bf16's five-pointer/six-u32/
/// FP32-scale ABI. Q and both outputs cover 6144 elements; each cache covers
/// capacity*1024 BF16 elements, with [0,past+1) initialized finite values.
/// Require past < capacity <= 262144 and positive finite scale. The suffix may
/// contain poison and is never read. Pointers are aligned, live until completion,
/// and writable ranges are disjoint from each other and every readable range.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn causal_attention_warp_fp64(
    q: *const u16,
    cache_k: *const u16,
    cache_v: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    scale: f32,
) {
    let (block, lane, threads) = coordinates();
    if rows != 1
        || query_heads != 24
        || kv_heads != 4
        || width != 256
        || past >= capacity
        || capacity > 262_144
        || block >= 96
        || threads != 32
    {
        return;
    }
    let buffers = Buffers {
        q,
        k: cache_k,
        v: cache_v,
        output,
        raw: unrounded,
    };
    // SAFETY: The entry's extent/geometry contract and uniform guards apply.
    unsafe {
        attend(&buffers, block as usize, lane as usize, past, scale);
    }
}

#[inline(always)]
unsafe fn attend(buffers: &Buffers, block: usize, lane: usize, past: u32, scale: f32) {
    let head = block / 4;
    let channel = (block % 4) * 64 + lane;
    let query_start = head * 256;
    let mut query = [0.0_f64; 8];
    for c in 0..8 {
        // SAFETY: The eight stripes cover precisely this head's 256 Q channels.
        query[c] = unsafe { decode_bf16(*buffers.q.add(query_start + lane + 32 * c)) };
    }
    let scale = fp32_to_fp64_exact(scale);
    let mut maximum = f64::NEG_INFINITY;
    let mut normalizer = 0.0_f64;
    let mut accumulator = [0.0_f64; 2];
    for token in 0..=past {
        let cache_row = token as usize * 1024 + (head / 6) * 256;
        let mut products = [0.0_f64; 8];
        for c in 0..8 {
            // SAFETY: All reads are in this head and the initialized causal prefix.
            let key = unsafe { decode_bf16(*buffers.k.add(cache_row + lane + 32 * c)) };
            products[c] = multiply_rn(query[c], key);
        }
        let score = multiply_rn(warp_dot(local_tree(products)), scale);
        let next_maximum = if score > maximum { score } else { maximum };
        let alpha = if normalizer == 0.0 {
            0.0_f64
        } else {
            super::exponential::exp_nonpositive(subtract_rn(maximum, next_maximum))
        };
        let beta = super::exponential::exp_nonpositive(subtract_rn(score, next_maximum));
        normalizer = add_rn(multiply_rn(normalizer, alpha), beta);
        maximum = next_maximum;
        for c in 0..2 {
            // SAFETY: Each lane owns two distinct channels in its 64-channel shard.
            let value = unsafe { decode_bf16(*buffers.v.add(cache_row + channel + 32 * c)) };
            accumulator[c] = add_rn(multiply_rn(accumulator[c], alpha), multiply_rn(beta, value));
        }
    }
    for c in 0..2 {
        let index = query_start + channel + 32 * c;
        let normalized = fp64_to_fp32_rn(divide_rn(accumulator[c], normalizer));
        // SAFETY: The head/shard/lane mapping uniquely covers all output channels.
        unsafe {
            buffers.output.add(index).write(encode_bf16_rn(normalized));
            buffers.raw.add(index).write(normalized);
        }
    }
}
