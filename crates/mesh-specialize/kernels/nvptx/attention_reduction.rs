use super::causal_attention::{add_rn, block_barrier, load_partial, store_partial};

const WARP_THREADS: u32 = 32;

#[inline(always)]
fn shuffle_down_u32(value: u32, offset: u32) -> u32 {
    let result: u32;
    // SAFETY: All lanes in the first warp execute each full-mask shuffle uniformly.
    unsafe {
        core::arch::asm!(
            "shfl.sync.down.b32 {result}, {value}, {offset}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            value = in(reg32) value,
            offset = in(reg32) offset,
            options(nomem, nostack),
        )
    };
    result
}

/// Reduces 256 FP64 partials with the original shared-tree addition order.
///
/// The first three tree levels are evaluated by warp zero. Its final five
/// levels use full-mask warp shuffles; the result is then published through
/// shared slot zero for every CTA thread to consume.
///
/// # Safety
/// Every one of the CTA's 256 threads must call this function uniformly with
/// its `%tid.x` in `0..256` and the same CTA-local shared-memory base. `shared`
/// must name an aligned 2,048-byte array of 256 FP64 values, and each thread
/// must own the corresponding slot. All callers must reach both block barriers.
#[inline(always)]
pub(super) fn reduce_dot(shared: u32, thread: u32, partial: f64) -> f64 {
    store_partial(shared, thread, partial);
    block_barrier();

    if thread < WARP_THREADS {
        let p0 = load_partial(shared, thread);
        let p1 = load_partial(shared, thread + WARP_THREADS);
        let p2 = load_partial(shared, thread + WARP_THREADS * 2);
        let p3 = load_partial(shared, thread + WARP_THREADS * 3);
        let p4 = load_partial(shared, thread + WARP_THREADS * 4);
        let p5 = load_partial(shared, thread + WARP_THREADS * 5);
        let p6 = load_partial(shared, thread + WARP_THREADS * 6);
        let p7 = load_partial(shared, thread + WARP_THREADS * 7);

        // This grouping reproduces the stride-128, stride-64, and stride-32
        // stages of the prior 256-element shared-memory tree.
        let left = add_rn(add_rn(p0, p4), add_rn(p2, p6));
        let right = add_rn(add_rn(p1, p5), add_rn(p3, p7));
        let mut sum = add_rn(left, right);

        let mut offset = WARP_THREADS / 2;
        while offset > 0 {
            let bits = sum.to_bits();
            let low = shuffle_down_u32(bits as u32, offset);
            let high = shuffle_down_u32((bits >> 32) as u32, offset);
            let partner = f64::from_bits((u64::from(high) << 32) | u64::from(low));
            sum = add_rn(sum, partner);
            offset /= 2;
        }

        if thread == 0 {
            store_partial(shared, 0, sum);
        }
    }

    block_barrier();
    load_partial(shared, 0)
}
