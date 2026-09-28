//! Control CTA schedule and arithmetic, changing only exponential coefficient delivery.

/// # Safety
/// Identical to causal_attention_bf16: grid [rows*query_heads,1,1], block [256,1,1].
/// Q/BF16-output/FP32-output each cover rows*query_heads*width elements; K/V each
/// cover capacity*kv_heads*width BF16 elements. Initialized causal prefix is finite,
/// unused tail may contain poison, scale is positive finite, and all writes are
/// disjoint from other ranges. Require valid positive GQA dimensions, width<=256,
/// past+rows<=capacity<=262144, valid aligned pointers, and lifetime through completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn causal_attention_unrolled_fp64(
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
    // SAFETY: Exactly the shared control body's pointer, extent and launch contract.
    unsafe {
        super::causal_attention::attention_body::<true>(
            q,
            cache_k,
            cache_v,
            output,
            unrounded,
            rows,
            query_heads,
            kv_heads,
            width,
            past,
            capacity,
            scale,
        );
    }
}
