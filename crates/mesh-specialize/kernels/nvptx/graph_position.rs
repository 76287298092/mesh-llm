//! Position-only adapters for whole-row graph replay. Arithmetic lives in the
//! original exact bodies; these entries only load a persistent u32 and offset RoPE.

/// Exact Q/K preparation with base RoPE tables and device-resident position.
///
/// # Safety
/// The `attention_qk_prepare` contract applies, with `rows == 1`. `past` is a
/// live, aligned u32, uniform across the grid and not concurrently modified.
/// Both RoPE tables cover `(past + rows) * rotary_dim/2` BF16 elements. The host
/// validates `past + rows <= capacity <= 262144` before replay. No other pointer
/// or arithmetic interpretation changes. Launch [heads,1,1] x [256,1,1].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_qk_prepare_position(
    input: *const u16,
    weight: *const u16,
    cos: *const u16,
    sin: *const u16,
    output: *mut u16,
    normalized: *mut u16,
    unrounded: *mut f32,
    gate: *mut u16,
    rows: u32,
    heads: u32,
    width: u32,
    rotary_dim: u32,
    with_gate: u32,
    epsilon: f32,
    past: *const u32,
) {
    // SAFETY: The host owns the position and tables through graph destruction;
    // stream-ordered uploads precede replay, and the validated offset is in range.
    unsafe {
        let offset = *past as usize * (rotary_dim / 2) as usize;
        super::attention_prepare::prepare_body(
            input,
            weight,
            cos.add(offset),
            sin.add(offset),
            output,
            normalized,
            unrounded,
            gate,
            rows,
            heads,
            width,
            rotary_dim,
            with_gate,
            epsilon,
        );
    }
}

/// Exact KV append with a device-resident position in place of the by-value u32.
///
/// # Safety
/// All `attention_kv_append` contracts apply, with `rows == 1`. `past` is a live,
/// aligned, grid-uniform u32, uploaded on the replay stream and unchanged until
/// completion. Its value satisfies the original capacity contract.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_kv_append_position(
    k: *const u16,
    v: *const u16,
    cache_k: *mut u16,
    cache_v: *mut u16,
    rows: u32,
    kv_heads: u32,
    width: u32,
    past: *const u32,
    capacity: u32,
) {
    // SAFETY: The position and all original append contracts are upheld by the host.
    unsafe {
        super::causal_attention::append_body(
            k, v, cache_k, cache_v, rows, kv_heads, width, *past, capacity,
        );
    }
}

/// Exact FP64 attention with a device-resident position in place of the u32.
///
/// # Safety
/// All `causal_attention_bf16` contracts apply, with `rows == 1`. `past` is a live,
/// aligned, grid-uniform u32, uploaded on the replay stream and unchanged until
/// completion. Its value satisfies the original capacity contract.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn causal_attention_bf16_position(
    q: *const u16,
    cache_k: *const u16,
    cache_v: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: *const u32,
    capacity: u32,
    scale: f32,
) {
    // SAFETY: The position and original attention contracts are upheld by the host.
    unsafe {
        super::causal_attention::attention_body(
            q,
            cache_k,
            cache_v,
            output,
            unrounded,
            rows,
            query_heads,
            kv_heads,
            width,
            *past,
            capacity,
            scale,
        );
    }
}
