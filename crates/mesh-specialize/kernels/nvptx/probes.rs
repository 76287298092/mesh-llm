#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]

use core::arch::asm;

mod attention_gate;
mod attention_online;
mod attention_prepare;
mod attention_reduction;
mod attention_unrolled_fp64;
mod attention_warp_fp64;
mod bf16_ab_decode_fp32;
mod bf16_ab_decode_fp64;
mod bf16_linear;
mod bf16_linear_decode;
mod bf16_linear_rounding;
mod causal_attention;
mod causal_conv4;
mod embedding_norm;
mod exponential;
mod exponential_probe;
mod exponential_unrolled;
mod fp8_a16_decode;
mod fp8_a16_head;
mod fp8_embedding_gather;
mod fp8_linear;
mod fp8_linear_exact;
mod fp8_linear_exact4;
mod fp8_linear_exact_vector16;
mod fp8_linear_rounding;
mod fp8_native_prefill;
mod fp8_prefill_exact;
mod fp8_quantize;
mod fp8_swiglu_exact;
mod fp8_verify_exact;
mod gated_rms_norm;
mod gdn_chunked;
mod gdn_gates_f32_params;
mod gdn_prepare;
mod gdn_recurrent;
mod graph_position;
mod greedy_bf16;
mod kv_fp8;
mod logprob_math;
mod memory;
mod nvfp4_gemm;
mod ordinary_mma;
mod register_budget;
mod residual_norm;
mod rms_norm;
mod row_logprob_topk;
mod silu;
mod silu_probe;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    // SAFETY: an unexpected device panic must terminate the kernel.
    unsafe { asm!("trap;", options(noreturn)) }
}

/// One warp computes a 16x8 output tile from prepacked FP4 register fragments.
/// This is an instruction correctness probe, not a production GEMM.
///
/// # Safety
/// Launch exactly one 32-thread block. A/B contain 128/64 u32 words; each scale
/// input contains 32 words. Output contains 128 f32 elements. All pointers are
/// device allocations, disjoint, aligned, and remain live through completion.
/// `selector_a` is 0..=1 and `selector_b` is 0..=3, uniform across the warp.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe_nvfp4_mma(
    a: *const u32,
    b: *const u32,
    scale_a: *const u32,
    scale_b: *const u32,
    output: *mut f32,
    selector_a: u16,
    selector_b: u16,
) {
    let lane: u32;
    // SAFETY: reads this thread's lane identifier without changing memory.
    unsafe { asm!("mov.u32 {}, %laneid;", out(reg32) lane, options(nomem, nostack)) };
    let lane = lane as usize;
    // SAFETY: the launch contract supplies complete per-lane fragments.
    let (a0, a1, a2, a3, b0, b1, sa, sb) = unsafe {
        (
            *a.add(lane * 4),
            *a.add(lane * 4 + 1),
            *a.add(lane * 4 + 2),
            *a.add(lane * 4 + 3),
            *b.add(lane * 2),
            *b.add(lane * 2 + 1),
            *scale_a.add(lane),
            *scale_b.add(lane),
        )
    };
    let (mut d0, mut d1, mut d2, mut d3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    // SAFETY: all 32 lanes execute the same SM120a instruction with legal scale
    // selectors. Byte selectors are zero as required for scale_vec::4X.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {sa}, {{0, {ta}}}, {sb}, {{0, {tb}}};",
            d0 = inout(reg32) d0, d1 = inout(reg32) d1,
            d2 = inout(reg32) d2, d3 = inout(reg32) d3,
            a0 = in(reg32) a0, a1 = in(reg32) a1,
            a2 = in(reg32) a2, a3 = in(reg32) a3,
            b0 = in(reg32) b0, b1 = in(reg32) b1,
            sa = in(reg32) sa, sb = in(reg32) sb,
            ta = in(reg16) selector_a, tb = in(reg16) selector_b,
            options(nomem, nostack),
        );
        // SAFETY: every lane owns four distinct output slots.
        output.add(lane * 4).write(d0);
        output.add(lane * 4 + 1).write(d1);
        output.add(lane * 4 + 2).write(d2);
        output.add(lane * 4 + 3).write(d3);
    }
}

mod nvfp4_quantize;

mod nvfp4_decode;
mod nvfp4_decode_exact;
mod nvfp4_decode_prmt;
mod nvfp4_linear;
mod nvfp4_prefill_tiled;
mod nvfp4_prefill_wide;

mod mlp_activation;
mod residual_add;

mod gdn_replay;

mod attention_split;
mod attention_split_math;
mod attention_split_reduce;
