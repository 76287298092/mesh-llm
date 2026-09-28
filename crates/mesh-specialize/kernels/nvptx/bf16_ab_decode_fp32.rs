//! Experimental M=1 BF16 A/B dots. No activation quantization or gate math.
use core::arch::asm;

#[inline(always)]
fn coordinates() -> (usize, usize, usize) {
    let thread: u32;
    let head: u32;
    let projection: u32;
    // SAFETY: Special-register reads have no memory effects.
    unsafe {
        asm!("mov.u32 {t}, %tid.x;", "mov.u32 {h}, %ctaid.x;",
            "mov.u32 {p}, %ctaid.y;", t=out(reg32) thread,
            h=out(reg32) head, p=out(reg32) projection, options(nomem, nostack));
    }
    (thread as usize, head as usize, projection as usize)
}

#[inline(always)]
unsafe fn load8(pointer: *const u16) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: Caller supplies eight readable BF16 elements at a 16-byte boundary.
    unsafe {
        asm!("ld.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a=out(reg32) a, b=out(reg32) b, c=out(reg32) c, d=out(reg32) d,
            p=in(reg64) pointer as u64, options(nostack));
    }
    [a, b, c, d]
}

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    let result: f32;
    // SAFETY: Explicit FP32 RNE FMA, no memory effects or FTZ.
    unsafe {
        asm!("fma.rn.f32 {r}, {a}, {b}, {c};", r=out(reg32) result,
            a=in(reg32) a, b=in(reg32) b, c=in(reg32) c, options(nomem, nostack));
    }
    result
}

#[inline(always)]
fn add(a: f32, b: f32) -> f32 {
    let result: f32;
    // SAFETY: Explicit FP32 RNE addition has no memory effects.
    unsafe {
        asm!("add.rn.f32 {r}, {a}, {b};", r=out(reg32) result,
            a=in(reg32) a, b=in(reg32) b, options(nomem, nostack));
    }
    result
}

#[inline(always)]
fn sum_warp(mut value: f32) -> f32 {
    let mut offset = 16_u32;
    while offset != 0 {
        let partner: u32;
        // SAFETY: Every lane of the participating warp executes the full-mask shuffle.
        unsafe {
            asm!("shfl.sync.bfly.b32 {r}, {v}, {o}, 0x1f, 0xffffffff;",
                r=out(reg32) partner, v=in(reg32) value.to_bits(),
                o=in(reg32) offset, options(nomem, nostack));
        }
        value = add(value, f32::from_bits(partner));
        offset >>= 1;
    }
    value
}

#[inline(always)]
fn shared_base() -> u32 {
    let base: u32;
    // SAFETY: Four warp partials, one 16-byte array per CTA.
    unsafe {
        asm!(".shared .align 4 .b8 bf16_ab_fp32_partials[16];",
            "mov.u32 {b}, bf16_ab_fp32_partials;", b=out(reg32) base, options(nostack));
    }
    base
}

#[inline(always)]
fn store_partial(base: u32, warp: usize, value: f32) {
    // SAFETY: Only lane zero writes its warp's distinct slot, warp in 0..4.
    unsafe {
        asm!("st.shared.f32 [{p}], {v};", p=in(reg32) base + warp as u32 * 4,
            v=in(reg32) value, options(nostack));
    }
}

#[inline(always)]
fn load_partial(base: u32, lane: usize) -> f32 {
    let value: f32;
    // SAFETY: Called after a CTA barrier, for initialized slots with lane in 0..4.
    unsafe {
        asm!("ld.shared.f32 {v}, [{p}];", v=out(reg32) value,
            p=in(reg32) base + lane as u32 * 4, options(nostack));
    }
    value
}

#[inline(always)]
fn pair_dot(x: u32, w: u32, accumulator: f32) -> f32 {
    let first = fma(
        f32::from_bits(x << 16),
        f32::from_bits(w << 16),
        accumulator,
    );
    fma(
        f32::from_bits(x & 0xffff_0000),
        f32::from_bits(w & 0xffff_0000),
        first,
    )
}

#[inline(always)]
unsafe fn partial(input: *const u16, weights: *const u16, k: usize, thread: usize) -> f32 {
    let (mut a, mut b, mut c, mut d) = (0.0, 0.0, 0.0, 0.0);
    let mut base = thread * 8;
    while base < k {
        // SAFETY: K is divisible by eight; row bases and every group are 16-byte aligned.
        let (x, w) = unsafe { (load8(input.add(base)), load8(weights.add(base))) };
        a = pair_dot(x[0], w[0], a);
        b = pair_dot(x[1], w[1], b);
        c = pair_dot(x[2], w[2], c);
        d = pair_dot(x[3], w[3], d);
        base += 128 * 8;
    }
    add(add(a, b), add(c, d))
}

#[inline(always)]
fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

/// Compute two independent BF16 matrices times a shared BF16 activation row.
///
/// CTA (head, 0) computes A; CTA (head, 1) computes B. Each CTA uses four warps
/// over K, four independent FP32 FMA chains per thread, one warp reduction, a
/// 16-byte shared exchange, and a second warp reduction. No global scratch.
///
/// # Safety
/// Launch grid [n, 2, 1], block [128, 1, 1], dynamic shared 0. Require
/// 1 <= n <= 256 and k divisible by 8 in 8..=32768; M is always one.
/// Input is BF16 [k], weights_a/b are separate row-major BF16 [n,k]. All three
/// bases must be 16-byte aligned. All inputs must be finite. Outputs a/b cover
/// n BF16 elements, raw_a/b cover n FP32 elements, naturally aligned. All seven
/// pointers must be disjoint and live until completion. Products, all partial
/// sums and outputs must remain finite; qualification rejects BF16 overflow.
/// Host validates shape/extent/launch constraints. This changes FP64 summation
/// to FP32 and must not replace the exact profile without separate admission.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bf16_ab_decode_fp32(
    input: *const u16,
    weights_a: *const u16,
    weights_b: *const u16,
    output_a: *mut u16,
    output_b: *mut u16,
    raw_a: *mut f32,
    raw_b: *mut f32,
    n: u32,
    k: u32,
) {
    let (thread, head, projection) = coordinates();
    // CTA-uniform guards. All remaining threads reach the barrier.
    if head >= n as usize || projection >= 2 {
        return;
    }
    let (weights, output, raw) = if projection == 0 {
        (weights_a, output_a, raw_a)
    } else {
        (weights_b, output_b, raw_b)
    };
    let lane = thread % 32;
    let warp = thread / 32;
    // SAFETY: The host contract covers the complete selected row and input.
    let local = unsafe { partial(input, weights.add(head * k as usize), k as usize, thread) };
    let sum = sum_warp(local);
    let shared = shared_base();
    if lane == 0 {
        store_partial(shared, warp, sum);
    }
    // SAFETY: Every thread of the 128-thread CTA reaches this barrier.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
    if warp == 0 {
        let value = if lane < 4 {
            load_partial(shared, lane)
        } else {
            0.0
        };
        let total = sum_warp(value);
        if lane == 0 {
            // SAFETY: Each CTA owns one distinct head in one selected output pair.
            unsafe {
                raw.add(head).write(total);
                output.add(head).write(round_bf16(total));
            }
        }
    }
}
