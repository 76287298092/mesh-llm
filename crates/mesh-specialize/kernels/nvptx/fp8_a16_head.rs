//! Bounded experimental A16 sliced-K head. Independent Rust implementation.
use core::arch::asm;

const WARPS: usize = 16;
const TILE_VALUES: usize = 128;

#[inline(always)]
fn coordinates() -> (usize, usize) {
    let thread: u32;
    let tile: u32;
    // SAFETY: Special-register reads have no memory effects.
    unsafe {
        asm!("mov.u32 {thread}, %tid.x;", "mov.u32 {tile}, %ctaid.x;",
            thread = out(reg32) thread, tile = out(reg32) tile,
            options(nomem, nostack));
    }
    (thread as usize, tile as usize)
}

#[inline(always)]
fn partial_base() -> u32 {
    let base: u32;
    // SAFETY: One CTA-local 16 * 128 * 4 byte allocation.
    unsafe {
        asm!(".shared .align 4 .b8 a16_head_partials[8192];",
            "mov.u32 {base}, a16_head_partials;", base = out(reg32) base,
            options(nostack));
    }
    base
}

#[inline(always)]
fn store_partial(base: u32, index: usize, value: f32) {
    // SAFETY: Internal callers supply a unique index below 2048.
    unsafe {
        asm!("st.shared.f32 [{address}], {value};",
            address = in(reg32) base + index as u32 * 4,
            value = in(reg32) value, options(nostack));
    }
}

#[inline(always)]
fn load_partial(base: u32, index: usize) -> f32 {
    let value: f32;
    // SAFETY: Internal callers supply an initialized index below 2048 after bar.sync.
    unsafe {
        asm!("ld.shared.f32 {value}, [{address}];",
            address = in(reg32) base + index as u32 * 4,
            value = out(reg32) value, options(nostack));
    }
    value
}

#[inline(always)]
fn add_rn(left: f32, right: f32) -> f32 {
    let value: f32;
    // SAFETY: Register-only scalar addition.
    unsafe {
        asm!("add.rn.f32 {value}, {left}, {right};", value = out(reg32) value,
            left = in(reg32) left, right = in(reg32) right, options(nomem, nostack));
    }
    value
}

#[inline(always)]
fn scale_rn(dot: f32, scale: u16) -> f32 {
    let value: f32;
    // SAFETY: Register-only multiplication after the complete CTA reduction.
    unsafe {
        asm!("mul.rn.f32 {value}, {dot}, {scale};", value = out(reg32) value,
            dot = in(reg32) dot, scale = in(reg32) f32::from_bits(u32::from(scale) << 16),
            options(nomem, nostack));
    }
    value
}

#[inline(always)]
fn widen(code: u8) -> u16 {
    let sign = u16::from(code & 128) << 8;
    let exp = u16::from((code >> 3) & 15);
    let frac = u16::from(code & 7);
    // All E4M3FN finite values, including subnormals, are exact BF16 values.
    let magnitude = if exp != 0 {
        ((exp + 120) << 7) | (frac << 4)
    } else {
        match frac {
            0 => 0,
            1 => 0x3b00,
            2 => 0x3b80,
            3 => 0x3bc0,
            4 => 0x3c00,
            5 => 0x3c20,
            6 => 0x3c40,
            _ => 0x3c60,
        }
    };
    sign | magnitude
}

#[inline(always)]
unsafe fn activation_pair(input: *const u16, row: usize, kk: usize, m: usize, k: usize) -> u32 {
    if row >= m {
        return 0;
    }
    // SAFETY: Caller supplies an in-range K pair and the full M*K allocation.
    unsafe { u32::from(*input.add(row * k + kk)) | (u32::from(*input.add(row * k + kk + 1)) << 16) }
}

#[inline(always)]
unsafe fn weight_pair(weight: *const u8, column: usize, kk: usize, k: usize) -> u32 {
    // SAFETY: Caller supplies an in-range output column and K pair.
    unsafe {
        u32::from(widen(*weight.add(column * k + kk)))
            | (u32::from(widen(*weight.add(column * k + kk + 1))) << 16)
    }
}

#[inline(always)]
fn mma(a: [u32; 4], b: [u32; 2], d: [f32; 4]) -> [f32; 4] {
    let [mut d0, mut d1, mut d2, mut d3] = d;
    // SAFETY: All 32 lanes participate with the documented m16n8k16 fragments.
    unsafe {
        asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d0, d1 = inout(reg32) d1,
            d2 = inout(reg32) d2, d3 = inout(reg32) d3,
            a0 = in(reg32) a[0], a1 = in(reg32) a[1],
            a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
    [d0, d1, d2, d3]
}

#[inline(always)]
fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Experimental BF16 input / E4M3FN weight projection, with signed BF16 scales.
///
/// # Safety
/// Launch exactly grid=[N/8,1,1], block=[512,1,1]. Require 1<=M<=8,
/// 8<=N<=262144 with N divisible by 8, and 16<=K<=32768 divisible by 16.
/// Input[M,K], weight[N,K], scales[N], out[M,N], unrounded[M,N] are contiguous
/// row-major arrays of their pointer element types, correctly aligned, pairwise
/// nonoverlapping and live until completion. Inputs and scales are finite BF16;
/// weights exclude E4M3FN NaN codes 0x7f/0xff. Host checks products, launch limits,
/// device BF16 MMA support, and finite FP32/BF16 results. Finite inputs do not
/// guarantee finite intermediate sums. This profile is not exact-A8 arithmetic.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_a16_head(
    input: *const u16,
    weight: *const u8,
    weight_scale: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (thread, tile) = coordinates();
    let (lane, warp) = (thread % 32, thread / 32);
    let (group, pair) = (lane / 4, (lane % 4) * 2);
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut d = [0.0; 4];
    let mut kk = warp * 16;
    while kk < k {
        // SAFETY: K is a multiple of 16; each warp consumes complete K16 tiles.
        // A elements 0/1,2/3,4/5,6/7 use (g,p),(g+8,p),(g,p+8),(g+8,p+8).
        // B elements 0/1,2/3 use K p,p+8 at output column g.
        let (a, b) = unsafe {
            (
                [
                    activation_pair(input, group, kk + pair, m, k),
                    activation_pair(input, group + 8, kk + pair, m, k),
                    activation_pair(input, group, kk + pair + 8, m, k),
                    activation_pair(input, group + 8, kk + pair + 8, m, k),
                ],
                [
                    weight_pair(weight, tile * 8 + group, kk + pair, k),
                    weight_pair(weight, tile * 8 + group, kk + pair + 8, k),
                ],
            )
        };
        d = mma(a, b, d);
        kk += WARPS * 16;
    }
    let shared = partial_base();
    for (slot, value) in d.into_iter().enumerate() {
        store_partial(shared, warp * TILE_VALUES + lane * 4 + slot, value);
    }
    // SAFETY: Every thread reaches this barrier once, including idle K slices.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
    if warp == 0 {
        for slot in 0..4 {
            let row = group + (slot / 2) * 8;
            let column = tile * 8 + pair + slot % 2;
            if row < m {
                let mut sum = load_partial(shared, lane * 4 + slot);
                for split in 1..WARPS {
                    sum = add_rn(
                        sum,
                        load_partial(shared, split * TILE_VALUES + lane * 4 + slot),
                    );
                }
                // SAFETY: Warp zero exclusively writes each valid tile element.
                unsafe {
                    let value = scale_rn(sum, *weight_scale.add(column));
                    unrounded.add(row * n + column).write(value);
                    out.add(row * n + column).write(round_bf16(value));
                }
            }
        }
    }
}
