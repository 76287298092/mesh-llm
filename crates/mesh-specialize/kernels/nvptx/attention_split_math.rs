//! SM120 split-attention primitives. All shared addresses are CTA-local byte addresses.
use core::arch::asm;

#[inline(always)]
pub(super) fn coordinates() -> (u32, u32, u32) {
    let (t, h, s): (u32, u32, u32);
    // SAFETY: Special-register reads have no memory effects.
    unsafe {
        asm!("mov.u32 {t}, %tid.x;", "mov.u32 {h}, %ctaid.x;", "mov.u32 {s}, %ctaid.y;",
        t=out(reg32)t, h=out(reg32)h, s=out(reg32)s, options(nomem, nostack));
    }
    (t, h, s)
}

#[inline(always)]
pub(super) fn shared_base() -> u32 {
    let base: u32;
    // SAFETY: Two BF16 [16,256] tiles, statically allocated per CTA.
    unsafe {
        asm!(".shared .align 16 .b8 attention_split_tiles[16384];",
        "mov.u32 {base}, attention_split_tiles;", base=out(reg32)base, options(nostack));
    }
    base
}

#[inline(always)]
pub(super) fn barrier() {
    // SAFETY: All 192 threads execute each call in the partial kernel.
    unsafe {
        asm!("bar.sync 0;", options(nostack));
    }
}

#[inline(always)]
pub(super) fn store_word(address: u32, value: u32) {
    // SAFETY: Caller supplies its unique aligned word in the 16 KiB tile.
    unsafe {
        asm!("st.shared.b32 [{address}], {value};",
        address=in(reg32)address, value=in(reg32)value, options(nostack));
    }
}

#[inline(always)]
pub(super) fn load_bf16(address: u32) -> f32 {
    let bits: u16;
    // SAFETY: Caller bounds this halfword in a published shared tile.
    unsafe {
        asm!("ld.shared.b16 {bits}, [{address}];",
        bits=out(reg16)bits, address=in(reg32)address, options(nostack));
    }
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
pub(super) fn warp_sum(mut value: f32) -> f32 {
    let mut mask = 16;
    while mask != 0 {
        let other: f32;
        // SAFETY: All 32 lanes in each query-head warp participate, full warp width.
        unsafe {
            asm!("shfl.sync.bfly.b32 {other}, {value}, {mask}, 31, -1;",
            other=out(reg32)other, value=in(reg32)value, mask=in(reg32)mask,
            options(nomem, nostack));
        }
        value += other;
        mask >>= 1;
    }
    value
}

#[inline(always)]
pub(super) fn exp(value: f32) -> f32 {
    let out: f32;
    let exponent = value * core::f32::consts::LOG2_E;
    // SAFETY: Scalar approximate exponential; callers use nonpositive finite arguments.
    unsafe {
        asm!("ex2.approx.f32 {out}, {exponent};",
        out=out(reg32)out, exponent=in(reg32)exponent, options(nomem, nostack));
    }
    out
}

#[inline(always)]
pub(super) fn round_bf16(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Scalar RNE conversion with no memory effects.
    unsafe {
        asm!("cvt.rn.bf16.f32 {bits}, {value};",
        bits=out(reg16)bits, value=in(reg32)value, options(nomem, nostack));
    }
    bits
}

#[inline(always)]
pub(super) fn divide(a: f32, b: f32) -> f32 {
    let out: f32;
    // SAFETY: Valid query rows have a positive combined denominator.
    unsafe {
        asm!("div.rn.f32 {out}, {a}, {b};", out=out(reg32)out,
        a=in(reg32)a, b=in(reg32)b, options(nomem, nostack));
    }
    out
}
