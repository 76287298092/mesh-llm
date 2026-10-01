use super::super::super::{half_to_f32, multiply_rn};
use core::arch::asm;

#[inline(always)]
fn half_pair(encoded: u32, scale: f32) -> (f32, f32) {
    let decoded: u32;
    let bias = 0x6408_6408_u32;
    // SAFETY: The mantissa decoder creates finite half values within subtraction range.
    unsafe {
        asm!(
            "sub.rn.f16x2 {decoded}, {encoded}, {bias};",
            decoded = out(reg32) decoded,
            encoded = in(reg32) encoded,
            bias = in(reg32) bias,
            options(nomem, nostack),
        );
    }
    (
        multiply_rn(half_to_f32(decoded as u16), scale),
        multiply_rn(half_to_f32((decoded >> 16) as u16), scale),
    )
}

#[inline(always)]
pub(super) fn decode_eight(packed: u32, scale_bits: u16) -> [f32; 8] {
    let word = packed ^ 0x8888_8888;
    let scale = half_to_f32(scale_bits);
    let (weight0, weight4) = half_pair((word & 0x000f_000f) | 0x6400_6400, scale);
    let (weight1, weight5) = half_pair(((word >> 4) & 0x000f_000f) | 0x6400_6400, scale);
    let (weight2, weight6) = half_pair(((word >> 8) & 0x000f_000f) | 0x6400_6400, scale);
    let (weight3, weight7) = half_pair(((word >> 12) & 0x000f_000f) | 0x6400_6400, scale);
    [weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7]
}
