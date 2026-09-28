//! Row-scaled E4M3 embedding lookup with an explicit BF16 boundary before norm.

use crate::embedding_norm::{encode_bf16_rne, fp32_multiply_rn, thread_and_row};

/// Logical E4M3FN decoding, including signed zero, subnormals and finite exponent 15.
#[inline(always)]
fn decode_e4m3(code: u8) -> f32 {
    let sign = u32::from(code & 0x80) << 24;
    let exponent = u32::from((code >> 3) & 0x0f);
    let fraction = u32::from(code & 7);
    let magnitude = if exponent == 0 {
        // Every subnormal is an exact integer multiple of 2^-9 in FP32.
        (fraction as f32) * 0.001_953_125_f32
    } else if exponent == 15 && fraction == 7 {
        f32::from_bits(0x7fc0_0000)
    } else {
        f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
    };
    f32::from_bits(magnitude.to_bits() | sign)
}

/// Gather only the requested rows: BF16_RNE(float(code) * float(row_scale)).
///
/// # Safety
/// Launch one 256-thread CTA per output row. `codes` is row-major E4M3FN
/// `[vocabulary, width]`, `scales` is BF16 `[vocabulary, 1]`, and `tokens` contains
/// one validated in-range u32 index per output row. `output` holds writable BF16
/// `[rows, width]`. Width is 1..=32768. Pointers must have their natural alignment,
/// have disjoint extents, and remain live until completion; all offset arithmetic
/// must fit usize. The caller validates rows, token bounds and allocation sizes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_embedding_gather(
    codes: *const u8,
    scales: *const u16,
    tokens: *const u32,
    output: *mut u16,
    width: u32,
) {
    let (thread, row) = thread_and_row();
    // SAFETY: Each CTA has a validated token and its per-vocabulary-row scale.
    let token = unsafe { *tokens.add(row as usize) } as usize;
    let scale = f32::from_bits(u32::from(unsafe { *scales.add(token) }) << 16);
    let source = token * width as usize;
    let destination = row as usize * width as usize;
    let mut column = thread;
    while column < width {
        // SAFETY: Token and width bound the source; threads own distinct output columns.
        let code = unsafe { *codes.add(source + column as usize) };
        let product = fp32_multiply_rn(decode_e4m3(code), scale);
        unsafe {
            output
                .add(destination + column as usize)
                .write(encode_bf16_rne(product));
        }
        column += 256;
    }
}
