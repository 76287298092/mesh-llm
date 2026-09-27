use core::arch::asm;

/// Refine a scaled FP8 dot only when its FP32 error interval crosses a BF16 boundary.
///
/// `shape` is `[row, column, width]` for row-major E4M3FN matrices `a[m, k]` and
/// `w[n, k]`. `absolute_sum` is the FP64 sum of zero-start FP32 MMA results for
/// `abs(A_tile) * abs(W_tile)` over the same K=32 partition as the fast kernel.
/// This bounds cancellation within each signed tile. Its 16-ppm allowance exceeds
/// typical FP32 roundoff within a 32-value MMA tile; the additional 1-ppm term
/// provides a result-relative floor. If this deliberately
/// widened interval maps to one BF16 value, the fast result is retained. If it
/// spans a BF16 boundary, the exact E4M3 dot is decoded and accumulated in FP64.
/// Each finite E4M3 value is an integer multiple of 1/512, so products and sums
/// for widths up to 32,768 remain exactly representable in FP64.
///
/// # Safety
/// `shape[2]` must be in `1..=32768`. `a` must address a row-major matrix with
/// stride `shape[2]` and at least `(shape[0] + 1) * shape[2]` readable bytes;
/// `w` must address a row-major matrix with the same stride and at least
/// `(shape[1] + 1) * shape[2]` readable bytes. All index arithmetic must fit
/// `usize`, and both pointers must remain live for this call. Matrix values must
/// be finite E4M3FN codes (excluding the two NaN encodings). `approx` and
/// `absolute_sum` must be finite, `absolute_sum` nonnegative, and both `scales`
/// finite and positive. `absolute_sum` must be computed from zero-start FP32
/// positive-product MMA tiles (`abs(A_tile)` by `abs(W_tile)`) over the same K=32
/// partition as `approx`.
#[inline(always)]
pub(super) unsafe fn refine(
    a: *const u8,
    w: *const u8,
    shape: [usize; 3],
    approx: f32,
    absolute_sum: f64,
    scales: [f32; 2],
) -> f32 {
    let [row, column, width] = shape;
    let error = 16.0e-6_f64 * absolute_sum * f64::from(scales[0]) * f64::from(scales[1])
        + 1.0e-6_f64 * f64::from(approx).abs();
    let lower = bf16_rne((f64::from(approx) - error) as f32);
    let upper = bf16_rne((f64::from(approx) + error) as f32);
    if lower == upper {
        return approx;
    }

    let row_start = row * width;
    let column_start = column * width;
    let mut dot = 0.0_f64;
    for index in 0..width {
        // SAFETY: The caller proves both row-major extents and each computed offset.
        let (a_code, w_code) = unsafe { (*a.add(row_start + index), *w.add(column_start + index)) };
        let product = f64::from(decode_e4m3fn(a_code)) * f64::from(decode_e4m3fn(w_code));
        dot += product;
    }

    let dot_f32 = fp64_to_fp32_rn(dot);
    let row_scaled = fp32_multiply_rn(dot_f32, scales[0]);
    fp32_multiply_rn(row_scaled, scales[1])
}

#[inline(always)]
fn decode_e4m3fn(code: u8) -> f32 {
    let magnitude = code & 0x7f;
    if magnitude == 0x7f {
        return f32::NAN;
    }

    let sign = u32::from(code & 0x80) << 24;
    let exponent = u32::from(magnitude >> 3);
    let fraction = u32::from(magnitude & 7);
    if exponent == 0 {
        let subnormal = (fraction as f32) * (1.0_f32 / 512.0_f32);
        f32::from_bits(subnormal.to_bits() | sign)
    } else {
        f32::from_bits(sign | ((exponent + 120) << 23) | (fraction << 20))
    }
}

#[inline(always)]
fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
fn fp64_to_fp32_rn(value: f64) -> f32 {
    let converted: f32;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.rn.f32.f64 {converted}, {value};",
            converted = out(reg32) converted,
            value = in(reg64) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
fn fp32_multiply_rn(left: f32, right: f32) -> f32 {
    let product: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f32 {product}, {left}, {right};",
            product = out(reg32) product,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    product
}
