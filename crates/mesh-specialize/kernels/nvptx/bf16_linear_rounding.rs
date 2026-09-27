use core::arch::asm;

/// Refine a BF16 dot product only when its estimated error crosses a BF16 boundary.
///
/// `shape` contains `[row, column, width]` for row-major BF16 matrices `a[m, k]`
/// and `w[n, k]`. `absolute_sum` is the FP64 sum of zero-start FP32 MMA results
/// for `abs(A_tile) * abs(W_tile)` over the same K=16 partition as `approx`. The
/// 16-ppm allowance is a conservative empirical bound for FP32 accumulation inside
/// each tensor-core tile, with a 1-ppm result-relative floor. It is not a formal
/// floating-point error proof. If the widened interval rounds to one BF16 value,
/// the fast result is retained. Otherwise, the BF16 inputs are decoded and dotted
/// in increasing-K order with FP64 operations before one explicit FP32 conversion.
///
/// # Safety
/// `shape[2]` must be in `1..=32768`, and `shape[0]`/`shape[1]` must be valid row
/// and output-column indices in their respective matrices. `a` and `w` must cover
/// complete row-major matrices with `shape[2]` elements per row, all index arithmetic
/// must fit `usize`, and both pointers must remain live for this call. Matrix values
/// must be finite BF16 encodings. `approx` and `absolute_sum` must be finite, and
/// `absolute_sum` must be nonnegative and computed from the matching zero-start
/// absolute-product MMA tiles.
#[inline(always)]
pub(super) unsafe fn refine(
    a: *const u16,
    w: *const u16,
    shape: [usize; 3],
    approx: f32,
    absolute_sum: f64,
) -> f32 {
    let [row, column, width] = shape;
    let error = 16.0e-6_f64 * absolute_sum + 1.0e-6_f64 * f64::from(approx).abs();
    let lower = bf16_rne((f64::from(approx) - error) as f32);
    let upper = bf16_rne((f64::from(approx) + error) as f32);
    if lower == upper {
        return approx;
    }

    let row_start = row * width;
    let column_start = column * width;
    let mut dot = 0.0_f64;
    for index in 0..width {
        // SAFETY: The caller proves both matrix extents and the indexed row/column.
        let (a_value, w_value) = unsafe {
            (
                decode_bf16(*a.add(row_start + index)),
                decode_bf16(*w.add(column_start + index)),
            )
        };
        let product = fp64_multiply_rn(f64::from(a_value), f64::from(w_value));
        dot = fp64_add_rn(dot, product);
    }
    fp64_to_fp32_rn(dot)
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
fn fp64_multiply_rn(left: f64, right: f64) -> f64 {
    let product: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f64 {product}, {left}, {right};",
            product = out(reg64) product,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    product
}

#[inline(always)]
fn fp64_add_rn(left: f64, right: f64) -> f64 {
    let sum: f64;
    // SAFETY: This scalar FP64 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f64 {sum}, {left}, {right};",
            sum = out(reg64) sum,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    sum
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
