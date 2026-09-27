use core::arch::asm;

const WARP_SIZE: u32 = 32;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile_n: u32;
    let row: u32;
    // SAFETY: Reads the calling thread's lane, CTA, and thread coordinates only.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {row}, %ctaid.y;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile_n = out(reg32) tile_n,
            row = out(reg32) row,
            options(nomem, nostack),
        )
    };
    (lane, thread, tile_n, row)
}

#[inline(always)]
fn e4m3fn_units(code: u8) -> i32 {
    let magnitude = code & 0x7f;
    let exponent = i32::from(magnitude >> 3);
    let fraction = i32::from(magnitude & 7);
    let units = if exponent == 0 {
        fraction
    } else {
        (8 + fraction) << (exponent - 1)
    };
    if code & 0x80 == 0 { units } else { -units }
}

#[inline(always)]
fn multiply_wide_s32(left: i32, right: i32) -> i64 {
    let product: i64;
    // SAFETY: This scalar signed integer operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.wide.s32 {product}, {left}, {right};",
            product = out(reg64) product,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    product
}

#[inline(always)]
fn add_s64(left: i64, right: i64) -> i64 {
    let sum: i64;
    // SAFETY: This scalar signed integer operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.s64 {sum}, {left}, {right};",
            sum = out(reg64) sum,
            left = in(reg64) left,
            right = in(reg64) right,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
fn warp_xor_u32(value: u32, lane_mask: u32) -> u32 {
    let result: u32;
    // SAFETY: All lanes in each warp execute every full-mask butterfly shuffle.
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {result}, {value}, {lane_mask}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            value = in(reg32) value,
            lane_mask = in(reg32) lane_mask,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn warp_reduce_s64(value: i64) -> i64 {
    let mut sum = value;
    let mut offset = WARP_SIZE / 2;
    while offset > 0 {
        let bits = sum as u64;
        let low = warp_xor_u32(bits as u32, offset);
        let high = warp_xor_u32((bits >> 32) as u32, offset);
        let partner = (((high as u64) << 32) | u64::from(low)) as i64;
        sum = add_s64(sum, partner);
        offset >>= 1;
    }
    sum
}

#[inline(always)]
fn i64_to_f32_rn(value: i64) -> f32 {
    let converted: f32;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.rn.f32.s64 {converted}, {value};",
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

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Compute an exact E4M3FN row-major linear layer with row and channel scales.
///
/// Each warp owns one output column, distributes K across its 32 lanes, and sums
/// exact signed integer products in units of `1/262144`. This avoids serial
/// rounding-refinement fallback while retaining the exact FP64-dot contract.
///
/// # Safety
/// Launch `grid = [ceil(n / 4), m, 1]`, `block = [128, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `a` and `w` must each
/// cover row-major matrices of `m * k` and `n * k` readable E4M3FN bytes. Codes
/// must be finite (not `0x7f` or `0xff`). `sa` must cover `m` readable finite
/// positive FP32 scales, and `sw` must cover `n` readable finite positive BF16
/// scales. `out` and `unrounded` must cover `m * n` writable BF16 and FP32 values.
/// All products and index arithmetic must fit the device address space. Pointers
/// must have their element alignment, be pairwise disjoint, and remain live until
/// kernel completion. The exact integer dot fits signed 64-bit for the stated K.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_linear_exact(
    a: *const u8,
    w: *const u8,
    sa: *const f32,
    sw: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (lane, thread, tile_n, row) = coordinates();
    let column = tile_n as usize * 4 + (thread / WARP_SIZE) as usize;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let row_valid = (row as usize) < m_usize;
    let mut partial = 0_i64;

    if row_valid && column < n_usize {
        let row_start = row as usize * k_usize;
        let weight_start = column * k_usize;
        let mut index = lane as usize;
        while index < k_usize {
            // SAFETY: The launch contract gives each valid row and column full K extents.
            let (a_code, w_code) =
                unsafe { (*a.add(row_start + index), *w.add(weight_start + index)) };
            let product = multiply_wide_s32(e4m3fn_units(a_code), e4m3fn_units(w_code));
            partial = add_s64(partial, product);
            index += WARP_SIZE as usize;
        }
    }

    let total = warp_reduce_s64(partial);
    if lane == 0 && row_valid && column < n_usize {
        let dot_units = i64_to_f32_rn(total);
        let dot = fp32_multiply_rn(dot_units, 1.0 / 262_144.0);
        // SAFETY: The valid row and output column have corresponding scale entries.
        let (row_scale, channel_scale) =
            unsafe { (*sa.add(row as usize), decode_bf16(*sw.add(column))) };
        let scaled = fp32_multiply_rn(fp32_multiply_rn(dot, row_scale), channel_scale);
        let output_index = row as usize * n_usize + column;
        // SAFETY: Lane zero of this warp uniquely owns this in-range matrix output.
        unsafe {
            unrounded.add(output_index).write(scaled);
            out.add(output_index).write(encode_bf16_rne(scaled));
        }
    }
}
