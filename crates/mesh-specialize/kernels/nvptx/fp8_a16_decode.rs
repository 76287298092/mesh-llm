use core::arch::asm;

const WARP_SIZE: usize = 32;
const WARPS_PER_BLOCK: usize = 4;
const VALUES_PER_LANE: usize = 4;
const K_STRIDE: usize = WARP_SIZE * VALUES_PER_LANE;

#[inline(always)]
fn coordinates() -> (usize, usize, usize) {
    let lane: u32;
    let thread: u32;
    let tile: u32;
    // SAFETY: These special registers identify the calling thread and CTA.
    unsafe {
        asm!(
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile}, %ctaid.x;",
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile = out(reg32) tile,
            options(nomem, nostack),
        )
    };
    (lane as usize, (thread as usize) / WARP_SIZE, tile as usize)
}

#[inline(always)]
unsafe fn load_packed_e4m3x4(pointer: *const u8) -> (u8, u8, u8, u8) {
    let packed: u32;
    // SAFETY: The caller checks four readable bytes and four-byte alignment.
    unsafe {
        asm!(
            "ld.global.u32 {packed}, [{pointer}];",
            packed = out(reg32) packed,
            pointer = in(reg64) pointer as u64,
            options(nostack),
        )
    };
    (
        packed as u8,
        (packed >> 8) as u8,
        (packed >> 16) as u8,
        (packed >> 24) as u8,
    )
}

#[inline(always)]
unsafe fn load_packed_bf16x4(pointer: *const u16) -> (u16, u16, u16, u16) {
    let packed: u64;
    // SAFETY: The caller checks four readable BF16 values and eight-byte alignment.
    unsafe {
        asm!(
            "ld.global.u64 {packed}, [{pointer}];",
            packed = out(reg64) packed,
            pointer = in(reg64) pointer as u64,
            options(nostack),
        )
    };
    (
        packed as u16,
        (packed >> 16) as u16,
        (packed >> 32) as u16,
        (packed >> 48) as u16,
    )
}

#[inline(always)]
fn decode_e4m3fn(code: u8) -> f32 {
    let magnitude = u32::from(code & 0x7f);
    if magnitude == 0x7f {
        return f32::NAN;
    }
    let exponent = magnitude >> 3;
    let fraction = magnitude & 7;
    let bits = if exponent == 0 {
        (fraction as f32 * f32::from_bits(0x3b00_0000)).to_bits()
    } else {
        ((exponent + 120) << 23) | (fraction << 20)
    };
    f32::from_bits(bits | (u32::from(code & 0x80) << 24))
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline(always)]
fn fma_rn(accumulator: f32, left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: Scalar fused multiply-add has no memory or stack effects.
    unsafe {
        asm!(
            "fma.rn.f32 {result}, {left}, {right}, {accumulator};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            accumulator = in(reg32) accumulator,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn add_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: Scalar addition has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {result}, {left}, {right};",
            result = out(reg32) result,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn warp_sum_f32(value: f32) -> f32 {
    let mut sum = value;
    let mut offset = WARP_SIZE / 2;
    while offset > 0 {
        let partner_bits: u32;
        // SAFETY: Every lane in each warp executes each full-mask XOR shuffle.
        unsafe {
            asm!(
                "shfl.sync.bfly.b32 {partner}, {value}, {offset}, 0x1f, 0xffffffff;",
                partner = out(reg32) partner_bits,
                value = in(reg32) sum.to_bits(),
                offset = in(reg32) offset as u32,
                options(nomem, nostack),
            )
        };
        sum = add_rn(sum, f32::from_bits(partner_bits));
        offset >>= 1;
    }
    sum
}

#[inline(always)]
unsafe fn row_partial(
    input: *const u16,
    weight: *const u8,
    column: usize,
    k: usize,
    lane: usize,
) -> f32 {
    let row_start = column * k;
    // One 32-bit FP32 FMA chain is kept for each of the four adjacent values.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let input_vector_aligned = (input as usize & 7) == 0;
    // K-row starts may be unaligned even when the allocation base is aligned.
    let weight_vector_aligned = ((weight as usize + row_start) & 3) == 0;
    let vector_loads = input_vector_aligned && weight_vector_aligned;
    let mut base = lane * VALUES_PER_LANE;

    while base < k {
        if vector_loads && base + VALUES_PER_LANE <= k {
            // SAFETY: This iteration has four in-range values; the addresses meet
            // the alignment requirements established above.
            let (x0_bits, x1_bits, x2_bits, x3_bits, w0, w1, w2, w3) = unsafe {
                let xs = load_packed_bf16x4(input.add(base));
                let ws = load_packed_e4m3x4(weight.add(row_start + base));
                (xs.0, xs.1, xs.2, xs.3, ws.0, ws.1, ws.2, ws.3)
            };
            acc0 = fma_rn(acc0, decode_bf16(x0_bits), decode_e4m3fn(w0));
            acc1 = fma_rn(acc1, decode_bf16(x1_bits), decode_e4m3fn(w1));
            acc2 = fma_rn(acc2, decode_bf16(x2_bits), decode_e4m3fn(w2));
            acc3 = fma_rn(acc3, decode_bf16(x3_bits), decode_e4m3fn(w3));
        } else {
            if base < k {
                // SAFETY: base is below K and this lane owns the current row.
                let (x, w) = unsafe { (*input.add(base), *weight.add(row_start + base)) };
                acc0 = fma_rn(acc0, decode_bf16(x), decode_e4m3fn(w));
            }
            if base + 1 < k {
                // SAFETY: base + 1 is below K and this lane owns the current row.
                let (x, w) = unsafe { (*input.add(base + 1), *weight.add(row_start + base + 1)) };
                acc1 = fma_rn(acc1, decode_bf16(x), decode_e4m3fn(w));
            }
            if base + 2 < k {
                // SAFETY: base + 2 is below K and this lane owns the current row.
                let (x, w) = unsafe { (*input.add(base + 2), *weight.add(row_start + base + 2)) };
                acc2 = fma_rn(acc2, decode_bf16(x), decode_e4m3fn(w));
            }
            if base + 3 < k {
                // SAFETY: base + 3 is below K and this lane owns the current row.
                let (x, w) = unsafe { (*input.add(base + 3), *weight.add(row_start + base + 3)) };
                acc3 = fma_rn(acc3, decode_bf16(x), decode_e4m3fn(w));
            }
        }
        base += K_STRIDE;
    }

    add_rn(add_rn(add_rn(acc0, acc1), acc2), acc3)
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

/// Compute one-row-A16 by row-major-E4M3 weights and BF16 per-output scales.
///
/// The kernel reads one BF16 activation row without activation quantization. Four
/// warps form one CTA, with each warp producing one output row. Each lane holds
/// four independent FP32 FMA chains; aligned code and activation groups use wide
/// loads, while misaligned rows and K tails use guarded scalar loads. The final
/// FP32 dot is multiplied by the represented BF16 row scale and rounded to BF16.
/// This arithmetic profile differs from the existing A8 decode path.
///
/// # Safety
/// Launch `grid = [ceil(n / 4), 1, 1]`, `block = [128, 1, 1]`, with nonzero `n`
/// and `k`. `input` must cover `k` readable BF16 values, `weight` must cover
/// `n * k` readable finite E4M3FN bytes in row-major order, and `weight_scale`
/// must cover `n` readable finite BF16 values. `out` and `unrounded` must each
/// cover `n` writable BF16 and FP32 values. All pointers must be correctly aligned
/// for their element types, pairwise disjoint, and live until kernel completion.
/// The host must check shape products, launch dimensions, and finite outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_a16_decode(
    input: *const u16,
    weight: *const u8,
    weight_scale: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    n: u32,
    k: u32,
) {
    let (lane, warp, tile) = coordinates();
    let column = tile * WARPS_PER_BLOCK + warp;
    let n = n as usize;
    let k = k as usize;
    let partial = if column < n {
        // SAFETY: The launch contract guarantees complete input and weight rows.
        unsafe { row_partial(input, weight, column, k, lane) }
    } else {
        0.0
    };
    let dot = warp_sum_f32(partial);

    if lane == 0 && column < n {
        // SAFETY: Lane zero exclusively owns the in-range output column.
        let scale = unsafe { decode_bf16(*weight_scale.add(column)) };
        let scaled = {
            let result: f32;
            // SAFETY: Scalar multiplication has no memory or stack effects.
            unsafe {
                asm!(
                    "mul.rn.f32 {result}, {dot}, {scale};",
                    result = out(reg32) result,
                    dot = in(reg32) dot,
                    scale = in(reg32) scale,
                    options(nomem, nostack),
                )
            };
            result
        };
        // SAFETY: Lane zero owns the unique in-range result elements.
        unsafe {
            unrounded.add(column).write(scaled);
            out.add(column).write(encode_bf16_rne(scaled));
        }
    }
}
