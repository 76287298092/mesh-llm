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
fn fp32_to_fp64_exact(value: f32) -> f64 {
    let converted: f64;
    // SAFETY: This scalar conversion has no memory or stack effects.
    unsafe {
        asm!(
            "cvt.f64.f32 {converted}, {value};",
            converted = out(reg64) converted,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    converted
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f64 {
    let value = f32::from_bits((bits as u32) << 16);
    fp32_to_fp64_exact(value)
}

#[inline(always)]
fn multiply_rn(left: f64, right: f64) -> f64 {
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
fn add_rn(left: f64, right: f64) -> f64 {
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
fn warp_reduce_f64(value: f64) -> f64 {
    let mut sum = value;
    let mut offset = WARP_SIZE / 2;
    while offset > 0 {
        let bits = sum.to_bits();
        let low = warp_xor_u32(bits as u32, offset);
        let high = warp_xor_u32((bits >> 32) as u32, offset);
        let partner = f64::from_bits(((high as u64) << 32) | u64::from(low));
        sum = add_rn(sum, partner);
        offset >>= 1;
    }
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

/// Compute a scale-free row-major BF16 linear layer with decoded FP64 dots.
///
/// One warp owns each output column. Each lane accumulates its strided K values in
/// FP64, then the warp reduces the 64-bit partials before a single FP32 and BF16
/// conversion. Reduction grouping differs from a scalar sequential FP64 dot, so
/// this is a high-accuracy reference kernel rather than a claim of universal bit
/// equality with every CPU summation order.
///
/// # Safety
/// Launch `grid = [ceil(n / 4), m, 1]`, `block = [128, 1, 1]`, with
/// `m in 1..=2048`, `n in 1..=32768`, and `k in 1..=32768`. `input` and `weight`
/// must cover row-major matrices of `m * k` and `n * k` readable BF16 values, all
/// finite. `output` and `unrounded` must cover `m * n` writable BF16 and FP32
/// values. Products and index arithmetic must fit the device address space. All
/// pointers must have their element alignment, be pairwise disjoint, and remain
/// live until kernel completion. The host must reject nonfinite or overflowing
/// outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bf16_linear_decode(
    input: *const u16,
    weight: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (lane, thread, tile_n, row) = coordinates();
    let column = tile_n as usize * 4 + (thread / WARP_SIZE) as usize;
    let row_valid = row < m;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let mut partial = 0.0_f64;

    if row_valid && column < n_usize {
        let input_start = row as usize * k_usize;
        let weight_start = column * k_usize;
        let mut index = lane as usize;
        while index < k_usize {
            // SAFETY: The launch contract supplies complete K rows for each valid coordinate.
            let (left, right) = unsafe {
                (
                    decode_bf16(*input.add(input_start + index)),
                    decode_bf16(*weight.add(weight_start + index)),
                )
            };
            partial = add_rn(partial, multiply_rn(left, right));
            index += WARP_SIZE as usize;
        }
    }

    let total = warp_reduce_f64(partial);
    if lane == 0 && row_valid && column < n_usize {
        let result = fp64_to_fp32_rn(total);
        let output_index = row as usize * n_usize + column;
        // SAFETY: Lane zero of this warp exclusively owns this in-range output.
        unsafe {
            unrounded.add(output_index).write(result);
            output.add(output_index).write(encode_bf16_rne(result));
        }
    }
}
