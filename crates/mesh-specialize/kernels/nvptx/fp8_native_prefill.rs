use core::arch::asm;

const TILE_M: usize = 32;
const TILE_N: usize = 64;
const TILE_K: usize = 64;
const WARP_M: usize = 16;
const WARP_N: usize = 32;
const WARPS_N: usize = 2;
const THREADS: usize = 128;
const A_BYTES: usize = TILE_M * TILE_K;
const SHARED_BYTES: usize = A_BYTES + TILE_N * TILE_K;
const VECTORS_PER_ROW: usize = TILE_K / 16;
const A_VECTOR_COUNT: usize = TILE_M * VECTORS_PER_ROW;
const W_VECTOR_COUNT: usize = TILE_N * VECTORS_PER_ROW;
const TOTAL_VECTOR_COUNT: usize = A_VECTOR_COUNT + W_VECTOR_COUNT;

#[inline(always)]
fn cta_coordinates() -> (u32, u32, u32, u32, u32) {
    let lane: u32;
    let thread: u32;
    let tile_n: u32;
    let tile_m: u32;
    let shared_base: u32;
    // SAFETY: Reads CTA coordinates and declares the kernel's statically sized shared tile.
    unsafe {
        asm!(
            ".shared .align 16 .b8 fp8_native_prefill_tile[6144];",
            "mov.u32 {shared_base}, fp8_native_prefill_tile;",
            "mov.u32 {lane}, %laneid;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {tile_m}, %ctaid.y;",
            shared_base = out(reg32) shared_base,
            lane = out(reg32) lane,
            thread = out(reg32) thread,
            tile_n = out(reg32) tile_n,
            tile_m = out(reg32) tile_m,
            options(nostack),
        )
    };
    (lane, thread, tile_n, tile_m, shared_base)
}

#[inline(always)]
fn cta_barrier() {
    // SAFETY: Every thread in the CTA reaches each barrier in the same order.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

/// Copy one aligned 16-byte input vector into the shared tile asynchronously.
///
/// # Safety
/// `destination` must be a 16-byte-aligned address inside this kernel's shared tile.
/// `source` must address 16 readable bytes in a 16-byte-aligned global allocation.
#[inline(always)]
unsafe fn copy_16_to_shared(destination: u32, source: *const u8) {
    // SAFETY: The caller proves global-source and shared-destination alignment and extent.
    unsafe {
        asm!(
            "cvta.to.global.u64 {global_source}, {source};",
            "cp.async.ca.shared.global [{destination}], [{global_source}], 16;",
            global_source = out(reg64) _,
            source = in(reg64) source as u64,
            destination = in(reg32) destination,
            options(nostack),
        )
    };
}

/// Zero one aligned 16-byte shared vector for an M, N, or K tail.
///
/// # Safety
/// `destination` must be a 16-byte-aligned address inside this kernel's shared tile.
#[inline(always)]
unsafe fn zero_shared_vector(destination: u32) {
    // SAFETY: The caller proves the full vector lies inside the declared shared tile.
    unsafe {
        asm!(
            "st.shared.v4.b32 [{destination}], {{{zero}, {zero}, {zero}, {zero}}};",
            destination = in(reg32) destination,
            zero = in(reg32) 0_u32,
            options(nostack),
        )
    };
}

#[inline(always)]
fn store_shared_byte(destination: u32, value: u8) {
    let value = u32::from(value);
    // SAFETY: The caller assigns every byte a unique in-range shared address.
    unsafe {
        asm!(
            "st.shared.u8 [{destination}], {value};",
            destination = in(reg32) destination,
            value = in(reg32) value,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_byte(source: u32) -> u32 {
    let value: u32;
    // SAFETY: The caller reads one byte from the fully staged shared tile.
    unsafe {
        asm!(
            "ld.shared.u8 {value}, [{source}];",
            value = out(reg32) value,
            source = in(reg32) source,
            options(nostack),
        )
    };
    value
}

/// Stage a 32x64 A tile and 64x64 transposed-weight tile through aligned copies.
///
/// # Safety
/// For each in-range row, `a` and `w` must cover their declared row-major matrices.
/// The shared tile must have been declared by `cta_coordinates`; all index products
/// must fit `usize`. `k` must be divisible by 16 and both base pointers 16-byte aligned.
#[inline(always)]
unsafe fn stage_vector_tile(
    a: *const u8,
    w: *const u8,
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    thread: usize,
    shared_base: u32,
) {
    let mut vector = thread;
    while vector < TOTAL_VECTOR_COUNT {
        if vector < A_VECTOR_COUNT {
            let row = vector / VECTORS_PER_ROW;
            let chunk = vector % VECTORS_PER_ROW;
            let global_row = tile_m * TILE_M + row;
            let global_k = tile_k + chunk * 16;
            let shared_offset = row * TILE_K + chunk * 16;
            let destination = shared_base + shared_offset as u32;
            if global_row < m && global_k + 16 <= k {
                let source_offset = global_row * k + global_k;
                // SAFETY: This full vector lies within the valid A row and aligned K range.
                unsafe { copy_16_to_shared(destination, a.add(source_offset)) };
            } else {
                // SAFETY: The destination belongs to this vector's unique shared tile slot.
                unsafe { zero_shared_vector(destination) };
            }
        } else {
            let weight_vector = vector - A_VECTOR_COUNT;
            let row = weight_vector / VECTORS_PER_ROW;
            let chunk = weight_vector % VECTORS_PER_ROW;
            let global_row = tile_n * TILE_N + row;
            let global_k = tile_k + chunk * 16;
            let shared_offset = A_BYTES + row * TILE_K + chunk * 16;
            let destination = shared_base + shared_offset as u32;
            if global_row < n && global_k + 16 <= k {
                let source_offset = global_row * k + global_k;
                // SAFETY: This full vector lies within the valid weight row and aligned K range.
                unsafe { copy_16_to_shared(destination, w.add(source_offset)) };
            } else {
                // SAFETY: The destination belongs to this vector's unique shared tile slot.
                unsafe { zero_shared_vector(destination) };
            }
        }
        vector += THREADS;
    }

    // SAFETY: Every CTA thread commits and waits for its own copies before the CTA barrier.
    unsafe {
        asm!(
            "cp.async.commit_group;",
            "cp.async.wait_group 0;",
            options(nostack),
        )
    };
    cta_barrier();
}

/// Stage the same tile with scalar global reads when 16-byte copies are inadmissible.
///
/// # Safety
/// For each in-range row, `a` and `w` must cover their declared row-major matrices.
/// The shared tile must have been declared by `cta_coordinates`; all index products
/// must fit `usize`.
#[inline(always)]
unsafe fn stage_scalar_tile(
    a: *const u8,
    w: *const u8,
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    thread: usize,
    shared_base: u32,
) {
    let mut index = thread;
    while index < SHARED_BYTES {
        let value = if index < A_BYTES {
            let row = index / TILE_K;
            let column = index % TILE_K;
            let global_row = tile_m * TILE_M + row;
            let global_k = tile_k + column;
            if global_row < m && global_k < k {
                let source_offset = global_row * k + global_k;
                // SAFETY: The guarded logical A coordinate is inside the row-major allocation.
                unsafe { *a.add(source_offset) }
            } else {
                0
            }
        } else {
            let weight_index = index - A_BYTES;
            let row = weight_index / TILE_K;
            let column = weight_index % TILE_K;
            let global_row = tile_n * TILE_N + row;
            let global_k = tile_k + column;
            if global_row < n && global_k < k {
                let source_offset = global_row * k + global_k;
                // SAFETY: The guarded logical weight coordinate is inside its row-major allocation.
                unsafe { *w.add(source_offset) }
            } else {
                0
            }
        };
        store_shared_byte(shared_base + index as u32, value);
        index += THREADS;
    }
    cta_barrier();
}

/// Stage one logical K64 tile, selecting the vector path uniformly for the CTA.
///
/// # Safety
/// The pointers and dimensions must satisfy either `stage_vector_tile` or
/// `stage_scalar_tile`'s contract.
#[inline(always)]
unsafe fn stage_tile(
    a: *const u8,
    w: *const u8,
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    thread: usize,
    shared_base: u32,
    vector_copies_admitted: bool,
) {
    if vector_copies_admitted {
        // SAFETY: Admission checks prove aligned bases and an aligned row stride.
        unsafe { stage_vector_tile(a, w, m, n, k, tile_m, tile_n, tile_k, thread, shared_base) };
    } else {
        // SAFETY: Scalar staging uses byte-addressed loads and stores for all tails.
        unsafe { stage_scalar_tile(a, w, m, n, k, tile_m, tile_n, tile_k, thread, shared_base) };
    }
}

#[inline(always)]
fn load_a_fragment(shared_base: u32, warp_m: usize, lane: usize, k_part: usize) -> [u32; 4] {
    let group = lane >> 2;
    let thread_in_group = lane & 3;
    let mut registers = [0_u32; 4];
    let mut element = 0_usize;
    while element < 16 {
        let row = warp_m * WARP_M
            + group
            + if element < 4 || (8..12).contains(&element) {
                0
            } else {
                8
            };
        let column =
            k_part * 32 + thread_in_group * 4 + (element & 3) + if element >= 8 { 16 } else { 0 };
        let address = shared_base + (row * TILE_K + column) as u32;
        let code = load_shared_byte(address);
        registers[element / 4] |= code << ((element & 3) * 8);
        element += 1;
    }
    registers
}

#[inline(always)]
fn load_b_fragment(
    shared_base: u32,
    warp_n: usize,
    fragment_n: usize,
    lane: usize,
    k_part: usize,
) -> [u32; 2] {
    let group = lane >> 2;
    let thread_in_group = lane & 3;
    let mut registers = [0_u32; 2];
    let mut element = 0_usize;
    while element < 8 {
        let row_k =
            k_part * 32 + thread_in_group * 4 + (element & 3) + if element >= 4 { 16 } else { 0 };
        let column_n = warp_n * WARP_N + fragment_n * 8 + group;
        let address = shared_base + (A_BYTES + column_n * TILE_K + row_k) as u32;
        let code = load_shared_byte(address);
        registers[element / 4] |= code << ((element & 3) * 8);
        element += 1;
    }
    registers
}

#[inline(always)]
fn mma_e4m3(a: [u32; 4], b: [u32; 2], accumulators: [f32; 4]) -> [f32; 4] {
    let (mut d0, mut d1, mut d2, mut d3) = (
        accumulators[0],
        accumulators[1],
        accumulators[2],
        accumulators[3],
    );
    // SAFETY: All lanes execute the same documented m16n8k32 FP8 MMA fragment.
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 ",
            "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
            "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d0,
            d1 = inout(reg32) d1,
            d2 = inout(reg32) d2,
            d3 = inout(reg32) d3,
            a0 = in(reg32) a[0],
            a1 = in(reg32) a[1],
            a2 = in(reg32) a[2],
            a3 = in(reg32) a[3],
            b0 = in(reg32) b[0],
            b1 = in(reg32) b[1],
            options(nomem, nostack),
        );
    }
    [d0, d1, d2, d3]
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
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

#[inline(always)]
fn fp32_multiply_rn(left: f32, right: f32) -> f32 {
    let product: f32;
    // SAFETY: This scalar FP32 multiply has no memory or stack effects.
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
fn fp32_add_rn(left: f32, right: f32) -> f32 {
    let sum: f32;
    // SAFETY: Scalar FP32 addition has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {sum}, {left}, {right};",
            sum = out(reg32) sum,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    sum
}

/// Store one output using the existing row-scale then channel-scale epilogue order.
///
/// # Safety
/// `row` and `column` must be in range. The scale pointers must cover `m` and `n`
/// entries, and the output pointers must each cover `m * n` entries.
#[inline(always)]
unsafe fn store_output(
    row_scales: *const f32,
    weight_scales: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    row: usize,
    column: usize,
    m: usize,
    n: usize,
    accumulator: f32,
) {
    if row >= m || column >= n {
        return;
    }
    // SAFETY: The coordinate guards establish valid row and channel scale indices.
    let (row_scale, weight_scale) = unsafe {
        (
            *row_scales.add(row),
            decode_bf16(*weight_scales.add(column)),
        )
    };
    let value = fp32_multiply_rn(fp32_multiply_rn(accumulator, row_scale), weight_scale);
    let index = row * n + column;
    // SAFETY: Each in-range coordinate has a unique writer in its owning warp tile.
    unsafe {
        unrounded.add(index).write(value);
        out.add(index).write(encode_bf16_rne(value));
    }
}

/// Compute row-major E4M3 projection tiles with SM120 native FP8 tensor-core MMA.
///
/// The experimental arithmetic profile is `f32` tensor-core accumulation followed
/// by `(accumulator * row_scale) * BF16(weight_scale)`. It is intentionally distinct
/// from the exact signed-integer kernel's arithmetic profile.
///
/// # Safety
/// Launch `grid = [ceil(n / 64), ceil(m / 32), 1]` and `block = [128, 1, 1]`.
/// Require `m in 1..=2048`, `n in 1..=262144`, and `k in 1..=32768`. `a` must
/// cover row-major `[m, k]` E4M3FN codes and `w` row-major `[n, k]` codes; neither
/// may contain NaN codes `0x7f` or `0xff`. `row_scales` must cover `m` finite
/// positive FP32 values, and `weight_scales` must cover `n` finite positive BF16
/// values. `out` and `unrounded` must cover `m*n` BF16 and FP32 values. All
/// dimensions, extents, and pointer arithmetic must fit the device address space.
/// Pointers must be correctly aligned for their element types, mutually disjoint,
/// and live through completion. The vector staging path additionally requires
/// 16-byte-aligned A/W pointers and `k % 16 == 0`; other inputs use scalar staging.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_prefill_native(
    a: *const u8,
    w: *const u8,
    row_scales: *const f32,
    weight_scales: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    // SAFETY: Both exported entries use the documented identical pointer/launch contract.
    unsafe { project::<false>(a, w, row_scales, weight_scales, out, unrounded, m, n, k) };
}

/// Experimental K64 partial sums, combined with explicit FP32 round-to-nearest adds.
///
/// # Safety
/// The complete pointer, dimensions, aliasing and launch contract of
/// `fp8_prefill_native` applies unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fp8_prefill_native_short(
    a: *const u8,
    w: *const u8,
    row_scales: *const f32,
    weight_scales: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    // SAFETY: Both exported entries use the documented identical pointer/launch contract.
    unsafe { project::<true>(a, w, row_scales, weight_scales, out, unrounded, m, n, k) };
}

#[inline(always)]
unsafe fn project<const SHORT: bool>(
    a: *const u8,
    w: *const u8,
    row_scales: *const f32,
    weight_scales: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (lane, thread, tile_n, tile_m, shared_base) = cta_coordinates();
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let tile_m = tile_m as usize;
    let tile_n = tile_n as usize;
    let thread = thread as usize;
    let lane = lane as usize;
    let vector_copies_admitted =
        k.is_multiple_of(16) && (a as usize).is_multiple_of(16) && (w as usize).is_multiple_of(16);
    let warp = thread / 32;
    let warp_m = warp / WARPS_N;
    let warp_n = warp % WARPS_N;
    let group = lane >> 2;
    let thread_in_group = lane & 3;
    let output_row_base = tile_m * TILE_M + warp_m * WARP_M;
    let output_column_base = tile_n * TILE_N + warp_n * WARP_N;
    let mut accumulators = [[0.0_f32; 4]; 4];

    for tile_k in (0..k).step_by(TILE_K) {
        // SAFETY: The kernel contract provides complete input matrices and scalar fallback
        // covers any alignment or K-tail shape that cannot use 16-byte copies.
        unsafe {
            stage_tile(
                a,
                w,
                m,
                n,
                k,
                tile_m,
                tile_n,
                tile_k,
                thread,
                shared_base,
                vector_copies_admitted,
            )
        };

        let mut partial = if SHORT {
            [[0.0_f32; 4]; 4]
        } else {
            accumulators
        };
        let mut k_part = 0_usize;
        while k_part < 2 {
            let a_fragment = load_a_fragment(shared_base, warp_m, lane, k_part);
            let mut fragment_n = 0_usize;
            while fragment_n < 4 {
                let b_fragment = load_b_fragment(shared_base, warp_n, fragment_n, lane, k_part);
                partial[fragment_n] = mma_e4m3(a_fragment, b_fragment, partial[fragment_n]);
                fragment_n += 1;
            }
            k_part += 1;
        }

        if SHORT {
            for fragment in 0..4 {
                for element in 0..4 {
                    accumulators[fragment][element] =
                        fp32_add_rn(accumulators[fragment][element], partial[fragment][element]);
                }
            }
        } else {
            accumulators = partial;
        }

        // All warps must finish shared fragment reads before the next K tile reuses it.
        cta_barrier();
    }

    let mut fragment_n = 0_usize;
    while fragment_n < 4 {
        let accumulator = accumulators[fragment_n];
        let mut element = 0_usize;
        while element < 4 {
            let row = output_row_base + group + if element >= 2 { 8 } else { 0 };
            let column = output_column_base + fragment_n * 8 + thread_in_group * 2 + (element & 1);
            // SAFETY: The helper guards M/N tails before reading scales or storing.
            unsafe {
                store_output(
                    row_scales,
                    weight_scales,
                    out,
                    unrounded,
                    row,
                    column,
                    m,
                    n,
                    accumulator[element],
                )
            };
            element += 1;
        }
        fragment_n += 1;
    }
}
