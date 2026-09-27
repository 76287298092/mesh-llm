use core::arch::asm;

const BLOCK_THREADS: u32 = 256;

#[inline(always)]
fn thread_and_row() -> (u32, u32) {
    let thread: u32;
    let row: u32;
    // SAFETY: Reads the calling thread's coordinates without changing memory.
    unsafe {
        asm!(
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {row}, %ctaid.x;",
            thread = out(reg32) thread,
            row = out(reg32) row,
            options(nomem, nostack),
        )
    };
    (thread, row)
}

#[inline(always)]
fn shared_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares one 256-element shared array for this CTA and returns its shared address.
    unsafe {
        asm!(
            ".shared .align 4 .b8 partials[1024];",
            "mov.u32 {base}, partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn store_shared_f32(base: u32, index: u32, value: f32) {
    // SAFETY: The caller uses indices 0..256 in this CTA's 1024-byte shared array.
    let address = base + index * 4;
    unsafe {
        asm!(
            "st.shared.f32 [{address}], {value};",
            address = in(reg32) address,
            value = in(reg32) value,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_f32(base: u32, index: u32) -> f32 {
    let value: f32;
    // SAFETY: The caller uses indices 0..256 in this CTA's 1024-byte shared array.
    let address = base + index * 4;
    unsafe {
        asm!(
            "ld.shared.f32 {value}, [{address}];",
            value = out(reg32) value,
            address = in(reg32) address,
            options(nostack),
        )
    };
    value
}

#[inline(always)]
fn block_barrier() {
    // SAFETY: All 256 threads in the CTA call each reduction barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn add_rn(left: f32, right: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
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
fn add_square(sum: f32, value: f32) -> f32 {
    let square: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f32 {square}, {value}, {value};",
            square = out(reg32) square,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    add_rn(sum, square)
}

#[inline(always)]
fn inverse_rms(sum_squares: f32, width: u32, epsilon: f32) -> f32 {
    let factor: f32;
    // SAFETY: All operands are scalar registers; width and epsilon satisfy the kernel contract.
    unsafe {
        asm!(
            "div.rn.f32 {mean}, {sum_squares}, {width};",
            "add.rn.f32 {denominator}, {mean}, {epsilon};",
            "sqrt.rn.f32 {root}, {denominator};",
            "div.rn.f32 {factor}, {one}, {root};",
            mean = out(reg32) _,
            denominator = out(reg32) _,
            root = out(reg32) _,
            factor = out(reg32) factor,
            sum_squares = in(reg32) sum_squares,
            width = in(reg32) (width as f32),
            epsilon = in(reg32) epsilon,
            one = in(reg32) 1.0_f32,
            options(nomem, nostack),
        )
    };
    factor
}

#[inline(always)]
fn multiply_rn(left: f32, right: f32) -> f32 {
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

/// Compute one row of RMSNorm and multiply by the learned weight vector.
///
/// # Safety
/// Launch a 1D grid with one 256-thread block per row (`block = [256, 1, 1]`;
/// `grid = [rows, 1, 1]`). `width` must be 1..=32768 and `epsilon` finite and
/// positive. `input` and `output` must be disjoint, 4-byte-aligned device pointers
/// to `rows * width` readable and writable `f32` values. `weight` must point to at
/// least `width` readable values and be disjoint from `output`. All pointers must
/// stay live until completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    width: u32,
    epsilon: f32,
) {
    let shared = shared_partials_base();
    let (thread, row) = thread_and_row();
    let width_usize = width as usize;
    let row_start = row as usize * width_usize;

    let mut partial = 0.0_f32;
    let mut column = thread;
    while column < width {
        // SAFETY: The launch contract supplies every row through `grid.x * width`.
        let value = unsafe { *input.add(row_start + column as usize) };
        partial = add_square(partial, value);
        column += BLOCK_THREADS;
    }
    store_shared_f32(shared, thread, partial);
    block_barrier();

    let mut stride = BLOCK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let left = load_shared_f32(shared, thread);
            let right = load_shared_f32(shared, thread + stride);
            store_shared_f32(shared, thread, add_rn(left, right));
        }
        block_barrier();
        stride /= 2;
    }

    let sum_squares = load_shared_f32(shared, 0);
    let factor = inverse_rms(sum_squares, width, epsilon);
    column = thread;
    while column < width {
        let index = row_start + column as usize;
        // SAFETY: The input and weight extents are covered by the launch contract.
        let (value, scale) = unsafe { (*input.add(index), *weight.add(column as usize)) };
        let normalized = multiply_rn(value, factor);
        let scaled = multiply_rn(normalized, scale);
        // SAFETY: Each thread writes distinct columns within this row's output extent.
        unsafe { output.add(index).write(scaled) };
        column += BLOCK_THREADS;
    }
}
