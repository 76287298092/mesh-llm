//! Experimental two-stage 128x128 NVFP4 CTA. See the pipeline knowledge entry.
use super::nvfp4_linear::{mma_nvfp4, store_scaled_output};
use core::arch::asm;
const STAGE_BYTES: u32 = 9216;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let (thread, tile_n, tile_m, base): (u32, u32, u32, u32);
    // SAFETY: CTA-local static storage and special registers.
    unsafe {
        asm!(
            ".shared .align 16 .b8 nvfp4_prefill_large_stages[18432];",
            "mov.u32 {base}, nvfp4_prefill_large_stages;",
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile_n}, %ctaid.x;",
            "mov.u32 {tile_m}, %ctaid.y;",
            base = out(reg32) base, thread = out(reg32) thread,
            tile_n = out(reg32) tile_n, tile_m = out(reg32) tile_m,
            options(nostack),
        );
    }
    (thread, tile_n, tile_m, base)
}

#[inline(always)]
fn barrier() {
    // SAFETY: All 256 threads reach each barrier, including output-tail threads.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn await_stage() {
    // SAFETY: Each thread waits for its own committed copies before CTA publication.
    unsafe { asm!("cp.async.wait_group 0;", options(nostack)) };
    barrier();
}

/// Both addresses must be four-byte aligned. Destination is a unique shared word;
/// source is live global storage, readable for four bytes when valid.
#[inline(always)]
unsafe fn copy_word(destination: u32, source: *const u8, valid: bool) {
    let size = if valid { 4_u32 } else { 0 };
    // SAFETY: Zero source size performs zero fill. Invalid rows use the valid
    // allocation base, never an out-of-bounds pointer.
    unsafe {
        asm!(
            "cvta.to.global.u64 {global}, {source};",
            "cp.async.ca.shared.global [{destination}], [{global}], 4, {size};",
            global = out(reg64) _, source = in(reg64) source as u64,
            destination = in(reg32) destination, size = in(reg32) size,
            options(nostack),
        );
    }
}

#[derive(Clone, Copy)]
struct MatrixStage {
    codes: *const u8,
    scales: *const u8,
    rows: usize,
    origin: usize,
}

#[derive(Clone, Copy)]
struct StageInputs {
    a: MatrixStage,
    w: MatrixStage,
    k: usize,
}

/// Copy an aligned row word, using the live base for zero-filled tail rows.
#[inline(always)]
unsafe fn copy_row_word(
    source_base: *const u8,
    row: usize,
    rows: usize,
    stride: usize,
    offset: usize,
    destination: u32,
) {
    let valid = row < rows;
    let source = if valid {
        // SAFETY: Caller proves a full aligned word in the logical row. The
        // invalid-row branch never forms an out-of-bounds source pointer.
        unsafe { source_base.add(row * stride + offset) }
    } else {
        source_base
    };
    // SAFETY: The caller owns this shared word and proves source alignment.
    unsafe { copy_word(destination, source, valid) };
}

/// Stage one 32-row code segment. Threads own distinct four-byte words.
#[inline(always)]
unsafe fn issue_codes(matrix: MatrixStage, k: usize, tile: usize, thread: usize, base: u32) {
    // SAFETY: K and every row stride are aligned; tile < K/64, thread < 256.
    unsafe {
        copy_row_word(
            matrix.codes,
            matrix.origin + thread / 8,
            matrix.rows,
            k / 2,
            tile * 32 + (thread % 8) * 4,
            base + thread as u32 * 4,
        );
    }
}

/// Stage all 128 rows of a matrix through four fixed producer segments.
#[inline(always)]
unsafe fn issue_matrix(matrix: MatrixStage, k: usize, tile: usize, thread: usize, base: u32) {
    // SAFETY: Each segment covers 1024 disjoint bytes with one word per thread.
    unsafe {
        issue_codes(matrix, k, tile, thread, base);
        issue_codes(
            MatrixStage {
                origin: matrix.origin + 32,
                ..matrix
            },
            k,
            tile,
            thread,
            base + 1024,
        );
        issue_codes(
            MatrixStage {
                origin: matrix.origin + 64,
                ..matrix
            },
            k,
            tile,
            thread,
            base + 2048,
        );
        issue_codes(
            MatrixStage {
                origin: matrix.origin + 96,
                ..matrix
            },
            k,
            tile,
            thread,
            base + 3072,
        );
    }
}

/// Caller supplies valid allocations and a shared slot without remaining readers.
#[inline(always)]
unsafe fn issue_stage(inputs: StageInputs, tile: usize, thread: usize, base: u32) {
    // SAFETY: A/W codes and scales occupy four disjoint regions. Every code
    // word has one producer; threads 0..128 cover every scale word exactly once.
    unsafe {
        issue_matrix(inputs.a, inputs.k, tile, thread, base);
        issue_matrix(inputs.w, inputs.k, tile, thread, base + 4096);
        if thread < 128 {
            copy_row_word(
                inputs.a.scales,
                inputs.a.origin + thread,
                inputs.a.rows,
                inputs.k / 16,
                tile * 4,
                base + 8192 + thread as u32 * 4,
            );
            copy_row_word(
                inputs.w.scales,
                inputs.w.origin + thread,
                inputs.w.rows,
                inputs.k / 16,
                tile * 4,
                base + 8704 + thread as u32 * 4,
            );
        }
    }
    // SAFETY: Every thread reconverges and commits its eight or ten copies once.
    unsafe { asm!("cp.async.commit_group;", options(nostack)) };
}

#[inline(always)]
fn shared_word(address: u32) -> u32 {
    let value: u32;
    // SAFETY: Called on aligned words in the current, waited/published stage.
    unsafe {
        asm!("ld.shared.b32 {value}, [{address}];",
            value = out(reg32) value, address = in(reg32) address, options(nostack));
    }
    value
}

#[derive(Clone, Copy)]
struct AFragment {
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    scale: u32,
}

type Accumulator = (f32, f32, f32, f32);

#[inline(always)]
fn load_a(base: u32, thread: usize, row_offset: usize) -> AFragment {
    let row = (thread / 128) * 16 + row_offset + (thread % 32) / 4;
    let part = thread % 4;
    let address = base + (row * 32 + part * 4) as u32;
    let scale = match part {
        0 => shared_word(base + 8192 + row as u32 * 4),
        1 => shared_word(base + 8192 + (row + 8) as u32 * 4),
        _ => 0,
    };
    AFragment {
        a0: shared_word(address),
        a1: shared_word(address + 256),
        a2: shared_word(address + 16),
        a3: shared_word(address + 272),
        scale,
    }
}

#[derive(Clone, Copy)]
struct BFragment {
    b0: u32,
    b1: u32,
    scale: u32,
}

#[inline(always)]
fn load_b(base: u32, thread: usize, column_offset: usize) -> BFragment {
    let column = ((thread / 32) % 4) * 32 + column_offset + (thread % 32) / 4;
    let part = thread % 4;
    let address = base + 4096 + (column * 32 + part * 4) as u32;
    let scale = if part == 0 {
        shared_word(base + 8704 + column as u32 * 4)
    } else {
        0
    };
    BFragment {
        b0: shared_word(address),
        b1: shared_word(address + 16),
        scale,
    }
}

#[inline(always)]
fn accumulate(a: AFragment, b: BFragment, accum: Accumulator) -> Accumulator {
    mma_nvfp4(a.a0, a.a1, a.a2, a.a3, b.b0, b.b1, a.scale, b.scale, accum)
}

/// Caller supplies unique lane coordinates and disjoint full output allocations.
#[inline(always)]
unsafe fn store_fragment(
    out: *mut u16,
    raw: *mut f32,
    row: usize,
    column: usize,
    m: usize,
    n: usize,
    accum: Accumulator,
    factor: f32,
) {
    // SAFETY: Four distinct lane coordinates, with M/N guards in the existing helper.
    unsafe {
        store_scaled_output(out, raw, row, column, m, n, accum.0, factor);
        store_scaled_output(out, raw, row, column + 1, m, n, accum.1, factor);
        store_scaled_output(out, raw, row + 8, column, m, n, accum.2, factor);
        store_scaled_output(out, raw, row + 8, column + 1, m, n, accum.3, factor);
    }
}

// Expand fixed column fragments to explicit stores, with no runtime iteration.
macro_rules! store_four_columns {
    ($out:expr, $raw:expr, $row:expr, $column:expr, $m:expr, $n:expr, $factor:expr;
        $c0:expr, $c1:expr, $c2:expr, $c3:expr) => {{
        store_fragment($out, $raw, $row, $column, $m, $n, $c0, $factor);
        store_fragment($out, $raw, $row, $column + 8, $m, $n, $c1, $factor);
        store_fragment($out, $raw, $row, $column + 16, $m, $n, $c2, $factor);
        store_fragment($out, $raw, $row, $column + 24, $m, $n, $c3, $factor);
    }};
}

/// Compute logical row-major packed NVFP4 A times W transpose.
///
/// # Safety
/// SM120a, grid `[ceil(n/128), ceil(m/128), 1]`, block `[256, 1, 1]`.
/// M in 1..=512; N a multiple of 8 in 8..=32768; K a multiple of 64
/// in 64..=32768. A/W cover M*K/2 and N*K/2 bytes; SA/SW cover M*K/16 and
/// N*K/16 bytes, all four-byte aligned. Codes are low-nibble-first E2M1 and
/// scales unsigned finite E4M3 codes 0..=126. Outputs cover M*N aligned u16/f32
/// items. All allocations are disjoint and live through completion. Factor is
/// positive finite; host acceptance rejects nonfinite results.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_prefill_large(
    a: *const u8,
    w: *const u8,
    sa: *const u8,
    sw: *const u8,
    out: *mut u16,
    raw: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    global_factor: f32,
) {
    let (thread, tile_n, tile_m, base) = coordinates();
    let thread = thread as usize;
    let inputs = StageInputs {
        a: MatrixStage {
            codes: a,
            scales: sa,
            rows: m as usize,
            origin: tile_m as usize * 128,
        },
        w: MatrixStage {
            codes: w,
            scales: sw,
            rows: n as usize,
            origin: tile_n as usize * 128,
        },
        k: k as usize,
    };
    let tiles = k as usize / 64;
    let mut c00 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c01 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c02 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c03 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c10 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c11 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c12 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c13 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c20 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c21 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c22 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c23 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c30 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c31 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c32 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut c33 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    // SAFETY: Stage zero has no readers and inputs satisfy the launch contract.
    unsafe { issue_stage(inputs, 0, thread, base) };
    for tile in 0..tiles {
        await_stage();
        let current = base + (tile % 2) as u32 * STAGE_BYTES;
        if tile + 1 < tiles {
            let next = base + ((tile + 1) % 2) as u32 * STAGE_BYTES;
            // SAFETY: The previous terminal barrier retired all readers of this
            // slot. Copies run concurrently with current-slot loads and MMA.
            unsafe { issue_stage(inputs, tile + 1, thread, next) };
        }
        let a0 = load_a(current, thread, 0);
        let a1 = load_a(current, thread, 32);
        let a2 = load_a(current, thread, 64);
        let a3 = load_a(current, thread, 96);
        let b0 = load_b(current, thread, 0);
        let b1 = load_b(current, thread, 8);
        let b2 = load_b(current, thread, 16);
        let b3 = load_b(current, thread, 24);
        c00 = accumulate(a0, b0, c00);
        c01 = accumulate(a0, b1, c01);
        c02 = accumulate(a0, b2, c02);
        c03 = accumulate(a0, b3, c03);
        c10 = accumulate(a1, b0, c10);
        c11 = accumulate(a1, b1, c11);
        c12 = accumulate(a1, b2, c12);
        c13 = accumulate(a1, b3, c13);
        c20 = accumulate(a2, b0, c20);
        c21 = accumulate(a2, b1, c21);
        c22 = accumulate(a2, b2, c22);
        c23 = accumulate(a2, b3, c23);
        c30 = accumulate(a3, b0, c30);
        c31 = accumulate(a3, b1, c31);
        c32 = accumulate(a3, b2, c32);
        c33 = accumulate(a3, b3, c33);
        barrier();
    }
    let warp = thread / 32;
    let lane = thread % 32;
    let row = inputs.a.origin + (warp / 4) * 16 + lane / 4;
    let column = inputs.w.origin + (warp % 4) * 32 + (lane % 4) * 2;
    // SAFETY: Fixed M/N fragment offsets partition the 128x128 CTA outputs.
    // Each fragment preserves baseline lane coordinates and guarded stores.
    unsafe {
        store_four_columns!(out, raw, row, column, inputs.a.rows, inputs.w.rows, global_factor; c00, c01, c02, c03);
        store_four_columns!(out, raw, row + 32, column, inputs.a.rows, inputs.w.rows, global_factor; c10, c11, c12, c13);
        store_four_columns!(out, raw, row + 64, column, inputs.a.rows, inputs.w.rows, global_factor; c20, c21, c22, c23);
        store_four_columns!(out, raw, row + 96, column, inputs.a.rows, inputs.w.rows, global_factor; c30, c31, c32, c33);
    }
}
