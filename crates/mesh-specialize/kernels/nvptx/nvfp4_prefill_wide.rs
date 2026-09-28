//! Experimental two-stage 32x128 NVFP4 CTA. See the pipeline knowledge entry.
use super::nvfp4_linear::{mma_nvfp4, store_scaled_output};
use core::arch::asm;
const STAGE_BYTES: u32 = 5760;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let (thread, tile_n, tile_m, base): (u32, u32, u32, u32);
    // SAFETY: CTA-local static storage and special registers.
    unsafe {
        asm!(
            ".shared .align 16 .b8 nvfp4_prefill_wide_stages[11520];",
            "mov.u32 {base}, nvfp4_prefill_wide_stages;",
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

/// Caller supplies valid allocations and a shared slot without remaining readers.
#[inline(always)]
unsafe fn issue_stage(inputs: StageInputs, tile: usize, thread: usize, base: u32) {
    let w = inputs.w;
    // SAFETY: Each fixed code region has 256 distinct producers. Named scale
    // regions have 32/128 distinct producers. All regions are disjoint.
    unsafe {
        issue_codes(inputs.a, inputs.k, tile, thread, base);
        issue_codes(w, inputs.k, tile, thread, base + 1024);
        issue_codes(
            MatrixStage {
                origin: w.origin + 32,
                ..w
            },
            inputs.k,
            tile,
            thread,
            base + 2048,
        );
        issue_codes(
            MatrixStage {
                origin: w.origin + 64,
                ..w
            },
            inputs.k,
            tile,
            thread,
            base + 3072,
        );
        issue_codes(
            MatrixStage {
                origin: w.origin + 96,
                ..w
            },
            inputs.k,
            tile,
            thread,
            base + 4096,
        );
        if thread < 32 {
            copy_row_word(
                inputs.a.scales,
                inputs.a.origin + thread,
                inputs.a.rows,
                inputs.k / 16,
                tile * 4,
                base + 5120 + thread as u32 * 4,
            );
        }
        if thread < 128 {
            copy_row_word(
                w.scales,
                w.origin + thread,
                w.rows,
                inputs.k / 16,
                tile * 4,
                base + 5248 + thread as u32 * 4,
            );
        }
    }
    // SAFETY: Every thread reconverges and commits its five to seven copies once.
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
fn load_a(base: u32, thread: usize) -> AFragment {
    let row = (thread / 128) * 16 + (thread % 32) / 4;
    let part = thread % 4;
    let address = base + (row * 32 + part * 4) as u32;
    let scale = match part {
        0 => shared_word(base + 5120 + row as u32 * 4),
        1 => shared_word(base + 5120 + (row + 8) as u32 * 4),
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

#[inline(always)]
fn accumulate_b(
    base: u32,
    thread: usize,
    column_offset: usize,
    a: AFragment,
    accum: Accumulator,
) -> Accumulator {
    let column = ((thread / 32) % 4) * 32 + column_offset + (thread % 32) / 4;
    let part = thread % 4;
    let address = base + 1024 + (column * 32 + part * 4) as u32;
    let scale = if part == 0 {
        shared_word(base + 5248 + column as u32 * 4)
    } else {
        0
    };
    mma_nvfp4(
        a.a0,
        a.a1,
        a.a2,
        a.a3,
        shared_word(address),
        shared_word(address + 16),
        a.scale,
        scale,
        accum,
    )
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

/// Compute logical row-major packed NVFP4 A times W transpose.
///
/// # Safety
/// SM120a, grid `[ceil(n/128), ceil(m/32), 1]`, block `[256, 1, 1]`.
/// M in 1..=512; N a multiple of 8 in 8..=32768; K a multiple of 64
/// in 64..=32768. A/W cover M*K/2 and N*K/2 bytes; SA/SW cover M*K/16 and
/// N*K/16 bytes, all four-byte aligned. Codes are low-nibble-first E2M1 and
/// scales unsigned finite E4M3 codes 0..=126. Outputs cover M*N aligned u16/f32
/// items. All allocations are disjoint and live through completion. Factor is
/// positive finite; host acceptance rejects nonfinite results.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_prefill_wide(
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
            origin: tile_m as usize * 32,
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
    let mut accum0 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut accum1 = accum0;
    let mut accum2 = accum0;
    let mut accum3 = accum0;
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
        let a_fragment = load_a(current, thread);
        accum0 = accumulate_b(current, thread, 0, a_fragment, accum0);
        accum1 = accumulate_b(current, thread, 8, a_fragment, accum1);
        accum2 = accumulate_b(current, thread, 16, a_fragment, accum2);
        accum3 = accumulate_b(current, thread, 24, a_fragment, accum3);
        barrier();
    }
    let warp = thread / 32;
    let lane = thread % 32;
    let row = inputs.a.origin + (warp / 4) * 16 + lane / 4;
    let column = inputs.w.origin + (warp % 4) * 32 + (lane % 4) * 2;
    // SAFETY: Each warp owns four disjoint 16x8 tiles; each lane stores its
    // original four coordinates in each tile, with guards for M/N tails.
    unsafe {
        store_fragment(
            out,
            raw,
            row,
            column,
            inputs.a.rows,
            inputs.w.rows,
            accum0,
            global_factor,
        );
        store_fragment(
            out,
            raw,
            row,
            column + 8,
            inputs.a.rows,
            inputs.w.rows,
            accum1,
            global_factor,
        );
        store_fragment(
            out,
            raw,
            row,
            column + 16,
            inputs.a.rows,
            inputs.w.rows,
            accum2,
            global_factor,
        );
        store_fragment(
            out,
            raw,
            row,
            column + 24,
            inputs.a.rows,
            inputs.w.rows,
            accum3,
            global_factor,
        );
    }
}
