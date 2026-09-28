//! Experimental two-stage 32x32 NVFP4 CTA. See the pipeline knowledge entry.
use super::nvfp4_linear::{mma_nvfp4, store_scaled_output};
use core::arch::asm;
const STAGE_BYTES: u32 = 2304;

#[inline(always)]
fn coordinates() -> (u32, u32, u32, u32) {
    let (thread, tile_n, tile_m, base): (u32, u32, u32, u32);
    // SAFETY: CTA-local static storage and special registers.
    unsafe {
        asm!(
            ".shared .align 16 .b8 nvfp4_prefill_stages[4608];",
            "mov.u32 {base}, nvfp4_prefill_stages;",
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

/// One fixed producer path; every thread copies codes and the first warp scales.
#[inline(always)]
unsafe fn issue_matrix(
    matrix: MatrixStage,
    k: usize,
    tile_k: usize,
    thread: usize,
    code_base: u32,
    scale_base: u32,
) {
    // SAFETY: Threads 0..256 cover all 256 code words exactly once. K/2 and
    // K/16 strides and tile offsets are four-byte aligned because K % 64 == 0.
    unsafe {
        copy_row_word(
            matrix.codes,
            matrix.origin + thread / 8,
            matrix.rows,
            k / 2,
            tile_k * 32 + (thread % 8) * 4,
            code_base + thread as u32 * 4,
        );
        if thread < 32 {
            // These 32 words cover all scale rows, without runtime matrix indexing.
            copy_row_word(
                matrix.scales,
                matrix.origin + thread,
                matrix.rows,
                k / 16,
                tile_k * 4,
                scale_base + thread as u32 * 4,
            );
        }
    }
}

/// Caller supplies validated allocations and a stage with no remaining readers.
#[inline(always)]
unsafe fn issue_stage(inputs: StageInputs, tile_k: usize, thread: usize, base: u32) {
    // SAFETY: Named A/W regions are disjoint. Every thread copies one word from
    // each code matrix; the first warp additionally copies both scale words.
    unsafe {
        issue_matrix(inputs.a, inputs.k, tile_k, thread, base, base + 2048);
        issue_matrix(inputs.w, inputs.k, tile_k, thread, base + 1024, base + 2176);
    }
    // SAFETY: All threads reconverge and commit once, after their two/four copies.
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

#[inline(always)]
fn accumulate_stage(base: u32, thread: usize, accum: (f32, f32, f32, f32)) -> (f32, f32, f32, f32) {
    let warp = thread / 32;
    let lane = thread % 32;
    let group = lane / 4;
    let part = lane % 4;
    let row = (warp / 4) * 16 + group;
    let column = (warp % 4) * 8 + group;
    let a = base + (row * 32 + part * 4) as u32;
    let b = base + 1024 + (column * 32 + part * 4) as u32;
    let scale_a = match part {
        0 => shared_word(base + 2048 + row as u32 * 4),
        1 => shared_word(base + 2048 + (row + 8) as u32 * 4),
        _ => 0,
    };
    let scale_b = if part == 0 {
        shared_word(base + 2176 + column as u32 * 4)
    } else {
        0
    };
    mma_nvfp4(
        shared_word(a),
        shared_word(a + 256),
        shared_word(a + 16),
        shared_word(a + 272),
        shared_word(b),
        shared_word(b + 16),
        scale_a,
        scale_b,
        accum,
    )
}

/// Compute logical row-major packed NVFP4 A times W transpose.
///
/// # Safety
/// SM120a, grid `[ceil(n/32), ceil(m/32), 1]`, block `[256, 1, 1]`.
/// M in 1..=512; N a multiple of 8 in 8..=32768; K a multiple of 64
/// in 64..=32768. A/W cover M*K/2 and N*K/2 bytes; SA/SW cover M*K/16 and
/// N*K/16 bytes, all four-byte aligned. Codes are low-nibble-first E2M1 and
/// scales unsigned finite E4M3 codes 0..=126. Outputs cover M*N aligned u16/f32
/// items. All allocations are disjoint and live through completion. Factor is
/// positive finite; host acceptance rejects nonfinite results.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn nvfp4_prefill_tiled(
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
            origin: tile_n as usize * 32,
        },
        k: k as usize,
    };
    let tiles = k as usize / 64;
    let mut accum = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
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
        accum = accumulate_stage(current, thread, accum);
        barrier();
    }
    let warp = thread / 32;
    let lane = thread % 32;
    let row = inputs.a.origin + (warp / 4) * 16 + lane / 4;
    let column = inputs.w.origin + (warp % 4) * 8 + (lane % 4) * 2;
    for (dr, dc, value) in [
        (0, 0, accum.0),
        (0, 1, accum.1),
        (8, 0, accum.2),
        (8, 1, accum.3),
    ] {
        // SAFETY: Unique lane ownership and the helper's M/N guards cover tails.
        unsafe {
            store_scaled_output(
                out,
                raw,
                row + dr,
                column + dc,
                inputs.a.rows,
                inputs.w.rows,
                value,
                global_factor,
            )
        };
    }
}
