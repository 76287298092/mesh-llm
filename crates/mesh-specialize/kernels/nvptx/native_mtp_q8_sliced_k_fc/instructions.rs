// SPDX-License-Identifier: Apache-2.0
// NInfer contributors, pin e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d.
// Modified: Rust PTX wrappers, exact signed-code conversion via FP32 registers.
use core::arch::asm;

#[inline(always)]
pub(super) fn coordinates() -> (u32, u32) {
    let thread: u32;
    let block: u32;
    // SAFETY: Register-only reads of CTA/thread identifiers.
    unsafe {
        asm!("mov.u32 {thread}, %tid.x;", "mov.u32 {block}, %ctaid.x;",
        thread = out(reg32) thread, block = out(reg32) block, options(nomem, nostack))
    };
    (thread, block)
}

#[inline(always)]
pub(super) fn barrier() {
    // SAFETY: All 256 threads reach every CTA barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
pub(super) fn commit() {
    // SAFETY: Commits the issuing thread's valid asynchronous copies.
    unsafe { asm!("cp.async.commit_group;", options(nostack)) };
}

#[inline(always)]
pub(super) fn wait() {
    // SAFETY: Waits for all of the issuing thread's committed copies.
    unsafe { asm!("cp.async.wait_group 0;", options(nostack)) };
}

#[inline(always)]
pub(super) fn load_code(address: u32) -> u32 {
    let value: u32;
    // SAFETY: Aligned, staged code pair inside the current CTA's tile.
    unsafe {
        asm!("ld.shared.u16 {value}, [{address}];", value = out(reg32) value,
        address = in(reg32) address, options(nostack))
    };
    value
}

#[inline(always)]
pub(super) fn load_scale(address: u32) -> u32 {
    let value: u32;
    // SAFETY: Aligned staged FP16 scale pair, after the CTA copy barrier.
    unsafe {
        asm!("ld.shared.u32 {value}, [{address}];", value = out(reg32) value,
        address = in(reg32) address, options(nostack))
    };
    value
}

#[inline(always)]
pub(super) unsafe fn global_scale(address: *const u16) -> u32 {
    let value: u32;
    // SAFETY: Caller supplies an aligned readable FP16 pair in the global plane.
    unsafe {
        asm!("{{ .reg .b64 global; cvta.to.global.u64 global, {address};",
        "ld.global.nc.u32 {value}, [global]; }}", value = out(reg32) value,
        address = in(reg64) address, options(nostack))
    };
    value
}

#[inline(always)]
pub(super) fn shuffle(value: u32, lane: u32) -> u32 {
    let result: u32;
    // SAFETY: All lanes participate, including those with no scale load.
    unsafe {
        asm!("shfl.sync.idx.b32 {result}, {value}, {lane}, 0x1f, 0xffffffff;",
        result = out(reg32) result, value = in(reg32) value, lane = in(reg32) lane,
        options(nomem, nostack))
    };
    result
}

#[inline(always)]
pub(super) fn signed_pair(bits: u32) -> u32 {
    let pair: u32;
    // SAFETY: Every signed byte is exactly representable in FP32 and BF16.
    unsafe {
        asm!("{{ .reg .b32 lo, hi; .reg .f32 flo, fhi; .reg .b16 blo, bhi;",
        "shl.b32 lo, {bits}, 24; shr.s32 lo, lo, 24;",
        "shl.b32 hi, {bits}, 16; shr.s32 hi, hi, 24;",
        "cvt.rn.f32.s32 flo, lo; cvt.rn.f32.s32 fhi, hi;",
        "cvt.rn.bf16.f32 blo, flo; cvt.rn.bf16.f32 bhi, fhi;",
        "mov.b32 {pair}, {{blo, bhi}}; }}", bits = in(reg32) bits,
        pair = out(reg32) pair, options(nomem, nostack))
    };
    pair
}

#[inline(always)]
pub(super) fn matrix(address: u32) -> [u32; 2] {
    let low: u32;
    let high: u32;
    // SAFETY: Full warp participates with aligned swizzled shared BF16 addresses.
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{low}, {high}}}, [{address}];",
        low = out(reg32) low, high = out(reg32) high, address = in(reg32) address,
        options(nostack))
    };
    [low, high]
}

#[inline(always)]
pub(super) fn mma(a: [u32; 4], b: [u32; 2], d: [f32; 4]) -> [f32; 4] {
    let [mut d0, mut d1, mut d2, mut d3] = d;
    // SAFETY: Full warp executes the upstream row.col BF16 fragment mapping.
    unsafe {
        asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
        "{{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, ",
        "{{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
        d0 = inout(reg32) d0, d1 = inout(reg32) d1,
        d2 = inout(reg32) d2, d3 = inout(reg32) d3,
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack))
    };
    [d0, d1, d2, d3]
}

#[inline(always)]
pub(super) fn half(bits: u32) -> f32 {
    let value: f32;
    // SAFETY: Register-only IEEE FP16 conversion of the selected low word.
    unsafe {
        asm!("{{ .reg .b16 half; cvt.u16.u32 half, {bits}; cvt.f32.f16 {value}, half; }}",
        bits = in(reg32) bits & 0xffff, value = out(reg32) value, options(nomem, nostack))
    };
    value
}

#[inline(always)]
pub(super) fn fma(dot: f32, scale: f32, acc: f32) -> f32 {
    let value: f32;
    // SAFETY: Register-only round-to-nearest fused multiply-add.
    unsafe {
        asm!("fma.rn.f32 {value}, {dot}, {scale}, {acc};",
        value = out(reg32) value, dot = in(reg32) dot, scale = in(reg32) scale,
        acc = in(reg32) acc, options(nomem, nostack))
    };
    value
}

#[inline(always)]
pub(super) fn add(left: f32, right: f32) -> f32 {
    let value: f32;
    // SAFETY: Register-only addition with explicit FP32 rounding.
    unsafe {
        asm!("add.rn.f32 {value}, {left}, {right};", value = out(reg32) value,
        left = in(reg32) left, right = in(reg32) right, options(nomem, nostack))
    };
    value
}

#[inline(always)]
pub(super) fn partial(address: u32) -> [f32; 4] {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    // SAFETY: Aligned initialized partner fragment after the corresponding CTA barrier.
    unsafe {
        asm!("ld.shared.v4.f32 {{{d0}, {d1}, {d2}, {d3}}}, [{address}];",
        d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
        address = in(reg32) address, options(nostack))
    };
    [d0, d1, d2, d3]
}

#[inline(always)]
pub(super) fn store_odd_partial_barrier(address: u32, warp: u32, acc: [f32; 4]) {
    unsafe {
        asm!("{{ .reg .b32 parity; .reg .pred odd;",
        "and.b32 parity, {warp}, 1; setp.ne.u32 odd, parity, 0;",
        "@odd st.shared.v4.f32 [{address}], {{{d0}, {d1}, {d2}, {d3}}};",
        "bar.sync 0; }}",
        address = in(reg32) address, warp = in(reg32) warp,
        d0 = in(reg32) acc[0], d1 = in(reg32) acc[1],
        d2 = in(reg32) acc[2], d3 = in(reg32) acc[3], options(nostack))
    };
}

#[inline(always)]
pub(super) fn store_partial(address: u32, d: [f32; 4]) {
    // SAFETY: Each lane owns its aligned 16-byte fragment during this reduction phase.
    unsafe {
        asm!("st.shared.v4.f32 [{address}], {{{d0}, {d1}, {d2}, {d3}}};",
        address = in(reg32) address, d0 = in(reg32) d[0], d1 = in(reg32) d[1],
        d2 = in(reg32) d[2], d3 = in(reg32) d[3], options(nostack))
    };
}

#[inline(always)]
pub(super) fn bf16(value: f32) -> u16 {
    let bits: u16;
    // SAFETY: Register-only BF16 round-to-nearest-even output conversion.
    unsafe {
        asm!("cvt.rn.bf16.f32 {bits}, {value};", bits = out(reg16) bits,
        value = in(reg32) value, options(nomem, nostack))
    };
    bits
}
