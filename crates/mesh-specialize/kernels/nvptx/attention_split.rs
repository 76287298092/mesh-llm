//! Unqualified BF16 split-sequence D256 attention. One CTA owns a KV head/split.
//! Six warps reuse each shared K/V tile across the six query heads and all M rows.
use super::attention_split_math as math;

const THREADS: usize = 192;
const TILE: usize = 16;
const WIDTH: usize = 256;
const STRIDE: usize = 258;

#[derive(Clone, Copy)]
struct Inputs {
    q: *const u16,
    k: *const u16,
    v: *const u16,
    partial: *mut f32,
    rows: usize,
    past: usize,
    capacity: usize,
    slots: usize,
    scale: f32,
}

#[inline(always)]
unsafe fn stage(input: Inputs, base: u32, start: usize, head: usize, thread: usize) {
    let mut word = thread;
    while word < TILE * WIDTH / 2 {
        let position = start + word / (WIDTH / 2);
        let channel = (word % (WIDTH / 2)) * 2;
        let (k, v) = if position < input.past + input.rows {
            let offset = (position * 4 + head) * WIDTH + channel;
            // SAFETY: The entire aligned word belongs to an initialized token/head row.
            unsafe {
                (
                    input.k.add(offset).cast::<u32>().read(),
                    input.v.add(offset).cast::<u32>().read(),
                )
            }
        } else {
            (0, 0)
        };
        math::store_word(base + word as u32 * 4, k);
        math::store_word(base + 8192 + word as u32 * 4, v);
        word += THREADS;
    }
}

#[inline(always)]
unsafe fn clear_inactive(input: Inputs, head: usize, split: usize, thread: usize) {
    let mut index = thread;
    while index < input.rows * 6 * STRIDE {
        let row = index / (6 * STRIDE);
        let qhead = head * 6 + (index / STRIDE) % 6;
        let item = index % STRIDE;
        let output = ((row * 24 + qhead) * input.slots + split) * STRIDE + item;
        let value = if item == 0 { f32::NEG_INFINITY } else { 0.0 };
        // SAFETY: Every lane owns distinct workspace elements in this inactive split.
        unsafe {
            input.partial.add(output).write(value);
        }
        index += THREADS;
    }
}

struct State<const M: usize> {
    q: [[f32; 8]; M],
    acc: [[f32; 8]; M],
    maximum: [f32; M],
    sum: [f32; M],
}

impl<const M: usize> State<M> {
    #[inline(always)]
    unsafe fn new(input: Inputs, qhead: usize, lane: usize) -> Self {
        let mut state = Self {
            q: [[0.0; 8]; M],
            acc: [[0.0; 8]; M],
            maximum: [f32::NEG_INFINITY; M],
            sum: [0.0; M],
        };
        for row in 0..M {
            for c in 0..8 {
                let index = (row * 24 + qhead) * WIDTH + lane + c * 32;
                // SAFETY: Dispatch specializes exactly the host-validated M, with one warp/head.
                state.q[row][c] =
                    unsafe { f32::from_bits((input.q.add(index).read() as u32) << 16) };
            }
        }
        state
    }

    #[inline(always)]
    fn key(&mut self, input: Inputs, position: usize, k: &[f32; 8], v: &[f32; 8]) {
        for row in 0..M {
            // Uniform across the warp. All lanes still reach every CTA barrier.
            if position <= input.past + row {
                let mut dot = 0.0;
                for c in 0..8 {
                    dot += self.q[row][c] * k[c];
                }
                let score = math::warp_sum(dot) * input.scale;
                let next = self.maximum[row].max(score);
                let alpha = if self.sum[row] == 0.0 {
                    0.0
                } else {
                    math::exp(self.maximum[row] - next)
                };
                let weight = math::exp(score - next);
                self.sum[row] = self.sum[row] * alpha + weight;
                self.maximum[row] = next;
                for c in 0..8 {
                    self.acc[row][c] = self.acc[row][c] * alpha + weight * v[c];
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn store(&self, input: Inputs, qhead: usize, split: usize, lane: usize) {
        for row in 0..M {
            let offset = ((row * 24 + qhead) * input.slots + split) * STRIDE;
            // SAFETY: One warp/head owns this workspace row; lane zero owns its metadata.
            unsafe {
                if lane == 0 {
                    input.partial.add(offset).write(self.maximum[row]);
                    input.partial.add(offset + 1).write(self.sum[row]);
                }
                for c in 0..8 {
                    input
                        .partial
                        .add(offset + 2 + lane + c * 32)
                        .write(self.acc[row][c]);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn compute<const M: usize>(
    input: Inputs,
    head: usize,
    split: usize,
    thread: usize,
    active: usize,
    base: u32,
) {
    let lane = thread % 32;
    let qhead = head * 6 + thread / 32;
    let tiles = (input.past + input.rows).div_ceil(TILE);
    let begin = tiles * split / active;
    let end = tiles * (split + 1) / active;
    // SAFETY: Checked entrypoint shapes and 192-thread launch fix all Q coordinates.
    let mut state = unsafe { State::<M>::new(input, qhead, lane) };
    for tile in begin..end {
        // SAFETY: All threads stage distinct words; tails zero-fill without reading poison.
        unsafe {
            stage(input, base, tile * TILE, head, thread);
        }
        math::barrier();
        for key in 0..TILE {
            let mut k = [0.0; 8];
            let mut v = [0.0; 8];
            for c in 0..8 {
                let address = base + ((key * WIDTH + lane + c * 32) * 2) as u32;
                k[c] = math::load_bf16(address);
                v[c] = math::load_bf16(address + 8192);
            }
            state.key(input, tile * TILE + key, &k, &v);
        }
        // No producer may replace a word before all six consumer warps finish.
        math::barrier();
    }
    // SAFETY: This CTA owns exactly (head,split); every query/channel is unique.
    unsafe {
        state.store(input, qhead, split, lane);
    }
}

#[inline(always)]
unsafe fn dispatch<const DECODE: bool>(input: Inputs, qheads: u32, kvheads: u32, width: u32) {
    if !(1..=8).contains(&input.rows)
        || qheads != 24
        || kvheads != 4
        || width != 256
        || input.capacity == 0
        || input.capacity > 262144
        || input.past > input.capacity
        || input.rows > input.capacity - input.past
        || input.slots == 0
        || input.slots > 85
        || !(input.scale > 0.0 && input.scale <= f32::MAX)
    {
        return;
    }
    if DECODE && input.rows != 1 {
        return;
    }
    let active = (input.past + input.rows).div_ceil(64).min(85);
    if input.slots < active {
        return;
    }
    let (thread, head, split) = math::coordinates();
    let (thread, head, split) = (thread as usize, head as usize, split as usize);
    if head >= 4 || split >= input.slots {
        return;
    }
    let base = math::shared_base();
    // SAFETY: Every CTA takes the same specialization or inactive branch uniformly.
    unsafe {
        if split >= active {
            clear_inactive(input, head, split, thread);
            return;
        }
        if DECODE {
            compute::<1>(input, head, split, thread, active, base);
            return;
        }
        match input.rows {
            1 => compute::<1>(input, head, split, thread, active, base),
            2 => compute::<2>(input, head, split, thread, active, base),
            3 => compute::<3>(input, head, split, thread, active, base),
            4 => compute::<4>(input, head, split, thread, active, base),
            5 => compute::<5>(input, head, split, thread, active, base),
            6 => compute::<6>(input, head, split, thread, active, base),
            7 => compute::<7>(input, head, split, thread, active, base),
            8 => compute::<8>(input, head, split, thread, active, base),
            _ => {}
        }
    }
}

/// Write FP32 (maximum, sum, unnormalized accumulator[256]) per split.
///
/// # Safety
/// Grid [4,slots,1], block [192,1,1], dynamic shared 0. Rows=1..8, Q heads=24,
/// KV heads=4, width=256; 0<=past, past+rows<=capacity<=262144. Positive finite
/// scale (model uses 0.0625). slots>=min(85,ceil((past+rows)/64)) and <=85.
/// Q: BF16 [rows,24,256]; K/V: BF16 [capacity,4,256], initialized through
/// past+rows. Q/K/V and scaled scores must be finite. Workspace: aligned FP32
/// [rows,24,slots,258]. Inputs/output are disjoint, aligned (K/V to 4 bytes),
/// device-resident and live through the later reduce launch. KV append and
/// upstream RoPE must have completed on this stream. No KV writes occur here.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_split_bf16(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    partial: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    split_slots: u32,
    scale: f32,
) {
    // SAFETY: Caller upholds the allocation, launch, and finite-input contracts above.
    unsafe {
        dispatch::<false>(
            Inputs {
                q,
                k,
                v,
                partial,
                rows: rows as usize,
                past: past as usize,
                capacity: capacity as usize,
                slots: split_slots as usize,
                scale,
            },
            query_heads,
            kv_heads,
            width,
        );
    }
}

/// Graph-friendly variant: replace the by-value `past` with a device u32 pointer.
///
/// # Safety
/// Same contract as attention_split_bf16. `position` points to one initialized
/// aligned device u32 holding past, updated before this launch on the same stream.
/// Argument order is q,k,v,partial,position,rows,qheads,kvheads,width,capacity,slots,scale.
/// Captured grids/workspace slots must cover every replay length. Every launched
/// inactive split writes a neutral record, so replay cannot retain stale partials.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_split_bf16_position(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    partial: *mut f32,
    position: *const u32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    capacity: u32,
    split_slots: u32,
    scale: f32,
) {
    // SAFETY: Position is a live initialized device scalar, stream-ordered before this launch.
    let past = unsafe { position.read_volatile() };
    // SAFETY: Caller upholds the same data/launch contract as the by-value variant.
    unsafe {
        dispatch::<false>(
            Inputs {
                q,
                k,
                v,
                partial,
                rows: rows as usize,
                past: past as usize,
                capacity: capacity as usize,
                slots: split_slots as usize,
                scale,
            },
            query_heads,
            kv_heads,
            width,
        );
    }
}

/// Single-row specialization prevents M=8 register pressure from taxing decode.
/// # Safety
/// Same ABI and contract as attention_split_bf16, except rows must equal one.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_split_decode_bf16(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    partial: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    split_slots: u32,
    scale: f32,
) {
    // SAFETY: Caller upholds the allocation, launch, and finite-input contracts above.
    unsafe {
        dispatch::<true>(
            Inputs {
                q,
                k,
                v,
                partial,
                rows: rows as usize,
                past: past as usize,
                capacity: capacity as usize,
                slots: split_slots as usize,
                scale,
            },
            query_heads,
            kv_heads,
            width,
        );
    }
}

/// Single-row device-position specialization.
/// # Safety
/// Same ABI/contract as attention_split_bf16_position, except rows must equal one.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_split_decode_bf16_position(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    partial: *mut f32,
    position: *const u32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    capacity: u32,
    split_slots: u32,
    scale: f32,
) {
    // SAFETY: Position is a live initialized device scalar, stream-ordered before this launch.
    let past = unsafe { position.read_volatile() };
    // SAFETY: Caller upholds the same data/launch contract as the by-value variant.
    unsafe {
        dispatch::<true>(
            Inputs {
                q,
                k,
                v,
                partial,
                rows: rows as usize,
                past: past as usize,
                capacity: capacity as usize,
                slots: split_slots as usize,
                scale,
            },
            query_heads,
            kv_heads,
            width,
        );
    }
}
