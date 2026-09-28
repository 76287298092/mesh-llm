//! Per-row log-softmax statistics for teacher-forced scoring.
//!
//! One 256-thread CTA scores one row of BF16 logits: log-sum-exp, the target
//! log-probability and the top-64 `(id, logprob)` pairs ordered by descending
//! logit with the lower id first on ties (positive and negative zero tie).
//!
//! Arithmetic: BF16 widens exactly to FP32. The maximum is exact. Each
//! `exp(x - max)` is an FP32 Rust polynomial (`logprob_math.rs`); terms are
//! accumulated in FP64 in a fixed per-thread order and a fixed shared-memory
//! tree, and the logarithm is FP64. Log-probabilities are formed in FP64 as
//! `(x - max) - ln(sum)` and rounded once to FP32.
//!
//! Top-64 selection is exact and order independent: a two-level 8-bit radix
//! histogram over the 16-bit ordered BF16 key finds the 64th key; ties at that
//! key are admitted in id order through a contiguous per-thread prefix count;
//! the 64 candidates are then sorted by (key descending, id ascending). Atomic
//! histogram and slot counters only affect counts and pre-sort placement, so
//! every output is deterministic run to run.
//!
//! Status word per row: bit 0 a nonfinite logit, bit 1 target outside the
//! vocabulary, bit 2 nonfinite target logit, bit 3 fewer than 64 finite
//! logits (padding ids are `u32::MAX` with `-inf` log-probabilities). Any
//! nonzero status invalidates the row.

use crate::logprob_math::{exp_nonpositive_f32, ln_positive_f64};
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

const THREADS: u32 = 256;
const WARPS: u32 = THREADS / 32;
const BUCKETS: u32 = 256;
const TOP_K: u32 = 64;
const INVALID_ID: u32 = u32::MAX;

const HISTOGRAM_OFFSET: usize = 0;
const COUNTS_OFFSET: usize = HISTOGRAM_OFFSET + (WARPS * BUCKETS) as usize * 4;
const SUMS_OFFSET: usize = COUNTS_OFFSET + THREADS as usize * 4;
const RANKS_OFFSET: usize = SUMS_OFFSET + THREADS as usize * 8;
const IDS_OFFSET: usize = RANKS_OFFSET + TOP_K as usize * 4;
const CONTROL_OFFSET: usize = IDS_OFFSET + TOP_K as usize * 4;
/// Control words: selected high byte, threshold key, keys above threshold,
/// needed ties, slot counter, nonfinite flag, retained count.
const CONTROL_WORDS: usize = 8;
const BUCKET: usize = 0;
const THRESHOLD: usize = 1;
const GREATER: usize = 2;
const NEEDED: usize = 3;
const SLOTS: usize = 4;
const NONFINITE: usize = 5;
const RETAINED: usize = 6;

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

/// Declare the CTA scratch once per kernel and return its generic address.
#[inline(always)]
fn shared_base() -> *mut u8 {
    let base: u64;
    // SAFETY: Declares 11,840 CTA-local bytes (8-byte aligned) and converts the
    // shared address to a generic pointer. Called exactly once per kernel.
    unsafe {
        asm!(
            ".shared .align 8 .b8 row_logprob_topk_scratch[11840];",
            "cvta.shared.u64 {base}, row_logprob_topk_scratch;",
            base = out(reg64) base,
            options(nostack),
        )
    };
    base as *mut u8
}

#[inline(always)]
fn barrier() {
    // SAFETY: Every thread of the CTA reaches each call site uniformly. The
    // missing `nomem` makes this a compiler memory barrier as well.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn atomic_add(address: *mut u32, value: u32) -> u32 {
    let previous: u32;
    // SAFETY: `address` is a generic pointer into this CTA's shared scratch.
    unsafe {
        asm!(
            "atom.add.u32 {previous}, [{address}], {value};",
            previous = out(reg32) previous,
            address = in(reg64) address,
            value = in(reg32) value,
            options(nostack),
        )
    };
    previous
}

#[derive(Clone, Copy)]
struct Scratch {
    base: *mut u8,
}

impl Scratch {
    #[inline(always)]
    fn word(self, offset: usize, index: usize) -> *mut u32 {
        // SAFETY: Callers index within the fixed layout above.
        unsafe { self.base.add(offset + index * 4).cast() }
    }
    #[inline(always)]
    fn sum(self, index: usize) -> *mut f64 {
        // SAFETY: `index < THREADS`; the region is 8-byte aligned.
        unsafe { self.base.add(SUMS_OFFSET + index * 8).cast() }
    }
    #[inline(always)]
    fn get(self, offset: usize, index: usize) -> u32 {
        // SAFETY: In-bounds shared word; ordering is provided by `barrier`.
        unsafe { read_volatile(self.word(offset, index)) }
    }
    #[inline(always)]
    fn set(self, offset: usize, index: usize, value: u32) {
        // SAFETY: In-bounds shared word; ordering is provided by `barrier`.
        unsafe { write_volatile(self.word(offset, index), value) }
    }
    #[inline(always)]
    fn control(self, index: usize) -> u32 {
        self.get(CONTROL_OFFSET, index)
    }
}

#[inline(always)]
fn is_finite_bf16(bits: u32) -> bool {
    bits & 0x7f80 != 0x7f80
}

/// Map finite BF16 into unsigned order, merging both zero signs.
#[inline(always)]
fn ordered_key(bits: u32) -> u32 {
    let canonical = if bits & 0x7fff == 0 { 0 } else { bits };
    if canonical & 0x8000 != 0 {
        canonical ^ 0xffff
    } else {
        canonical ^ 0x8000
    }
}

#[inline(always)]
fn widen(bits: u32) -> f32 {
    f32::from_bits(bits << 16)
}

#[inline(always)]
fn load(row: *const u16, index: u32) -> u32 {
    // SAFETY: Callers bound `index` by the host-validated vocabulary.
    u32::from(unsafe { *row.add(index as usize) })
}

/// Zero the per-warp histograms.
#[inline(always)]
fn clear_histograms(scratch: Scratch, thread: u32) {
    let mut warp = 0;
    while warp < WARPS {
        scratch.set(HISTOGRAM_OFFSET, (warp * BUCKETS + thread) as usize, 0);
        warp += 1;
    }
}

/// Count finite keys into per-warp histograms of the high byte, or of the low
/// byte for keys whose high byte equals `high` when `high` is not `u32::MAX`.
#[inline(always)]
fn histogram(scratch: Scratch, row: *const u16, vocabulary: u32, thread: u32, high: u32) {
    let warp_base = (thread >> 5) * BUCKETS;
    let mut index = thread;
    while index < vocabulary {
        let bits = load(row, index);
        if is_finite_bf16(bits) {
            let key = ordered_key(bits);
            let bucket = if high == u32::MAX {
                Some(key >> 8)
            } else if key >> 8 == high {
                Some(key & 0xff)
            } else {
                None
            };
            if let Some(bucket) = bucket {
                atomic_add(scratch.word(HISTOGRAM_OFFSET, (warp_base + bucket) as usize), 1);
            }
        } else if high == u32::MAX {
            scratch.set(CONTROL_OFFSET, NONFINITE, 1);
        }
        index += THREADS;
    }
}

/// Fold per-warp histograms into `COUNTS[bucket]`, one bucket per thread.
#[inline(always)]
fn fold_histograms(scratch: Scratch, thread: u32) {
    let mut total = 0;
    let mut warp = 0;
    while warp < WARPS {
        total += scratch.get(HISTOGRAM_OFFSET, (warp * BUCKETS + thread) as usize);
        warp += 1;
    }
    scratch.set(COUNTS_OFFSET, thread as usize, total);
}

/// Thread 0: find the bucket holding the `retained`-th largest key, given
/// `above` keys in higher buckets. Returns (bucket, keys above that bucket).
#[inline(always)]
fn select_bucket(scratch: Scratch, retained: u32, above: u32) -> (u32, u32) {
    let mut cumulative = above;
    let mut bucket = BUCKETS;
    while bucket > 0 {
        bucket -= 1;
        let count = scratch.get(COUNTS_OFFSET, bucket as usize);
        if cumulative + count >= retained {
            return (bucket, cumulative);
        }
        cumulative += count;
    }
    (0, cumulative)
}

/// Contiguous vocabulary slice owned by `thread` for id-ordered tie handling.
#[inline(always)]
fn owned_slice(vocabulary: u32, thread: u32) -> (u32, u32) {
    let chunk = vocabulary.div_ceil(THREADS);
    let start = (thread * chunk).min(vocabulary);
    (start, (start + chunk).min(vocabulary))
}

/// Count threshold ties in this thread's slice, then prefix them in id order.
#[inline(always)]
fn prefix_ties(scratch: Scratch, row: *const u16, vocabulary: u32, thread: u32) {
    let threshold = scratch.control(THRESHOLD);
    let (start, end) = owned_slice(vocabulary, thread);
    let mut ties = 0;
    let mut index = start;
    while index < end {
        let bits = load(row, index);
        if is_finite_bf16(bits) && ordered_key(bits) == threshold {
            ties += 1;
        }
        index += 1;
    }
    scratch.set(COUNTS_OFFSET, thread as usize, ties);
    barrier();
    if thread == 0 {
        let mut running = 0;
        let mut slot = 0;
        while slot < THREADS as usize {
            let count = scratch.get(COUNTS_OFFSET, slot);
            scratch.set(COUNTS_OFFSET, slot, running);
            running += count;
            slot += 1;
        }
    }
    barrier();
}

/// Place every key above the threshold (atomic slots) and the lowest-id ties.
#[inline(always)]
fn collect(scratch: Scratch, row: *const u16, vocabulary: u32, thread: u32) {
    let threshold = scratch.control(THRESHOLD);
    let greater = scratch.control(GREATER);
    let needed = scratch.control(NEEDED);
    let mut order = scratch.get(COUNTS_OFFSET, thread as usize);
    let (start, end) = owned_slice(vocabulary, thread);
    let mut index = start;
    while index < end {
        let bits = load(row, index);
        if is_finite_bf16(bits) {
            let key = ordered_key(bits);
            let slot = if key > threshold {
                Some(atomic_add(scratch.word(CONTROL_OFFSET, SLOTS), 1))
            } else if key == threshold {
                order += 1;
                (order - 1 < needed).then_some(greater + order - 1)
            } else {
                None
            };
            if let Some(slot) = slot {
                scratch.set(RANKS_OFFSET, slot as usize, key);
                scratch.set(IDS_OFFSET, slot as usize, index);
            }
        }
        index += 1;
    }
}

/// Thread 0: insertion sort by (key descending, id ascending).
#[inline(always)]
fn sort_candidates(scratch: Scratch, retained: u32) {
    let sort_key = |slot: usize| -> u64 {
        (u64::from(scratch.get(RANKS_OFFSET, slot)) << 32)
            | u64::from(INVALID_ID - scratch.get(IDS_OFFSET, slot))
    };
    let mut next = 1;
    while next < retained as usize {
        let (key, id) = (
            scratch.get(RANKS_OFFSET, next),
            scratch.get(IDS_OFFSET, next),
        );
        let packed = (u64::from(key) << 32) | u64::from(INVALID_ID - id);
        let mut slot = next;
        while slot > 0 && sort_key(slot - 1) < packed {
            scratch.set(RANKS_OFFSET, slot, scratch.get(RANKS_OFFSET, slot - 1));
            scratch.set(IDS_OFFSET, slot, scratch.get(IDS_OFFSET, slot - 1));
            slot -= 1;
        }
        scratch.set(RANKS_OFFSET, slot, key);
        scratch.set(IDS_OFFSET, slot, id);
        next += 1;
    }
}

/// Fixed-order FP64 sum of FP32 `exp(x - max)` over finite logits.
#[inline(always)]
fn sum_exponentials(
    scratch: Scratch,
    row: *const u16,
    vocabulary: u32,
    thread: u32,
    maximum: f32,
) -> f64 {
    let mut local = 0.0_f64;
    let mut index = thread;
    while index < vocabulary {
        let bits = load(row, index);
        if is_finite_bf16(bits) {
            local += f64::from(exp_nonpositive_f32(widen(bits) - maximum));
        }
        index += THREADS;
    }
    // SAFETY: Each thread owns its FP64 slot until the barrier below.
    unsafe { write_volatile(scratch.sum(thread as usize), local) };
    barrier();
    let mut stride = THREADS / 2;
    while stride > 0 {
        if thread < stride {
            // SAFETY: Disjoint pairs per level; levels are separated by barriers.
            unsafe {
                let left = read_volatile(scratch.sum(thread as usize));
                let right = read_volatile(scratch.sum((thread + stride) as usize));
                write_volatile(scratch.sum(thread as usize), left + right);
            }
        }
        barrier();
        stride >>= 1;
    }
    // SAFETY: The final barrier published slot 0.
    unsafe { read_volatile(scratch.sum(0)) }
}

struct Outputs {
    target_logprobs: *mut f32,
    logsumexps: *mut f32,
    top_ids: *mut u32,
    top_logprobs: *mut f32,
    status: *mut u32,
}

/// Write one row's outputs; threads 0..64 each own one top-k slot.
#[inline(always)]
fn write_row(
    scratch: Scratch,
    row: *const u16,
    outputs: &Outputs,
    context: [u32; 5],
    maximum: f32,
    log_sum: f64,
) {
    let [thread, row_index, vocabulary, target, retained] = context;
    let base = row_index as usize;
    let maximum = f64::from(maximum);
    let logprob = |bits: u32| (f64::from(widen(bits)) - maximum) - log_sum;
    if thread < TOP_K {
        let slot = base * TOP_K as usize + thread as usize;
        let (id, value) = if thread < retained {
            let id = scratch.get(IDS_OFFSET, thread as usize);
            (id, logprob(load(row, id)) as f32)
        } else {
            (INVALID_ID, f32::NEG_INFINITY)
        };
        // SAFETY: Each (row, slot) output element has exactly one writer.
        unsafe {
            outputs.top_ids.add(slot).write(id);
            outputs.top_logprobs.add(slot).write(value);
        }
    }
    if thread == 0 {
        let mut status = scratch.control(NONFINITE);
        let mut target_logprob = f32::NAN;
        if target >= vocabulary {
            status |= 2;
        } else {
            let bits = load(row, target);
            if is_finite_bf16(bits) {
                target_logprob = logprob(bits) as f32;
            } else {
                status |= 4;
            }
        }
        if retained < TOP_K {
            status |= 8;
        }
        // SAFETY: Thread 0 is the unique writer of this row's scalars.
        unsafe {
            outputs.target_logprobs.add(base).write(target_logprob);
            outputs.logsumexps.add(base).write((maximum + log_sum) as f32);
            outputs.status.add(base).write(status);
        }
    }
}

/// Radix-select the retained candidates into sorted shared lists.
/// Returns the retained count (uniform across the CTA).
#[inline(always)]
fn select_top(scratch: Scratch, row: *const u16, vocabulary: u32, thread: u32) -> u32 {
    clear_histograms(scratch, thread);
    if thread < CONTROL_WORDS as u32 {
        scratch.set(CONTROL_OFFSET, thread as usize, 0);
    }
    barrier();
    histogram(scratch, row, vocabulary, thread, u32::MAX);
    barrier();
    fold_histograms(scratch, thread);
    clear_histograms(scratch, thread);
    barrier();
    if thread == 0 {
        let mut finite = 0;
        let mut bucket = 0;
        while bucket < BUCKETS as usize {
            finite += scratch.get(COUNTS_OFFSET, bucket);
            bucket += 1;
        }
        let retained = finite.min(TOP_K);
        let (high, above) = select_bucket(scratch, retained, 0);
        scratch.set(CONTROL_OFFSET, BUCKET, high);
        scratch.set(CONTROL_OFFSET, GREATER, above);
        scratch.set(CONTROL_OFFSET, RETAINED, retained);
    }
    barrier();
    let retained = scratch.control(RETAINED);
    if retained == 0 {
        return 0;
    }
    histogram(scratch, row, vocabulary, thread, scratch.control(BUCKET));
    barrier();
    fold_histograms(scratch, thread);
    barrier();
    if thread == 0 {
        let high = scratch.control(BUCKET);
        let (low, greater) = select_bucket(scratch, retained, scratch.control(GREATER));
        scratch.set(CONTROL_OFFSET, THRESHOLD, (high << 8) | low);
        scratch.set(CONTROL_OFFSET, GREATER, greater);
        scratch.set(CONTROL_OFFSET, NEEDED, retained - greater);
    }
    barrier();
    prefix_ties(scratch, row, vocabulary, thread);
    collect(scratch, row, vocabulary, thread);
    barrier();
    if thread == 0 {
        sort_candidates(scratch, retained);
    }
    barrier();
    retained
}

/// Score one row of BF16 logits per CTA.
///
/// Outputs per row `r`: `target_logprobs[r]`, `logsumexps[r]`, `status[r]`,
/// and 64 entries at `top_ids[r * 64..]` / `top_logprobs[r * 64..]`.
///
/// # Safety
/// Launch with grid `[rows, 1, 1]` and block `[256, 1, 1]`, no dynamic shared
/// memory. `vocabulary` must be in `64..=262144` and `row_stride >= vocabulary`.
/// `logits` must hold `(rows - 1) * row_stride + vocabulary` readable `u16`
/// values; `targets`, `target_logprobs`, `logsumexps` and `status` hold `rows`
/// elements; `top_ids`/`top_logprobs` hold `rows * 64`. All allocations are
/// live, aligned and nonoverlapping until completion.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn row_logprob_topk_bf16(
    logits: *const u16,
    targets: *const u32,
    target_logprobs: *mut f32,
    logsumexps: *mut f32,
    top_ids: *mut u32,
    top_logprobs: *mut f32,
    status: *mut u32,
    row_stride: u32,
    vocabulary: u32,
) {
    let (thread, row_index) = thread_and_row();
    let scratch = Scratch {
        base: shared_base(),
    };
    // SAFETY: The launch contract bounds this row and its target.
    let (row, target) = unsafe {
        (
            logits.add(row_index as usize * row_stride as usize),
            *targets.add(row_index as usize),
        )
    };
    let outputs = Outputs {
        target_logprobs,
        logsumexps,
        top_ids,
        top_logprobs,
        status,
    };
    let retained = select_top(scratch, row, vocabulary, thread);
    let context = [thread, row_index, vocabulary, target, retained];
    if retained == 0 {
        write_row(scratch, row, &outputs, context, f32::NAN, f64::NAN);
        return;
    }
    let maximum = widen(load(row, scratch.get(IDS_OFFSET, 0)));
    let sum = sum_exponentials(scratch, row, vocabulary, thread, maximum);
    write_row(scratch, row, &outputs, context, maximum, ln_positive_f64(sum));
}
