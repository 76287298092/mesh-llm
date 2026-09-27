use core::arch::asm;

const THREADS: u32 = 128;
const TILE_ELEMENTS: u32 = 1024;
const INDEX_MASK: u32 = 0x3ffff;
const INVALID_INDEX: u32 = u32::MAX;

#[inline(always)]
fn thread_and_tile() -> (u32, u32) {
    let thread: u32;
    let tile: u32;
    // SAFETY: Reads the calling thread's coordinates without changing memory.
    unsafe {
        asm!(
            "mov.u32 {thread}, %tid.x;",
            "mov.u32 {tile}, %ctaid.x;",
            thread = out(reg32) thread,
            tile = out(reg32) tile,
            options(nomem, nostack),
        )
    };
    (thread, tile)
}

#[inline(always)]
fn thread_index() -> u32 {
    let thread: u32;
    // SAFETY: Reads the calling thread's x coordinate without changing memory.
    unsafe {
        asm!(
            "mov.u32 {thread}, %tid.x;",
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    thread
}

#[inline(always)]
fn is_finite_bf16(bits: u32) -> bool {
    bits & 0x7f80 != 0x7f80
}

/// Map finite BF16 values into unsigned integer order, merging both zero signs.
#[inline(always)]
fn ordered_bf16(bits: u32) -> u32 {
    let canonical = if bits & 0x7fff == 0 { 0 } else { bits };
    if canonical & 0x8000 != 0 {
        canonical ^ 0xffff
    } else {
        canonical ^ 0x8000
    }
}

/// Pack sort order, inverted token index, and the original BF16 bits.
#[inline(always)]
fn candidate_key(index: u32, bits: u32) -> u64 {
    let rank = ordered_bf16(bits);
    let first_index_rank = INDEX_MASK - index;
    ((rank as u64) << 34) | ((first_index_rank as u64) << 16) | bits as u64
}

#[inline(always)]
fn candidate_index(key: u64) -> u32 {
    if key == 0 {
        INVALID_INDEX
    } else {
        INDEX_MASK ^ ((key >> 16) as u32 & INDEX_MASK)
    }
}

#[inline(always)]
fn candidate_bits(key: u64) -> u32 {
    key as u32 & 0xffff
}

#[inline(always)]
fn shuffle_down_u64(value: u64, offset: u32) -> u64 {
    let result: u64;
    // SAFETY: All 32 lanes in every warp call this with the same offset and full mask.
    unsafe {
        asm!(
            "{{",
            ".reg .b32 low, high, partner_low, partner_high;",
            "mov.b64 {{low, high}}, {value};",
            "shfl.sync.down.b32 partner_low, low, {offset}, 0x1f, 0xffffffff;",
            "shfl.sync.down.b32 partner_high, high, {offset}, 0x1f, 0xffffffff;",
            "mov.b64 {result}, {{partner_low, partner_high}};",
            "}}",
            value = in(reg64) value,
            offset = in(reg32) offset,
            result = out(reg64) result,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn shuffle_down_u32(value: u32, offset: u32) -> u32 {
    let result: u32;
    // SAFETY: All 32 lanes in every warp call this with the same offset and full mask.
    unsafe {
        asm!(
            "shfl.sync.down.b32 {result}, {value}, {offset}, 0x1f, 0xffffffff;",
            result = out(reg32) result,
            value = in(reg32) value,
            offset = in(reg32) offset,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn warp_reduce(mut best_key: u64, mut first_nonfinite: u32) -> (u64, u32) {
    let mut offset = 16;
    while offset != 0 {
        let partner_key = shuffle_down_u64(best_key, offset);
        if partner_key > best_key {
            best_key = partner_key;
        }
        first_nonfinite = first_nonfinite.min(shuffle_down_u32(first_nonfinite, offset));
        offset >>= 1;
    }
    (best_key, first_nonfinite)
}

#[inline(always)]
fn shared_partials_base() -> u32 {
    let base: u32;
    // SAFETY: Declares 4 CTA-local records, each 16 bytes, and returns their shared address.
    unsafe {
        asm!(
            ".shared .align 8 .b8 greedy_bf16_warp_partials[64];",
            "mov.u32 {base}, greedy_bf16_warp_partials;",
            base = out(reg32) base,
            options(nostack),
        )
    };
    base
}

#[inline(always)]
fn publish_warp_partial_and_sync(base: u32, thread: u32, key: u64, first_nonfinite: u32) {
    let lane = thread & 31;
    let warp = thread >> 5;
    // SAFETY: All CTA threads call this site uniformly. Exactly one leader per
    // warp writes its two initialized fields before every thread reaches the barrier.
    unsafe {
        asm!(
            "{{",
            ".reg .pred leader;",
            ".reg .b32 address, nonfinite_address;",
            "setp.eq.u32 leader, {lane}, 0;",
            "mad.lo.u32 address, {warp}, 16, {base};",
            "add.u32 nonfinite_address, address, 8;",
            "@leader st.shared.b64 [address], {key};",
            "@leader st.shared.u32 [nonfinite_address], {first_nonfinite};",
            "bar.sync 0;",
            "}}",
            lane = in(reg32) lane,
            warp = in(reg32) warp,
            base = in(reg32) base,
            key = in(reg64) key,
            first_nonfinite = in(reg32) first_nonfinite,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_key(base: u32, warp: u32) -> u64 {
    let key: u64;
    let address = base + warp * 16;
    // SAFETY: The preceding CTA barrier makes the warp leader's key visible.
    unsafe {
        asm!(
            "ld.shared.b64 {key}, [{address}];",
            key = out(reg64) key,
            address = in(reg32) address,
            options(nostack),
        )
    };
    key
}

#[inline(always)]
fn load_shared_nonfinite(base: u32, warp: u32) -> u32 {
    let index: u32;
    let address = base + warp * 16 + 8;
    // SAFETY: The preceding CTA barrier makes the warp leader's index visible.
    unsafe {
        asm!(
            "ld.shared.u32 {index}, [{address}];",
            index = out(reg32) index,
            address = in(reg32) address,
            options(nostack),
        )
    };
    index
}

/// Select the best finite BF16 logit in each 1,024-element vocabulary tile.
///
/// Each partial record is four `u32` words: winning index, original BF16 bits,
/// first nonfinite index in the tile or `u32::MAX`, and a zero reserved word.
/// A tile with no finite values uses `u32::MAX` and zero for its winning pair.
///
/// # Safety
/// Launch with grid `[ceil(vocabulary / 1024), 1, 1]` and block
/// `[128, 1, 1]`. `vocabulary` must be in `1..=262144`; `logits` must contain
/// that many readable `u16` values and `partials` must contain four writable
/// `u32` values per CTA. The allocations must be live until completion,
/// correctly aligned, and nonoverlapping. The caller validates these extents.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn greedy_bf16_tiles(
    logits: *const u16,
    partials: *mut u32,
    vocabulary: u32,
) {
    let (thread, tile) = thread_and_tile();
    let mut best_key = 0_u64;
    let mut first_nonfinite = INVALID_INDEX;
    let tile_start = tile * TILE_ELEMENTS;
    let mut item = 0;
    while item < TILE_ELEMENTS / THREADS {
        let index = tile_start + thread + item * THREADS;
        if index < vocabulary {
            // SAFETY: The launch contract bounds the tile and guards the vocabulary tail.
            let bits = unsafe { *logits.add(index as usize) } as u32;
            if is_finite_bf16(bits) {
                let key = candidate_key(index, bits);
                if key > best_key {
                    best_key = key;
                }
            } else {
                first_nonfinite = first_nonfinite.min(index);
            }
        }
        item += 1;
    }

    let (best_key, first_nonfinite) = warp_reduce(best_key, first_nonfinite);
    let base = shared_partials_base();
    publish_warp_partial_and_sync(base, thread, best_key, first_nonfinite);

    if thread == 0 {
        let mut cta_best_key = 0_u64;
        let mut cta_first_nonfinite = INVALID_INDEX;
        let mut warp_index = 0;
        while warp_index < 4 {
            let warp_key = load_shared_key(base, warp_index);
            if warp_key > cta_best_key {
                cta_best_key = warp_key;
            }
            cta_first_nonfinite = cta_first_nonfinite.min(load_shared_nonfinite(base, warp_index));
            warp_index += 1;
        }

        let partial_start = tile as usize * 4;
        let winning_index = candidate_index(cta_best_key);
        // SAFETY: CTA x owns its unique four-word partial record.
        unsafe {
            partials.add(partial_start).write(winning_index);
            partials
                .add(partial_start + 1)
                .write(candidate_bits(cta_best_key));
            partials.add(partial_start + 2).write(cta_first_nonfinite);
            partials.add(partial_start + 3).write(0);
        }
    }
}

/// Finish the deterministic reduction over tile records and report nonfinites.
///
/// The four output words are winning index, status (`0` finite, `1` contains a
/// nonfinite input), lowest nonfinite input index or `u32::MAX`, and original
/// winning BF16 bits. If no finite input exists, the winning pair is
/// `u32::MAX` and zero; status still rejects the selection.
///
/// # Safety
/// Launch with grid `[1, 1, 1]` and block `[128, 1, 1]`. `tile_count` must be
/// in `1..=256`; `partials` must contain `tile_count` records of four `u32`
/// words emitted by `greedy_bf16_tiles`, and `result` must contain four writable
/// `u32` words. All pointers must be live, aligned, and nonoverlapping until
/// completion. The host must validate the vocabulary and exact tile count.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn greedy_bf16_finish(
    partials: *const u32,
    result: *mut u32,
    tile_count: u32,
) {
    let thread = thread_index();
    let mut best_key = 0_u64;
    let mut first_nonfinite = INVALID_INDEX;
    let mut tile = thread;
    while tile < tile_count {
        let partial_start = tile as usize * 4;
        // SAFETY: Each thread reads its assigned records within the host-validated extent.
        let (index, bits, nonfinite) = unsafe {
            (
                *partials.add(partial_start),
                *partials.add(partial_start + 1),
                *partials.add(partial_start + 2),
            )
        };
        if index != INVALID_INDEX {
            let key = candidate_key(index, bits & 0xffff);
            if key > best_key {
                best_key = key;
            }
        }
        first_nonfinite = first_nonfinite.min(nonfinite);
        tile += THREADS;
    }

    let (best_key, first_nonfinite) = warp_reduce(best_key, first_nonfinite);
    let base = shared_partials_base();
    publish_warp_partial_and_sync(base, thread, best_key, first_nonfinite);

    if thread == 0 {
        let mut cta_best_key = 0_u64;
        let mut cta_first_nonfinite = INVALID_INDEX;
        let mut warp_index = 0;
        while warp_index < 4 {
            let warp_key = load_shared_key(base, warp_index);
            if warp_key > cta_best_key {
                cta_best_key = warp_key;
            }
            cta_first_nonfinite = cta_first_nonfinite.min(load_shared_nonfinite(base, warp_index));
            warp_index += 1;
        }

        let winning_index = candidate_index(cta_best_key);
        let status = u32::from(cta_first_nonfinite != INVALID_INDEX);
        // SAFETY: The single CTA exclusively writes the four-word result record.
        unsafe {
            result.add(0).write(winning_index);
            result.add(1).write(status);
            result.add(2).write(cta_first_nonfinite);
            result.add(3).write(candidate_bits(cta_best_key));
        }
    }
}
