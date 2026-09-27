# F02 native FP8 shared-tile prefill

Status: first Rust candidate written; not compiled, assembled, launched, compared
on GPU, sanitized, or benchmarked. The exact integer-decomposition kernel remains
the default and its existing evidence is unchanged.

## Candidate contract

`kernels/nvptx/fp8_native_prefill.rs` defines `fp8_prefill_native` with the same
nine-argument ABI shape as `fp8_prefill_exact`, in this order:

1. `codes_a: *const u8` — row-major E4M3FN `[m, k]`.
2. `codes_w: *const u8` — row-major E4M3FN `[n, k]`, interpreted as `W^T`.
3. `row_scales: *const f32` — one positive FP32 scale per input row.
4. `weight_scales: *const u16` — one positive BF16 scale per output channel.
5. `out: *mut u16` — row-major BF16 `[m, n]` output.
6. `unrounded: *mut f32` — row-major FP32 `[m, n]` output.
7. `m: u32`, `n: u32`, `k: u32`.

The parent launch contract is `grid=[ceil(n/64), ceil(m/32), 1]`,
`block=[128, 1, 1]`, and 6,144 bytes of statically allocated shared memory.
The kernel requires positive dimensions with `m<=2048`, `n<=262144`, and
`k<=32768`; scales must be finite and positive, and input codes must exclude
E4M3FN NaNs `0x7f` and `0xff`.

## Tile and arithmetic

Each CTA stages one logical `32x64` A tile and one `64x64` row-major W tile for
`K=64`. Four warps cover a `2x2` arrangement of `16x32` output tiles. Each warp
issues four `m16n8k32` fragments for its N slice and two K fragments per shared
tile. Fragment byte mapping follows the PTX ISA's documented `m16n8k32` E4M3
layout. Fragment values are read directly from shared memory; `ldmatrix` is not
used.

The device instruction is `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32` and targets the existing SM120a NVPTX compilation lane. Its accumulator is FP32. The epilogue rounds each multiplication separately in this order:
`(accumulator * row_scale) * BF16(weight_scale)`, then stores the FP32 value and
BF16 round-to-nearest-even value. This is an explicit experimental arithmetic
profile. MMA accumulation order and FP32 rounding need not match the exact
signed-integer kernel or the independent FP64 reference bit for bit.

When both base pointers are 16-byte aligned and `k % 16 == 0`, each CTA thread
stages aligned 16-byte vectors with `cp.async`, commits and waits for its copy
group, then reaches a CTA barrier before fragment loads. Invalid row, channel,
and K vectors are zeroed. Otherwise the whole CTA uniformly uses scalar byte
staging, including K tails that cannot form complete vectors. A CTA barrier at
the end of each compute tile ensures every warp has finished shared reads before
the next tile overwrites the single shared buffer.

This first version deliberately has one 6,144-byte shared tile. It does not
overlap staging of a future K tile with current MMAs and does not use two shared
buffers. A double-buffered copy/compute pipeline is a follow-up after this
candidate assembles and its fragment/tail behavior passes GPU qualification.

## Independent reference and qualification

`reference/fp8_native_prefill.rs` independently decodes logical E4M3FN values,
accumulates products in FP64, casts once to FP32, applies the same scale order,
and rounds to BF16 RNE. Its comparison helper reports nonfinite counts, BF16
mismatches, maximum absolute/relative error, and RMS error. It intentionally has
no pass threshold; the parent owns the qualification budgets. Host tests cover
signed values, nonuniform scales, odd M/N/K tails, large cancellation, and the
error report shape.

The retained exact prefill profile reached 259.38 input tokens/s at 128 tokens
and 296.58 at 512 in earlier qualification. Parent profile attribution for the
512-input run reported 725.04 ms of 1,563.93 ms in `fp8_prefill_exact` (46.4%).
That is time spent in our current exact kernel, not a predicted speedup for this
candidate. Candidate speed and model-level impact are **not measured**.

Before changing the default, the parent must include both files in the NVPTX and
host reference module registries, build the pinned NVPTX target, run `ptxas` and
driver JIT on SM120a, compare signed/nonuniform/tail/cancellation fixtures with
the reference and exact control, run memory/race/synchronization sanitizers, and
measure matched shapes at 128/512/2048 input rows with real N/K dimensions.
No model prefill, parity, or serving-readiness claim follows from this source.

## References and evidence limits

- [NVIDIA PTX ISA: MMA syntax and `m16n8k32` fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k32)
- [NVIDIA PTX ISA: asynchronous copy commit/wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group)
- Source-layout reference: `kernels/nvptx/fp8_prefill_exact.rs` in this crate.
- GPU architecture/toolchain/revision/evidence path for this candidate: not yet measured; parent qualification must record them.

Parent integration: 229 crate tests and host Clippy passed on macOS on 2026-09-27, together with F03. NVPTX compilation passed. No GPU or model qualification has occurred. The parent added a synthetic projection-check command with fixed numerical budgets and workspace reuse checks; Linux compilation and GPU execution remain pending.

Linux compile caught the new parent probe entrypoint visibility; corrected to crate visibility before GPU execution.

The parent Linux probe also required the current Clippy byte-chunk API; updated before execution.

Parent GPU check, 2026-09-27: eleven synthetic F01/F02 cases passed on Carrack RTX5090, including M/N/K tails and K=5120. BF16 outputs matched the independent fixtures exactly; native FP8 raw FP32 scaled error was at most 9.58e-7. Workspace stable-address reuse and aborted-lease poisoning passed. All three CUDA sanitizer tools reported zero errors/hazards. Evidence: `../evidence/iterate-20260927/features-projection/`. PTX SHA256 `35c12bcfe57d282985b02bae256be0770991816519bc1a6cbf825a9cf369aa75`. JIT decode uses 33 registers and no local memory; prefill uses 56 registers, 64 local bytes and 6144 shared bytes. These are synthetic correctness checks, not model qualification or speed measurements. Ninfer and other GPU processes remained running.

## Full-model ablation integration

The process-fixed internal selector `MESH_SPECIALIZE_FP8_PROFILE=native-prefill` selects the native 32x64x64 kernel for FP8 projections with at least 16 rows. Smaller shapes retain exact A8 kernels; NVFP4 remains unchanged. The default `exact` profile is unchanged, and invalid names fail closed. Model bench/profile/check reports label the arithmetic profile. Native accumulation can change BF16 rounding and recurrent state; strict legacy partition checks remain strict. Additional final-logit drift metrics are diagnostic only, not a relaxed qualification gate. Next evidence is matched exact/native full-model prefill and per-kernel profiles, with process-contention sampling. No promotion or throughput claim yet.

Initial Linux Clippy caught diagnostic helper placement after the test module; moved it before tests. No GPU trial had started.
