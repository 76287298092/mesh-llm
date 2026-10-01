# Native MTP Q8 FC sliced-K port

Status: unregistered source candidate only. Compilation, emitted PTX inspection,
GPU execution, independent numerical comparison, sanitizers and timing were NOT RUN.
No native FC qualification or whole-MTP claim follows from this port. Existing Q4
and residency evidence is separate.

## Attribution and pin

Derived from NInfer contributors' Apache-2.0 implementation pinned at
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`. Each Rust file carries the license,
attribution, pin and modification notice. The supplied source excerpt has no
top-level license or NOTICE file; no additional author identity was inferred.
Source root read for this task:
`/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/ninfer-ref-e31bc99b`.

Source correspondence:

- `src/ops/linear/q8/shapes/n5120_k10240.cu:9` selects C4 direct scales and C8 shared scales.
- `src/ops/linear/q8/q8_instance_launch.cuh:8` binds static K and capacity with default Exact=false.
- `src/ops/linear/q8/q8_schedule.cuh:107` defines the sliced-K geometry and storage sizes.
- `src/ops/linear/q8/q8_a16_sliced_k_mma.cuh:115` defines RuntimeActive staging and code swizzling.
- `src/ops/linear/q8/q8_a16_sliced_k_mma.cuh:249` defines scale distribution and G32 contraction.
- `src/ops/linear/q8/q8_a16_sliced_k_mma.cuh:331` defines paired split reduction and output ownership.
- `src/ops/common/mma.cuh:7` and `:33` define nontransposed ldmatrix.x2 and BF16 row.col MMA.
- `src/ops/common/memory.cuh:35` defines asynchronous cache policy and commit/wait operations.

## Candidate contract

Two exported entries in `kernels/nvptx/native_mtp_q8_sliced_k_fc.rs` retain
distinct static shared allocations. `native_mtp_q8_sliced_k_fc_c4` accepts T1,
capacity four, direct FP16 scales, 16400 shared bytes.
`native_mtp_q8_sliced_k_fc_c8` accepts T5, capacity eight, shared FP16 scales,
16896 shared bytes. Both accept code plane, scale plane, BF16 input, BF16 output,
and token count in that order. Host dispatch is intentionally absent.

Launch grid `[320,1,1]`, block `[256,1,1]`. N=5120, K=10240, 16 rows/CTA,
eight K-split warps of 64 elements and 20 K512 iterations. Both use one staging
slot, an eight-column physical activation tile and non-exact live token count.
Code rows have 10240 signed bytes; scale rows have 320 IEEE FP16 words.
Input and output are token-major. All pointers require 16-byte alignment and
nonoverlapping allocations. Device must support BF16 MMA and cp.async.

Each G32 dot uses unscaled signed codes represented exactly as BF16, two K16
MMAs from FP32 zero, then an FP16-to-FP32 scale FMA into the running FP32 sum.
Signed-byte conversion uses sign extension and exact FP32/BF16 conversions
instead of upstream's byte permutation and BF16 pair bias subtraction. This
preserves code values, including -128 and zero, but not its conversion instruction
sequence. There is no SIMT dot-product replacement.

Codes and shared scales use `cp.async.cg` 16-byte copies; activations use
`cp.async.ca` 16-byte copies. Only live columns are copied. Inactive columns are
not zero-filled, matching RuntimeActive source behavior. Full-warp ldmatrix/MMA
still execute; discarded columns do not become observable output. Sanitizer
behavior for these intentionally unstaged addresses needs explicit parent review.

CTA barriers protect staging consumption, single-slot reuse and staging/partial
union reuse. Odd splits write partials, even splits add their partners, and split
zero adds paired splits in order `((p01+p23)+p45)+p67` using explicit `add.rn.f32`.
Output conversion uses BF16 RNE and stores `token*5120+row` only for live tokens.

## Assembly inventory pending parent integration

New PTX sites live only in this candidate's private subtree: special-register
reads, static shared allocation, global address conversion, cp.async ca/cg16,
commit/wait, CTA barriers, shared scalar/vector loads/stores, global.nc scale
loads, full-warp indexed shuffle, signed-code conversion, ldmatrix.x2,
BF16 m16n8k16 MMA, FP16 conversion, FP32 FMA/add and BF16 RNE conversion.
References are the pinned memory/MMA/contraction files above. Parent must add
the corresponding central assembly inventory entries before registration.
No registration, mod.rs, manifest, runtime, admission or independent oracle file
was changed by this worker.

## Validation limits

### FC reduction synchronization correction

Parent-reported validation failed with 5504 synccheck errors. Parent inspection
of isolated PTX found the common second reduction barrier duplicated at static
lines 36312 (even warps) and 36346 (odd warps). These observations were supplied
by the parent, not reproduced by this worker.

The source correction replaces the Rust odd-warp branch/store and separate second
barrier with one uniformly called, memory-aware inline assembly helper. Its
internal odd-warp predicate guards only `st.shared.v4.f32`; `bar.sync 0` is
unconditional within the same assembly block. The block does not use `nomem`.
The first and third reduction barriers, FP32 additions and reduction order,
and RuntimeActive staging are unchanged. Inactive columns remain uninitialized.

This fix is unqualified. No build, emitted PTX inspection, GPU execution or
sanitizer rerun was performed by this worker. The parent owns propagation to
the isolated source, compilation, PTX inspection and all GPU qualification gates.
No native FC or whole-MTP qualification claim follows from this correction.

Work location: `/Users/ndizazzo/dev/worktrees/ninfer-direct-runtime`.
Worker used only source reads/searches, apply_patch and LSP diagnostics.
LSP diagnostics for all three Rust files timed out after 30000 ms each;
no clean diagnostic result is available. The 20-tool-call limit ended the task
after source review corrected FP16 low-word masking before conversion.
GPU architecture, driver, compiler version, resource use, numerical results and
timings are not measured. Parent owns build/integration and independent oracle
qualification for T1/T5, signed codes, nonuniform FP16 scales, repeated launches,
BF16 boundaries and all applicable CUDA sanitizers. No runnable reproduction
command exists until the parent integrates a standalone launch harness.
