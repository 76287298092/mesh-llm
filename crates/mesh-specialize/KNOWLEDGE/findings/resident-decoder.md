# Connecting persistent decoder execution

Status: resident layer-zero GDN execution passes independent comparisons and all
three sanitizers for one/17 tokens. Whole/chunk/token execution is bit-exact across
partitions, including persistent history and recurrent state. Full decoder pending.

The next stage separates GPU operation execution from the old component-check
harnesses. Weight views bind exact dtype, row-major layout, shape and byte extent.
Forward calls validate input/state extents and CUDA context ownership, use the
same previously qualified kernels, and never substitute scalar or downloaded
intermediates. Host scalar reads are limited to immutable global scales at binding.

The persistent state arena has disjoint named histories, recurrent matrices and
K/V regions. Causal convolution must not alias old and next history: it writes
temporary next-history storage, synchronizes, then uses a checked device-to-device
copy into the persistent region. Recurrent updates operate on the existing region.
Device copies require distinct allocations, matching contexts and in-range slices.

The intended connected path is embedding, ordered 64-layer attention/GDN and
MLP/residual execution, final norm and logits. Existing kernel diagnostics remain
allocated temporarily to preserve kernel ABIs; allocation reuse and diagnostics-free
kernels are later optimizations. Full model correctness and partition equivalence
must pass before any model-throughput or usable-context claim.

No new device arithmetic was added in this extraction. GDN head width is capped
at 256 because its Q/K and gated-normalization kernels use one element per thread
in 256-thread reductions. The pinned model uses width 128. The embedding trial
currently computes an unused normalized output before the block normalizes again;
this duplicates work without changing the arithmetic. Neither temporary allocation
reuse nor launch/synchronization optimization has been implemented.

Initial integration: macOS passed 181 tests, Clippy and no-console checks;
Linux passed 224 library and 20 validator tests. Linux Clippy caught an unnecessary
explicit drop of the embedding view, which has no destructor. Remove that drop;
the view's last use already ends its borrow before freeing the weight arena.
Preserve the first lint log and rerun Linux qualification on the correction.

## Carrack qualification, 2026-09-27

Implementation `cafd236d8c1ee8069f6f9220c76aeafdb6343c68`; qualified corrected
source `45a0dd6762b1b7c43625a7bfede94df488743f1a`. Release xtask SHA256
`45b9f64aa0adab1e32e830713492c7cac75f09ba85a6143feaae37990f2a93ba`.
Unchanged PTX SHA256
`ee13b6bf3d34ee2ccceeaf4b9420c32ddc0fee2c97fc7f7d529f86ba06c306b9`.
The subsequent usage-message update only lists the new trial command.

RTX5090 GPU0, SM120, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`,
driver 615.71.09 / Driver API 13040; Rust 1.98.1 / LLVM 22.1.8; CUDA 13.4.92.
Pretrial idle sample: P1, graphics 195 MHz, memory 405 MHz, 9.12 W, 600 W limit.
No sustained clocks or inference throughput were measured.

The trial loads all 1,620 text tensors once. Layer zero uses resident FP8 QKV/Z
and output projections, BF16 A/B projections, causal convolution, GDN preparation,
recurrence, gated norm, both residual additions, and its NVFP4 MLP. CPU references
start independently from original checkpoint weights and token IDs; their host
weight storage is dropped before the GPU load. No reference values are fed into
the forward path. GPU readbacks are limited to the trial's final comparisons.
Immutable NVFP4 scale reads happen at binding, before forward execution.

| Tokens | Hidden BF16 differences | Aggregate normalized L2 | Worst-token normalized L2 | Minimum token cosine |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 / 5,120 | 0 | 0 | 0.9999999999999999 |
| 17 | 9,123 / 87,040 | 0.0018555658500450244 | 0.006882154344273434 | 0.9999763226918187 |

All hidden, history and recurrent-state aggregate and partition comparisons pass
the unchanged 1% normalized-L2 / 0.9999 cosine limits. History is exact against
the CPU reference (30,720 BF16 values per case). Recurrent-state aggregate L2 is
`3.799465151171864e-7` for one token and `0.00019173284034971095` for 17 tokens.
These bounds are single-block evidence, not model/logit parity.

The 17-token whole batch, `[1,16]` chunks and 17 one-token calls produce identical
BF16 hidden outputs/history and bit-identical FP32 recurrent state. Each sequence
starts from a fresh zeroed state arena. A live device-copy fixture verifies offset
copies and rejects source/destination overrun, same-allocation aliasing and foreign
CUDA contexts before a driver copy. The earlier FP8 MLP trial also passes all four
layer-56/63 one/17-token cases through the new FP8/NVFP4 dispatch enum.

Normal, memcheck, racecheck and synccheck reports all pass. Harness durations are
49.270652017 / 56.309219149 / 54.532773444 / 49.476875756 seconds respectively.
The FP8 regression takes 24.996667486 seconds. These durations include independent
CPU references, artifact validation and loading; they are not prefill/decode rates.
Memcheck and synccheck report zero errors; racecheck reports zero errors, warnings
or hazards. Linux repeats 224 library and 20 validator tests, Clippy with warnings
denied, and the release build successfully. The initial Clippy failure is retained.

Weight arena: 21,646,588,928 bytes. Full compiled state layout at capacity 17:
155,058,176 bytes. CUDA free memory before weights and after release is
32,221,822,976 bytes; after each sequence, while weights/state remain allocated,
10,419,830,784 bytes. Total CUDA-visible memory is 33,731,248,128 bytes. These are
allocation checkpoints, not peak measurements, and do not establish usable model
context. Only layer zero executes; the other state regions remain unused.

Reproduce via the saved `run-trials.sh`: build with `just specialize-tools-build`,
then `xtask specialize qwen-resident-gdn-check --artifact PATH --ptx PATH --device 0
--output NEW_FILE`, using the pinned raw-v1 artifact from the earlier residency
trial. Each invocation is bounded to 240 seconds and 8 GiB host memory, without
swap. [Raw evidence](../evidence/qwen-resident-gdn-20260927/) includes the exact
script, source/toolchain/device details, four reports, FP8 regression, build and
sanitizer logs, and restoration evidence. Working copies and PTX remain under
`target/specialize/qwen-resident-gdn-20260927/` on both hosts.

Ninfer stopped at 05:53:59 EDT, active at 05:57:54, engine ready at 05:58:00,
health HTTP 200. Ninfer PID 3099976 uses 30,046 MiB. ComfyUI PID 448118 remains
unchanged at 498 MiB. Carrack's original branch remains preserved at
`d8949ab608a8771115b8e9d2bc23aefad94a9cf9`.

Remaining: reference-free full-attention composition with persistent K/V, ordered
64-layer execution, final norm/logits, independent model-level correctness, then
actual model prefill/decode, memory peaks and usable-context comparison to Ninfer.

## Resident attention connection (qualification pending)

The next extraction binds resident Q/K norm weights and uses compact BF16 text
RoPE tables generated from a named CPU profile in `engine::rope`. It preserves
FP32 position/frequency multiplication, FP64 trigonometry rounded to FP32, then
BF16 RNE. Forward calls upload only position tables; projection and attention
intermediates remain on the device. No reference code is used in forward execution.

K/V append uses current-chunk pointers and the caller's `past` cursor, validates
exact cache extents and capacity, and writes named regions in the common state
arena. The caller must commit its cursor only after all decoder layers succeed;
on execution failure the session must be discarded because updates are not atomic.
The block connects Q/K preparation, causal attention, sigmoid gating, output
projection, post-attention norm, MLP and final residual using existing kernels.

`qwen-resident-attention-check` will compare one/17-token layer-three execution
against the independent whole-block reference, using embedding rows as synthetic
hidden input. It checks whole/chunk/token output/cache equivalence, exact initialized
K/V and the zero unused tail at every append boundary, plus capacity rejection
without cache mutation. Capacity is 20 in this bounded check, not a context claim.
Local/Linux checks and all three Carrack sanitizers remain pending for this change.

Initial attention Linux tests caught an incorrect expected value in the maximum
RoPE-table extent test: 2,048 rows * 128 frequencies * 2 bytes is 524,288 bytes,
not 262,144. The implementation's checked extent was correct; fix the test literal
and preserve the failure log. macOS passed 184 tests, Clippy and no-console checks.
