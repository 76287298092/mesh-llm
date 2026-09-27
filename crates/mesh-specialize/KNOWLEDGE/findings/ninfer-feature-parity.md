# Ninfer feature and performance gap assessment

Parent-only source analysis, 2026-09-27. Ninfer source pin:
[`9e163eee4b8acec21ab0ac765107b6a3f287b217`](https://github.com/Neroued/ninfer/tree/9e163eee4b8acec21ab0ac765107b6a3f287b217).
Our implementation inspected at `9d72c0fc2657330337db765e2038a7f7c28d3aea`.
The archived source is from Carrack; its correspondence to the installed Ninfer
binary is not proven. These are source-supported mechanisms, not an assertion
that every route ran in every measured request.

## What the percentages mean

Performance shortfall = `100 * (1 - our throughput / Ninfer throughput)` on a
matched workload. For latency, use `100 * (1 - Ninfer latency / our latency)`.
A missing implementation is not evidence of a 100% performance shortfall.
Implementation completeness, tile dimensions, instruction counts, and memory
compression ratios are not substitutes for measured speed.

We have no matched per-feature Ninfer kernel timings. Therefore the third column
below is **unmeasured**, not a guessed percentage. This measurement gap must be
closed before claiming feature parity. Feature speedups also interact and cannot
be added to explain the end-to-end gap.

The available end-to-end numbers show the scale, but use different prompts,
lengths and runtime settings:

| Measurement | Our retained result versus installed Ninfer | Unmatched throughput shortfall |
| --- | --- | --- |
| Ordinary short decode | 25.26 versus 164.2–201.0 tokens/s; Ninfer has MTP4 enabled | 84.6–87.4% |
| Prefill | 296.58 at 512 inputs versus 5,956–10,416 at 724–42,837 inputs | 95.0–97.2% |
| Best tested MTP decode | 38.21 on the 32-output Python fixture versus 164.2–201.0 on Ninfer's 512-output cases | 76.7–81.0% |

These are scale indicators, **not matched benchmark results or per-feature
attribution**. New overlapping Carrack trials are explicitly contended at the
user's direction and cannot establish clean speedup percentages.

## Feature comparison

| Ninfer performance feature | Our equivalent implementation and actual gap | Performance shortfall against this Ninfer feature |
| --- | --- | --- |
| **F01. Shape-specific decode and small-batch GEMV.** Direct BF16 activations with FP8/NVFP4 weights; vector loads, multiple independent accumulators, shape/token thresholds. | Dedicated exact quantized decode exists: FP8 A8 integer dots and NVFP4 A4 grouped DP4A/i64 dots. It is a different arithmetic path, still quantizes inputs, and does not implement Ninfer's A16 GEMV routes. | **Unmeasured.** No same-shape, same-arithmetic Ninfer GEMV timing. |
| **F02. Native tensor-core, pipelined prefill GEMM.** FP8 MMA; typical 64×128×128 tile, two shared-memory stages, asynchronous copies and fragment pipelining. NVFP4 also has native block-scaled MMA routes. | NVFP4 already uses native block-scaled MMA, but small warp tiles. FP8 uses 16×8×32 exact integer decomposition: nine INT8 MMAs plus i64 reconstruction for each K tile, direct global loads, no shared-memory pipeline. This is a major algorithm/hardware mismatch. | **Unmeasured.** Overall unmatched prefill is 95.0–97.2% behind, but that cannot be assigned solely to GEMM. |
| **F03. Fused projection and epilogues.** Joined attention Q/gate/K/V and GDN QKV/Z entrypoints; gate/up projections with SwiGLU output; residual and norm/control specializations. | Separate projections repeatedly quantize the same input; gate/up outputs and activation intermediates are materialized. BF16 rounding boundaries differ from Ninfer's fused FP32 epilogues. | **Unmeasured.** Need fusion-on/off timings with a declared rounding contract. |
| **F04. Chunked GDN prefill.** Matrix-based chunk preparation, state passing and output kernels; recurrent route for decode/tails. | All 48 GDN layers use a token-serial recurrence even during prefill, repeatedly reading/writing the recurrent matrix. Increasing the outer prompt batch does not remove this serial algorithm. | **Unmeasured.** Need equal-length GDN hidden/state comparisons and timings. |
| **F05. Tiled attention with online softmax and context splitting.** Separate packed-prefill and cache-decode kernels; shared-memory staging and bounded partial reductions. | Scalar/CTA causal attention with FP64 reduction arithmetic, BF16 KV, and separate stages. It is a correctness-oriented path rather than a competitive long-context attention implementation. | **Unmeasured.** Need matched context lengths, head geometry, cache format and causal masks. |
| **F06. FP8 KV storage and cache-aware kernels.** Row-scaled E4M3 cache codec, FP16 represented scales, paging-aware addressing and FP8 cache attention routes. | Persistent BF16 K/V only. Capacity allocation is not long-context qualification. FP8 would roughly halve payload bytes before scale metadata; that is not a proven 2× speedup. | **Unmeasured.** No FP8 KV equivalent or same-context comparison. |
| **F07. Planned reusable workspace.** Program construction allocates a persistent scratch arena; per-operation views reuse planned storage. | Model weights and state are persistent, but projections and intermediate operators allocate/free temporary CUDA buffers repeatedly. | **Unmeasured.** Need allocator-call counts and matched wall-time ablation. |
| **F08. CUDA graph execution and GPU-resident control.** Prepared decode profiles replay graphs; stream-ordered work and device-side token/control operations avoid repeated host dispatch. | Launches use the default stream, wrappers synchronize frequently, and logits/control return to the host. No graph capture/replay. Workspace stability is a prerequisite. | **Unmeasured.** Kernel event totals alone do not measure host-submission savings. |
| **F09. Efficient MTP verification and recurrent-state recovery.** Batched target verification, device acceptance/hidden selection and compact GDN replay records; graph-aware MTP rounds. | Working greedy MTP, independent head checks and exact target output/state preservation. It forks full state, downloads logits, and reruns accepted target inputs after rejection. Depth-four low-acceptance prose regresses to about 13 tokens/s. | **Unmeasured.** Our depth/acceptance results are not a matched Ninfer MTP ablation. |
| **F10. Prefix and continuation reuse.** Prefix checkpoints include recurrent state and KV ownership; warm requests avoid recomputing the common prefix. | Session forks exist for MTP, but no identity-bound reusable prompt checkpoint cache. Every standalone request prefills again. | **Unmeasured.** Ninfer reused 42,830/42,837 tokens in one warm case; that reuse fraction is not our speed parity. |

Concurrency/batching and serving policy are additional product capabilities, but
the existing comparison is serial. Do not attribute its single-request speed gap
to concurrency capacity two. CUDA graphs and fused kernels reduce overhead, but
they cannot alone explain a prefill gap while our core algorithms remain serial
or use much more expensive arithmetic.

## Source map

All Ninfer paths below are relative to the pinned source tree linked above:

- F01: `src/ops/linear/fp8/fp8_gemv.cuh`, `fp8_config.h`, `shapes/n5120_k17408.cu`; NVFP4 counterparts under `src/ops/linear/nvfp4/`.
- F02: `src/ops/common/mma.cuh:59`, `src/ops/linear/fp8/fp8_a8_schedule.cuh`, `fp8_a8_mma.cuh:145` (staging), `:247` (native FP8 MMA). Our `kernels/nvptx/fp8_prefill_exact.rs`, `fp8_verify_exact.rs`, `nvfp4_linear.rs`.
- F03: `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu`, `fp8_linear_swiglu_output.cuh`; `src/models/qwen3_5/execution/attention.cpp`, `gdn.cpp`; our `resident_mlp.rs`, `resident_attention.rs`, `resident_gdn.rs`.
- F04: `src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:240` and `chunked/`; our `kernels/nvptx/gdn_recurrent.rs:141`.
- F05: `src/ops/softmax_attention/dense/packed/kernel.cuh`, `context/kernel.cuh`, `causal_cache/`; our `kernels/nvptx/causal_attention.rs` and `attention_reduction.rs`.
- F06: `src/ops/kv_cache/fp8_e4m3_row_codec.cuh`, `src/core/paged_kv_cache.cpp`; our `src/engine/layout.rs` and `resident_state.rs`.
- F07: `src/models/qwen3_5/program/program_impl.cpp:38`, `src/core/arena.cu`; our `resident_fp8.rs`, `resident_mlp.rs`, `driver.rs`.
- F08: `src/models/qwen3_5/program/graphs.cpp`, `graph_execution.h`, `src/core/decode_graph.cpp`; our `driver.rs` launch/synchronization methods and `resident_head.rs`.
- F09: `src/models/qwen3_5/program/speculative/target_verification.cpp`, `mtp.cpp`, `src/core/gdn_replay_records.cpp`; our `resident_speculation.rs`, `resident_mtp.rs`.
- F10: `src/models/qwen3_5/program/prefill.cpp`, `src/core/paged_kv_cache.cpp`; our session cursor/fork is not a prefix cache.

Hardware instruction contracts: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
The SM120 native FP8 route uses `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4...e4m3.e4m3`;
do not substitute SM100-only tensor-memory instructions merely because both GPUs
are called Blackwell. Async-copy completion and CTA barriers must be explicit.

## Measured priorities within our runtime

These percentages are shares of **our measured GPU event time**, not Ninfer
parity. The qualified 512-input profile totals 1,563.93 ms: FP8 projections
725.04 ms (46.4%), causal attention 243.78 ms (15.6%), GDN recurrence 238.44 ms
(15.2%), and NVFP4 projections 195.95 ms (12.5%). Together those paths explain
89.7% of our prefill kernel time. Fixing submission overhead alone cannot remove
that work. Source: `evidence/iterate-20260927/prefill-qualified/profile-long.json`.

The qualified short-decode profile totals 30.32 ms: FP8 projections 32.6%,
NVFP4 projections 27.2%, BF16 projections 8.9%, FP8 input quantization 8.6%,
and GDN recurrence 8.2%. Source:
`evidence/iterate-20260927/exact-qualified/profile.json`. Event instrumentation
changes scheduling; these are attribution measurements, not throughput samples.

## Implementation assignments and qualification strategy

One distinct subagent owns each F01–F10 feature. The parent performed this analysis
and owns integration, compilation, benchmark orchestration and service control.
Only three workers can run alongside the parent, so execute bounded waves.
Workers must not delegate, run Cargo/SSH/Git, edit shared module registries or
switch the resident default before parent qualification.

1. **F01:** Independently implement packed-weight A16 GEMV candidates, starting FP8. Vectorize loads, reuse BF16 input, use multiple FP32 accumulator chains and warp reductions. Preserve exact baseline as a control. A16 changes activation precision: create an independent A16 reference and an explicit experimental profile; do not pretend old bits must match or loosen old gates.
2. **F02:** Implement native FP8 MMA first with independently verified fragments, then a modest shared-memory tiled kernel with guarded K/M/N tails and double-buffered `cp.async`. Benchmark 128/512/2048 rows and real N/K shapes. Keep NVFP4's already-native MMA distinct from FP8's integer emulation. No Ninfer code copying or compute-library dependency.
3. **F03:** Start by sharing input quantization for gate/up and fuse the existing BF16-boundary SwiGLU epilogue. Preserve both projection rounding and SiLU rounding for the first candidate; later FP32 epilogues require a separate arithmetic profile. Avoid allocating concatenated weight copies just to claim fusion.
4. **F04:** Derive chunked gated-delta state updates from the recurrence independently. Implement a bounded chunk operator and host oracle, with prefix/tail and final-state checks. Stage matrix reassociation as experimental; compare per-token hidden and recurrent-state error against fixed budgets, not only final logits.
5. **F05:** Implement FP32 online-softmax attention with stable max/sum rescaling, causal masking and grouped-query head mapping. Start BF16 KV so cache quantization does not confound the first measurement. Include nonzero past, odd tails and extreme scores; qualify against the existing FP64 reference before resident dispatch.
6. **F06:** Implement an independent row-scaled FP8 KV codec and bounded cache layout. Keep BF16 cache as the default/control. Validate zero rows, saturation, represented scale boundaries, append/read addressing and long-context error; integrate attention only after F05 is stable.
7. **F07:** Implement context-owned reusable scratch with checked alignment, nonoverlap, high-water/capacity and lifetimes. Start one projection chain, count avoided allocations, then extend. Live outputs must never be overwritten; failure must poison/reject reuse safely.
8. **F08:** After F07, add dynamically loaded stream/graph driver primitives with RAII and failure cleanup. Capture only stable allocations/addresses and device-side mutable position inputs; reject capture if host synchronization/allocation occurs. Start a bounded fixed-shape replay probe before whole-model capture.
9. **F09:** Implement compact recurrence replay records (decay, key, update vector) so accepted GDN state can be recovered without rerunning the whole target. Preserve convolution history and accepted KV rows as well as GDN state. Force rejection at each draft position and all-accepted commits; exact target tokens/state remain the gate for the existing arithmetic profile.
10. **F10:** Implement a bounded identity-bound prefix checkpoint policy over full session state: weights, arithmetic profile, KV format, context geometry and token prefix must match. Include recurrent/conv state and hidden/token boundary semantics. Test mismatch rejection, eviction and suffix equivalence; do not add cross-request reuse until ownership is sound.

Per-feature measurement contract: same GPU, same bytes/scales, shape, arithmetic
profile, warmup and timing boundary; both isolated kernel and full-model impact;
median plus range; process-contention evidence; quality and sanitizer gates.
If a Ninfer-equivalent path changes arithmetic, report performance and quality
separately. Store Ninfer timings from its own tooling outside our runtime—never
import its source or parser into the bespoke engine. Until these measurements
exist, keep the percentage column unmeasured.

## Feature owner ledger

| Feature | Dedicated subagent | First bounded implementation status |
| --- | --- | --- |
| F01 | `feature_decode_gemv` | A16 FP8 synthetic GPU checks and all three sanitizers pass; model integration pending |
| F02 | `feature_native_prefill` | Native FP8 tile synthetic GPU checks and all three sanitizers pass; model integration pending |
| F03 | `feature_fusion` | Exact FP8 gate/up and SwiGLU fusion delivered; CPU tests and PTX compilation pass |
| F04 | `feature_chunked_gdn` | Chunked operator delivered; CPU tests and PTX compilation pass |
| F05 | `feature_tiled_attention` | Online-softmax synthetic GPU checks and all three sanitizers pass; model integration pending |
| F06 | `feature_fp8_kv` | Codec delivered; CPU tests and PTX compilation pass; GPU/cache integration pending |
| F07 | `feature_workspace` | Reusable layout and lease checks pass on GPU; operator integration pending |
| F08 | `feature_graphs` | Driver graph API passes Linux checks, fixed-shape replay and leak/memory check; model capture pending |
| F09 | `feature_mtp_recovery` | Queued; native spawn rejected by agent thread limit after prior workers completed |
| F10 | `feature_prefix_cache` | Queued; native spawn rejected by agent thread limit after prior workers completed |

A delivered primitive is not an integrated or performance-qualified feature.
Only the parent promotes candidates after independent tests, GPU qualification
and appropriately labeled measurements. Existing defaults remain the control.

Allocation limit: eight distinct feature owners have run. The native agent tool repeatedly rejected F09 and F10 with `agent thread limit reached`, including after all three current workers completed. No worker was reassigned to another feature. F09/F10 remain unimplemented queued assignments, not claimed dispatched work.
