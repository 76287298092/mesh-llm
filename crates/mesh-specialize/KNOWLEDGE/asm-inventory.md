# Assembly inventory

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, lane read | Thread identity | NVPTX | Lane-indexed input/output contract | Qualified within the 32-case probe |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, MMA | 16x8x64 block-scaled FP4 multiply | SM120a, PTX8.7 | Independent scalar decoded matrices | 4,096 exact matches; see [evidence](findings/rust-nvfp4-probe.md) |
| `kernels/nvptx/probes.rs:panic` | Fail-fast trap | NVPTX | Unexpected panic must fail launch | Unqualified |
| `kernels/nvptx/memory.rs:probe_shared_load`, lane/address/copy/barrier | `cp.async.ca/cg`, commit and wait, shared synchronization | SM120a | Two seeded 256-halfword arrays | 16 cases pass; sanitizers clean |
| `kernels/nvptx/memory.rs:load_x2/load_x4` | Normal/transposed `ldmatrix` x2/x4 | SM120a | Logical 8x8 row/column mapping in `memory_fixtures.rs` | 2,048 exact outputs with copies |
| `kernels/nvptx/ordinary_mma.rs` | BF16/FP16 m16n8k16 and INT8 m16n8k32 MMA, lane read | SM120a | Twelve independent scalar matrix products in `ordinary_fixtures.rs` | 1,536 exact outputs; sanitizers clean |
| `kernels/nvptx/register_budget.rs` | `setmaxnreg` dec24/inc64, barriers, thread read | SM120a | 128 exact XOR outputs; launch requires reported allocation >=64 registers | 128 outputs pass; 64 registers reported; sanitizers clean |
| `kernels/nvptx/rms_norm.rs` | Shared reduction, barriers, explicit rounded FP32 arithmetic, thread/block coordinates | SM120a | Independent f64 RMSNorm | Qualified; see [representative kernels](findings/representative-kernels.md) and Qwen entry regression |
| `kernels/nvptx/nvfp4_gemm.rs` | Repeated NVFP4 MMA, lane/block coordinates | SM120a | Independent logical GEMM and separate cuBLAS reference | Qualified; see [representative kernels](findings/representative-kernels.md) and Qwen entry regression |

Compiler emission alone is not qualification. Keep execution evidence and any
failed attempts in a findings/dead-ends entry before promoting these rows.

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/embedding_norm.rs` | Thread/block coordinates, shared 256-thread reduction and barriers, explicit rounded FP32 add/multiply/divide/square-root | SM120a | Independent scalar real-weight embedding and zero-centered RMSNorm in `reference/embedding_norm.rs` | 696,320 values pass; memory/race/sync checks clean; see [entry trial](findings/qwen-entry.md) |

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/fp8_quantize.rs` | CTA coordinates, shared max reduction/barriers, rounded FP32 division | SM120a | Independent exhaustive nearest-value FP8 encoder; finite/tie/zero/tiny GPU fixtures | Exact codes/scales; all three sanitizers clean; see [projections](findings/qwen-projections.md) |
| `kernels/nvptx/fp8_linear.rs` | Warp/CTA coordinates, E4M3 m16n8k32 MMA, rounded FP32 scale multiplication | SM120a | Logical f64 dot products and NVIDIA fragment mapping | 294,912 real outputs meet tolerance; all three sanitizers clean; see [projections](findings/qwen-projections.md) |
| `kernels/nvptx/bf16_linear.rs` | Warp/CTA coordinates and zero-start BF16 m16n8k16 MMA tiles with FP64 accumulation and absolute-product bounds | SM120a | Logical f64 BF16 dot products, cancellation/tail fixtures and independent model reference | Original component results in [projections](findings/qwen-projections.md); full-model midpoint correction pending qualification in [resident model](findings/resident-model.md) |
| `kernels/nvptx/bf16_linear_rounding.rs` | Rounded FP64 multiply/add and FP64-to-FP32 RNE for BF16-ambiguous outputs | SM120a | Independent logical FP64 dot, cancellation/tail fixtures and layer-12 real gate regression | Pending qualification in [resident model](findings/resident-model.md); conservative empirical error interval, performance cost unmeasured |
| `kernels/nvptx/causal_conv4.rs` | CTA/thread coordinates, rounded FP32 multiply/add; SiLU now uses shared Rust FP64 polynomial | SM120a | Independent f64 convolution/SiLU and raw-state reference; whole/chunk/single-token equivalence | Prior activation passed component tolerances; full-model boundary diagnostic found rounding divergence; correction pending qualification in [resident model](findings/resident-model.md) |
| `kernels/nvptx/gdn_prepare.rs` | CTA/thread coordinates, two shared norm reductions/barriers, rounded arithmetic/sqrt, approximate exp2/log2 with subnormal support | SM120a | Independent f64 norm/exp/log1p reference; head/zero/extreme/underflow fixtures | 73,728 real Q/K and 2,592 gate values pass; all three sanitizers clean; see [preparation](findings/gdn-preparation.md) |
| `kernels/nvptx/gdn_recurrent.rs` | CTA/thread coordinates and explicit rounded FP32 multiply/add/subtract | SM120a | Independent logical scalar recurrence with hand-computed fixtures and separate f64 reduction diagnostics | 110,592 real outputs and state match scalar exactly; chunk state exact and sanitizers clean; see [recurrence](findings/gdn-recurrence.md) |
| `kernels/nvptx/gated_rms_norm.rs` | CTA/thread coordinates, shared square reduction/barriers, rounded arithmetic/sqrt; shared Rust FP64 SiLU | SM120a | Independent f64 gated norm reference with explicit BF16 boundaries and shared direct gamma | Prior activation qualified in [GDN output](findings/gdn-output.md); shared SiLU correction pending qualification in [resident model](findings/resident-model.md) |
| `kernels/nvptx/residual_norm.rs` | CTA/thread coordinates, shared square reduction/barriers, rounded FP32 add/multiply/divide/sqrt | SM120a | Independent scalar BF16 residual sum and f64 RMSNorm with zero-centered gamma | 92,160 real values pass; sanitizers clean; see [post-attention](findings/post-attention.md) |
| `kernels/nvptx/nvfp4_quantize.rs` | CTA/lane coordinates, full-mask butterfly shuffles, rounded FP32 multiply/divide | SM120a | Independent logical per-group nearest-value encoders and hand-computed ties/packing | 184,320 quantized values and scales match exactly; sanitizers clean; see [post-attention](findings/post-attention.md) |
| `kernels/nvptx/nvfp4_linear.rs` | CTA/lane coordinates, block-scaled m16n8k64 NVFP4 MMA and rounded global-factor multiply | SM120a | Independent logical f64 packed-matrix reference | 718,848 real matrix outputs pass; sanitizers clean; see [MLP](findings/qwen-mlp.md) |
| `kernels/nvptx/mlp_activation.rs` | CTA/thread coordinates, rounded FP32 product and shared Rust FP64 SiLU | SM120a | Independent f64 SiLU with explicit BF16 activation/product boundaries | Prior activation qualified in [MLP](findings/qwen-mlp.md); shared SiLU correction pending qualification in [resident model](findings/resident-model.md) |
| `kernels/nvptx/silu_probe.rs` | CTA/thread coordinate reads; shared pure-Rust FP64 range reduction and polynomial | SM120a | Exhaustive finite BF16 domain versus independent libm-backed SiLU, plus negative 1/256 midpoint regression | Pending live qualification in [resident model](findings/resident-model.md); no host table or reference execution in device code |
| `kernels/nvptx/residual_add.rs` | CTA/thread coordinates and rounded FP32 addition | SM120a | Independent scalar BF16 residual addition | 92,160 real BF16 outputs exact; sanitizers clean; see [MLP](findings/qwen-mlp.md) |
| `kernels/nvptx/attention_prepare.rs` | CTA/thread coordinates, shared per-head reduction/barriers, rounded FP32 add/subtract/multiply/divide, approximate reciprocal square root and BF16 RNE | SM120a | Independent f64 head norm, explicit BF16 RoPE products and gate-layout fixtures in `reference/attention_prepare.rs` | 129,024 real and 10,566 fixture values pass; sanitizers clean; see [attention preparation](findings/attention-preparation.md) |
| `kernels/nvptx/causal_attention.rs:attention_kv_append` | CTA/thread coordinates and guarded BF16 cache copies | SM120a | Independent append helper and poisoned-tail cache comparisons in `reference/causal_attention.rs` | Real outputs and all chunk/cache checks pass; sanitizers clean; see [causal attention](findings/causal-attention.md) |
| `kernels/nvptx/causal_attention.rs:causal_attention_bf16` | CTA/thread coordinates, FP64 shared dot reduction/barriers, rounded FP64 arithmetic, FP64-to-FP32 RNE and BF16 RNE; Rust exponential polynomial | SM120a | Independent f64 logical GQA/softmax/value oracle and whole/chunk/token equivalence | Original FP32 component qualification in [causal attention](findings/causal-attention.md); FP64 model correction awaiting qualification in [resident model](findings/resident-model.md) |
| `kernels/nvptx/attention_gate.rs` | CTA/thread coordinates, rounded FP32 multiply/add/divide and non-FTZ approximate exp2 | SM120a | Independent f64 stable sigmoid with explicit BF16 activation/product boundaries in `reference/attention_gate.rs` | 110,592 real gates and signed/subnormal fixtures pass; all sanitizers clean; see [full attention layer](findings/full-attention-layer.md) |
| `kernels/nvptx/fp8_linear.rs:fp8_linear_wide` | Existing E4M3 MMA/layout with explicit FP32/FP64 conversions and FP64 sums between K32 tiles | SM120a | Unchanged logical f64 projection and independent whole attention-layer oracle | Refined projections pass fixed complete-layer budgets and cancellation/tail fixtures; all sanitizers clean; performance cost unmeasured; see [full attention layer](findings/full-attention-layer.md) |
| `kernels/nvptx/fp8_linear_rounding.rs` | FP64-to-FP32 RNE conversion and rounded FP32 scale multiplies after exact scalar FP64 recomputation of BF16-ambiguous projections | SM120a | Unchanged logical f64 reference; cancellation/midpoint fixture and whole-layer gates | Cancellation fixture and fixed complete-layer budgets pass; all sanitizers clean; see [full attention layer](findings/full-attention-layer.md) |

## Exact FP8 decode

| Source | Instructions | Reference | Status |
| --- | --- | --- | --- |
| `kernels/nvptx/fp8_linear_exact.rs` | Thread/CTA coordinates, wide signed integer product and sum, paired-word warp shuffle, rounded i64-to-FP32 conversion and scale products | Independent decoded FP64 dots; all finite code pairs, tails, width32768 and cancellation | Independent fixture/full-model checks and all three sanitizers pass; see [decode projections](optimizations/decode-projections.md) |

| `kernels/nvptx/bf16_linear_decode.rs` | Thread/CTA coordinates, rounded FP64 product/sum, paired-word warp shuffle, rounded FP64-to-FP32 conversion | Independent sequential BF16 FP64 dot, cancellation/tails and full-model gates | Fixtures/full-model checks and all three sanitizers pass; parallel reduction is not universally bit equal to sequential FP64 |

NVFP4 logical linear loads now use aligned complete `u32` words with the existing
byte fallback for tails/unaligned scale rows. MMA operands and order are unchanged;
K16/K80 independent fixtures and full-model tests qualify this load-only change.

`nvfp4_linear_warp4` shares the existing instructions and arithmetic, with a
thread-coordinate read to select four independent N tiles per CTA. Qualified
results are recorded in the decode projection optimization entry.


## Attention reduction and scalar broadcast

`attention_reduction.rs` uses paired `shfl.sync.down.b32` with full-warp clamp
0x1f after the first three exact-order shared reduction levels. It preserves the
old FP64 addition tree and reduces CTA barriers. Causal attention evaluates its
unchanged online softmax scalars on thread zero and broadcasts alpha, beta and
normalizer through disjoint shared slots. Full-model equivalence and sanitizer
qualification are required; status is recorded with the performance iterations.

The `nvfp4_linear_warp4` experiment was rejected after unchanged timings. Its
entrypoint is removed; the candidate commit and raw evidence preserve the trial.

`fp8_linear_exact4.rs` reuses the qualified signed-integer dot instructions and
paired-word warp reduction across four activation rows per weight load. Its only
new assembly site reads CTA/thread coordinates. Both variants run independent
finite-code/tail/cancellation fixtures; retained results are in the optimization log.

The reduction's final form keeps shared loads/stores, exact FP64 tree additions,
paired-word shuffles and both barriers in one opaque inline PTX block. This avoids
LLVM branch threading without a device function call. Existing logical attention
fixtures and the full-model comparison remain the independent oracle. The interim
out-of-line variant passed sanitizers but failed the profiler's global free-memory
gate. Stack allocation was suspected, not established; the final inline variant
passes both profiles and all sanitizers. See the preserved failed evidence.

## Dedicated NVFP4 decode

`nvfp4_decode.rs` reads lane/CTA coordinates and reuses the qualified NVFP4 MMA,
packed loaders and output conversion with weight/activation operands transposed.
Independent signed/tail and whole-model checks pass, but model gain was below 1%.
The resident path selects the grouped-dot candidate instead; see
[dedicated decode](optimizations/dedicated-decode.md).

`nvfp4_decode_exact.rs` adds packed signed-byte DP4A and coordinates, reusing the
qualified i64 warp reduction and conversion helpers. Reference: independent
logical NVFP4 oracle and [NVIDIA DP4A semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a).
Independent exact output fixtures, full-model state/logits and all three sanitizers
pass. The dedicated-decode record contains hashes and retained timings.

- `kernels/nvptx/fp8_prefill_exact.rs`: exact base-128 decomposition
  uses the already-qualified `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
  instruction from `ordinary_mma.rs`. Four signed64 output sums reconstruct nine
  digit-pair products per K32. Finite-code/tail/max-width independent probes cover
  the new tile. Exact full-model/512-token partition and all sanitizer checks pass; see
  [larger prefill](optimizations/larger-prefill.md).

The MTP verification candidate adds `kernels/nvptx/fp8_verify_exact.rs`: the same
nine exact signed INT8 MMA digit products as the 16x8 prefill tile, with weights in
operand A and activations in operand B. Its tile covers eight input rows and 16
output channels; stores transpose the accumulator mapping back to row-major
output. Independent FP8 fixtures cover every finite code pair, signed M/N/K tails,
maximum K and cancellation. GPU qualification and resource results are pending in
[the MTP continuation](optimizations/mtp.md).

F01 experimental `fp8_a16_decode.rs` adds packed global loads, FP32 FMA and
warp-shuffle reduction for BF16 activations/E4M3 weights. Independent host oracle
and ABI are in `optimizations/feature-f01-decode.md`; resident dispatch stays
unchanged pending GPU and model qualification.

F02 experimental `fp8_native_prefill.rs` adds native SM120a FP8 MMA, shared
loads/stores and `cp.async` staging with explicit waits/barriers. The independent
FP64 oracle and 32x64x64 launch contract are in `optimizations/feature-f02-prefill.md`.
This first tile uses one shared buffer; double-buffer overlap is not implemented.
No resident dispatch change or throughput claim follows from PTX compilation.

F03 experimental `fp8_swiglu_exact.rs` shares activation decoding across exact
gate/up dots and preserves every BF16 epilogue boundary. Coordinate reads and
existing exact integer arithmetic PTX are covered by the independent composed
reference in `optimizations/feature-f03-fusion.md`. GPU qualification is pending.

F04 experimental `gdn_chunked.rs` implements coefficient/RHS preparation,
triangular solve and output/final-state reconstruction using explicit FP32
arithmetic. Its independent sequential FP64 oracle and supported contracts are
in `optimizations/feature-f04-gdn.md`; GPU/model qualification remains pending.

F05 experimental `attention_online.rs` stages BF16 KV tiles in shared memory,
uses FP32 reductions and online softmax, and preserves causal/GQA addressing.
The independent FP64 oracle, fixed budgets and ABI are documented in
`optimizations/feature-f05-attention.md`. GPU/model qualification is pending.

F06 experimental `kv_fp8.rs` uses CTA shared reductions and explicit FP16,
BF16 and FP32 conversion/arithmetic instructions. The independent logical codec
reference and invalid-row status contract are in `optimizations/feature-f06-kv.md`.
GPU and cache-attention qualification remain pending.

F02 K64 experiment adds explicit scalar `add.rn.f32` in
`fp8_native_prefill.rs::fp32_add_rn`, combining independent K64 native MMA
partials. Existing shared-copy/barrier/fragment instructions and epilogue remain
shared with the original entry. Independent reference is
`reference/fp8_native_prefill.rs`; both native entries run the same finite-code,
signed/tail/cancellation fixtures without relaxing budgets. NVIDIA PTX FP8 MMA
rounding/order is unspecified. Compilation passes; GPU/model qualification pending.

F09 `gdn_replay.rs` records GDN deltas with the existing explicit
`mul.rn.f32`/`sub.rn.f32`/`add.rn.f32` order, then replays accepted
rows from a retained base using rounded decay and key/update terms. Coordinate
reads and BF16 conversion have independent `reference/gdn_replay.rs` coverage,
anchored against the original ordered recurrence for each accepted prefix.
See `optimizations/feature-f09-mtp-recovery.md` for all pointer/shape contracts.
GPU/model integration and qualification remain pending.


F09 parent GPU qualification now passes exact record and accepted-prefix replay
for widths 1/2/128; all three sanitizer tools are clean. See
`evidence/iterate-20260927/features-f09-1/` for the pinned source/PTX and resources.
This supersedes the primitive's pending-device status above; whole-model recovery
is still pending.


F02 exact split-K experiment in `fp8_verify_exact.rs` factors the existing
nine-INT8-MMA/i64 dot reconstruction into a shared range function. New
`fp8_verify_splitk` reads CTA z to select disjoint K32 ranges and writes i64
partials. `fp8_verify_reduce` reads CTA/thread x, sums partials with existing
`add.s64`, and uses the unchanged i64-to-FP32/scale/BF16 epilogue. The unchanged
independent logical FP64 `projection_reference::linear` validates both raw FP32
and BF16 through `fp8_exact_trial`: all finite code pairs, tails, cancellation,
maximum K and empty split ranges at 2/4/8/16 splits. The full absolute dot bound
is below 2^51 for K<=32768, so partial ordering cannot overflow i64 or introduce
floating reassociation. PTX compilation passes; GPU/model qualification pending.

Split-K follow-up: `evidence/iterate-20260927/splitk-check-1` qualifies all 48
independent raw-FP32/BF16 cases under normal execution and memcheck/racecheck/
synccheck, with zero errors/hazards. `splitk-sweep-1` additionally verifies full
model output/state and forced acceptance/rejection for each 2/4/8/16-way split.
Source `a06b71bfa`; PTX SHA256
`e18df0fb02524af65d00e3b313183da18ce549fce93a65a6a22011ccfc3c6136`.


Exact GPU greedy reduction: `greedy_bf16.rs` has ten assembly sites, fully
listed by function in [GPU greedy inventory](optimizations/gpu-greedy.md#complete-per-assembly-site-inventory-for-parent-integration).
They cover thread/CTA coordinates, two-word warp shuffle for integer winner
keys, nonfinite-index shuffle, 64-byte shared storage declaration/address,
64-bit key and 32-bit invalid-index shared loads/stores, and CTA barrier.
Independent `reference/greedy_bf16.rs` uses direct FP32 comparison and first-index
ties; GPU `greedy-check` additionally compares existing CPU sampling and exact
nonfinite positions. This path changes only token selection, not model arithmetic.
Compilation and GPU qualification are pending; default model execution stays CPU
selection, with full-logit diagnostics retained even when experimenting.


Greedy follow-up: `evidence/iterate-20260927/greedy-check-1` passes77 independent
cases under normal execution and all three sanitizer tools (zero errors/hazards).
Source `11b84609f`; tile/finish24/20registers,64sharedbytes each,0localbytes.
Whole-model selected-token execution and throughput remain under qualification.


## Deep-dive follow-up candidates (not resident dispatch)

`fp8_a16_head.rs` adds eight assembly sites: coordinates, partial_base,
store_partial, load_partial, add_rn, scale_rn, BF16m16n8k16mma, and a CTAbarrier.
Complete site contracts and independent tests are in
[the A16 head inventory](optimizations/a16-head-schedule.md#ptx-inventory-handoff).
`reference/fp8_a16_head.rs` uses logical FP64 products, not fragment execution.
Parent checked A/B/C mappings against NVIDIA PTX ISA9.4 section9.7.16.5.8;
GPU asymmetric fixtures, resource inspection and sanitizers remain required.

`nvfp4_prefill_tiled.rs` adds six sites: coordinate/shared declaration,
cvta+cp.async four-byte/zero-fill copies, commit_group, wait_group0, CTAbarrier,
and shared word load. Complete bounds and publication/reuse contracts are in
[the NVFP4 pipeline inventory](optimizations/nvfp4-prefill-pipeline.md#ptx-inventory-for-parent-integration).
It reuses the previously inventoried native NVFP4 MMA and output-scale sites.
`reference/nvfp4_prefill_tiled.rs` invokes the independent logical decoded-product
reference with bounded candidate shapes. Neither source nor emitted PTX is GPU
qualification, and neither candidate changes current resident dispatch.

## NVFP4 wide prefill experiment

`nvfp4_prefill_wide.rs` adds six assembly sites: shared declaration and coordinates,
CTA barrier, async wait, four-byte zero-filling async copy with global conversion,
async commit, and shared word load. Native block-scaled MMA and BF16 rounding/store
reuse the inventoried `nvfp4_linear.rs` helpers. Two 5760-byte stages are disjoint;
eight warps reuse each A fragment for four adjacent output fragments.
`reference/nvfp4_prefill_tiled.rs` supplies the independent decoded-product FP64
oracle under identical admission bounds. `nvfp4_pipeline_trial.rs` compares both
CTA schedules against that oracle and requires raw/BF16 bit identity with native
`nvfp4_linear`, including N120/128/136 tails. Compilation, GPU resources and
sanitisers remain pending for this new symbol; prior tiled evidence does not qualify it.

Wide follow-up: `evidence/iterate-20260927/nvfp4-pipeline-check-3` at
`475c1b35f` passes33 cases and all three sanitizers with zero errors/hazards.
JIT:62 registers,11520 shared bytes,0 local bytes. Raw/BF16 native-baseline
identity holds for tested fixtures; this does not establish model throughput.

## Teacher-forced row log-prob/top-64 (`kernels/nvptx/row_logprob_topk.rs`)

Entry `row_logprob_topk_bf16`: one 256-thread CTA per logits row. Inline sites are
`mov.u32 %tid.x`/`%ctaid.x` reads, one `.shared` scratch declaration with
`cvta.shared.u64`, `bar.sync 0`, and generic `atom.add.u32` into that shared
scratch. No tensor-core, async-copy or global atomic site. `logprob_math.rs` is
pure Rust (FP32 exp, FP64 ln). Independent oracle: `reference/row_logprob_topk.rs`
(FP64 over the same BF16 logits). The scorer also checks the first four rows of
the first chunk against that oracle on GPU. GPU execution, JIT resources and
sanitizers are pending.

## BF16 split attention v2 (unqualified)

`kernels/nvptx/attention_split_math.rs` adds nine sites: coordinates, static
16KiB shared declaration/address, CTA barrier, shared word store/halfword load,
full-warp butterfly shuffle, approximate exp2, BF16 RNE conversion and FP32
division. Partial/reduce sources only call these helpers. Complete ABI, ownership,
reference and per-site contracts: [attention v2](optimizations/attention-v2.md#complete-assembly-inventory).
Independent FP64 oracle semantics are unchanged. No PTX/GPU evidence yet.

## BF16 paired A/B FP32 candidate (unqualified)

`kernels/nvptx/bf16_ab_decode_fp32.rs`: special-register coordinates; aligned
`ld.global.v4.u32`; `fma.rn.f32`; `add.rn.f32`; full-mask butterfly shuffle;
16-byte shared declaration/address; shared store/load; CTA barrier. Independent
logical FP64 oracle: `reference/bf16_ab_decode_fp32.rs`. ABI and admission
contracts: [BF16 A/B candidate](optimizations/decode-a16-pack.md). Rust PTX
compilation and host checks pass locally; GPU results are pending.

## Native encoded embedding and F32 GDN parameters

`fp8_embedding_gather` and `gdn_gates_f32_params` add entry points but no new
inline assembly sites. They reuse inventoried indexing/rounding helpers from
`embedding_norm.rs` and `gdn_prepare.rs`; the existing BF16 gate entry calls the
same factored arithmetic body with BF16 loads. Independent oracles are
`reference/fp8_embedding_gather.rs` and `reference/gdn_gates_f32_params.rs`.
See [consumer contract](optimizations/ninfer-parameter-consumers.md) for parameter
order, round-before-norm boundary, alias contract and the nine-case harness.
Host tests and Rust PTX compilation pass; GPU qualification is pending.
