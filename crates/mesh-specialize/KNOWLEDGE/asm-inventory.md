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
| `kernels/nvptx/native_mtp_q8.rs::native_mtp_q8_gemv` | Parent-row-mapped signed Q8 GEMV with FP16 group-32 scales and BF16 input | SM120a | Independent FP64 scalar oracle in `reference/native_mtp_q8_gemv.rs` | Synthetic operator test wired; parent build and GPU qualification pending |
| `kernels/nvptx/native_mtp_q8_sliced_k_fc.rs` and private instruction/staging modules | FC C4/C8 signed-code conversion, shared allocation, global address conversion, cp.async ca/cg16, commit/wait, CTA barriers, shared scalar/vector loads/stores, global.nc scale loads, indexed shuffle, ldmatrix.x2, BF16 m16n8k16 MMA, FP16 conversion, FP32 FMA/add and BF16 RNE | SM80+ | Pinned Ninfer e31bc99b sliced-K schedule and independent scalar FP32/FP64 reference in `reference/native_mtp_q8_sliced_k_fc.rs`; scalar summation does not certify internal MMA order | Synthetic-only Trial B PTX `850a8b0e8e939576...` passes six C4/T1 and C8/T5 cases with two repeats each in normal execution, memcheck, racecheck and synccheck; zero sanitizer errors/hazards. Trial A's 5,504 synccheck errors were fixed by a uniformly predicated store with an unconditional barrier. No real-weight, model, MTP runtime admission, or throughput claim; see [Q8 FC qualification](findings/native-mtp-q8-fc-qualification.md) |
| `kernels/nvptx/native_mtp_q4.rs::native_mtp_q4_head_gemv` | Experimental four-row/CTA, one-warp/row Q4 GEMV with 8-code packed-word lanes, FP16-mantissa decode, `cp.async.ca` vector16 code and pair32 scale staging, ordered FP32 FMA, five-step warp reduction, and BF16 logits | SM120a | Independent FP64 scalar projection plus separate exact FP32 lane/reduction schedule oracle in `reference/native_mtp_q4_gemv.rs` and `native_mtp_q4_operator/schedule_reference.rs` | Uncommitted Q4-only snapshot: seven synthetic cases pass normal execution and memcheck/racecheck/synccheck with zero issues; exact FP32/BF16 schedule, repeats and mapping pass; unchanged FP64 bound 2e-4; real-weight and complete native MTP qualification remain open; see [Q4 qualification](findings/native-q4-operator-qualification.md) |

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

## Device-position exact graph entries

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `graph_position.rs:attention_qk_prepare_position` | No new inline assembly; device u32 load and BF16 pointer offset, then `attention_prepare::prepare_body` reuses CTA/thread reads, shared reduction/barriers, rounded FP32 operations and BF16 RNE listed above | `reference/graph_position.rs` base/compact positional tables and independent `reference/attention_prepare.rs` norm/RoPE/gate oracle | Source only; PTX, device equivalence and sanitizers pending |
| `graph_position.rs:attention_kv_append_position` | No new inline assembly; device u32 load then `causal_attention::append_body` reuses CTA/thread reads and guarded BF16 copies | `reference/graph_position.rs` nonzero/final-position append and poison-tail checks; independent causal-attention append reference | Source only; qualification pending |
| `graph_position.rs:causal_attention_bf16_position` | No new inline assembly; device u32 load then `causal_attention::attention_body` reuses original FP64 arithmetic, shared/shuffle reduction, barriers and conversion sites | Independent logical FP64 GQA/softmax reference; full-model eager-stream versus graph token/logit/all-state comparison hook | Source only; qualification pending; see [stream graph scope](optimizations/stream-forward.md#whole-model-exact-graph-implementation-2026-09-28) |

The original entry symbols call these same extracted bodies, with their original
by-value position and sliced RoPE arguments. This is an arithmetic-preserving
source refactor, not a claim that compiler emission or runtime results are equal.
Required target remains SM120a. Parent must regenerate PTX and run both controls.

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


## Three-stage FP64 attention

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `attention_staged_fp64.rs:coordinates` | `%ctaid.x`, `%ctaid.y`, `%tid.x`, `%ntid.x` reads; score grid[length,24], scan grid24, value grid[24,4] | Pure fixed-geometry/overflow/workspace-prefix tests in `attention_staged_plan` | Unqualified, SM120a; [contract](findings/attention-staged-fp64.md) |
| `attention_staged_scores_fp64` | Reuses warp local-tree and paired-word DOWN/IDX FP64 shuffles; rounded multiply and exact BF16 decode; no CTA barrier | Original attention raw FP32/BF16 bits, existing independent FP64 oracle, wide-exponent fixture | Unqualified; one score per head/key, no split/reassociated dot |
| `attention_staged_coefficients_fp64` | Lane-zero-only ascending scan, existing explicit rounded FP64 arithmetic and unrolled exponential; f64 global stores to disjoint alpha/beta/norm regions | Strict original-control output bits; initialized-prefix/suffix and long/short/poison reuse checks | Unqualified; normalizer never parallel-associated |
| `attention_staged_values_fp64` | Ascending-key FP64 accumulator, separate rounded mul/add/div and original output conversions; f64 coefficient loads | Same strict control/oracle gates; three stages on one stream, guards/repeats/readback | Unqualified; no per-key CTA barrier or duplicated exponentials |
| `attention_staged_max_prefix_fp64` | Per-head ascending FP64 maximum prefix using the serial comparison order; writes only the initialized capacity-strided prefix | Host serial/prefix bit recurrence plus independent FP64 attention oracle and strict GPU control when run | Prefix-parallel candidate unqualified; same-stream predecessor scores |
| `attention_staged_coefficients_parallel_fp64` | Per-key FP64 exponentials from adjacent running maxima; one thread owns each key's alpha/beta | Host serial/prefix bit recurrence plus independent FP64 attention oracle and strict GPU control when run | Prefix-parallel candidate unqualified; one writer per coefficient pair |
| `attention_staged_normalizer_fp64` | Ascending FP64 normalizer recurrence with explicit rounded multiply/add and serial-compatible alpha zero correction | Host serial/prefix bit recurrence plus independent FP64 attention oracle and strict GPU control when run | Prefix-parallel candidate unqualified; normalizer is not parallel-reduced |

## Isolated unrolled exponential

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `exponential_unrolled.rs:exp_nonpositive/horner` | No new inline assembly; reuses explicit `mul.rn.f64`, `add.rn.f64`, `sub.rn.f64` with17 scalar literal-coefficient Horner stages and original range/factor order | Unchanged device exponential's raw FP64 bits, dense/cutoff/range-boundary/NaN fixtures, supplemental host order and coefficient-bit checks | Unqualified; [contract and observed control traffic](findings/attention-unrolled-fp64.md) |
| `exponential_probe.rs:index` | `%ctaid.x`/`%tid.x` reads for independent count-bounded raw-bit output entries | Explicit raw-u64 inputs and old/new output bits, guards, repeats and immutable input readback | Unqualified; SM120a |
| `attention_unrolled_fp64.rs:causal_attention_unrolled_fp64` | Reuses original CTA256 shared/shuffle reduction, barriers, rounded FP64 recurrence and conversions; true const specialization changes only exponential helper | Strict all-FP32/BF16 control bits plus unchanged independent FP64 attention oracle, M1/17/128 tests | Unqualified; original and graph entries explicitly retain false specialization |

## Exact-order FP64 warp attention

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `attention_warp_fp64.rs:coordinates` | Reads `%ctaid.x`, `%tid.x`, `%ntid.x`; fixed grid96/block32 admission | `attention_warp_plan` geometry tests and exact source geometry checks | Unqualified; [contract](findings/attention-warp-fp64.md) |
| `attention_warp_fp64.rs:warp_dot` | Paired-word `shfl.sync.down.b32` offsets16/8/4/2/1, `add.rn.f64`, paired-word `shfl.sync.idx.b32` lane-zero broadcast, register moves/loop predicates; no shared memory or barriers | Separate host 256-slot tree and lane-stripe simulators, existing logical FP64 oracle, strict GPU control-bit comparison | Unqualified; no device measurements |
| `attention_warp_fp64.rs:local_tree/attend` | Reuses only existing rounded FP64 arithmetic, exact BF16 decode, FP64-to-FP32 and BF16 RNE helpers in `causal_attention.rs`; unchanged Rust exponential | Same strict control and independent oracle gates, ascending-key recurrence and signed/wide-exponent fixtures | Unqualified; larger rows and graph path retain baseline |

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

## BF16 paired A/B FP64 candidate (unqualified)

`kernels/nvptx/bf16_ab_decode_fp64.rs` adds eleven inline-asm sites:
`coordinates` (tid/CTA reads), `load8` (`ld.global.v4.u32`), `widen`
(`cvt.f64.f32`), `fma` (`fma.rn.f64`), `add` (`add.rn.f64`), `shuffle`
(full-mask `shfl.sync.bfly.b32`, applied to both FP64 words), `shared_base`
(32-byte aligned shared declaration/address), `store_partial`/`load_partial`
(`st.shared.f64`/`ld.shared.f64`), `narrow` (`cvt.rn.f32.f64`), and the
kernel's `bar.sync 0`. Independent oracle remains
`reference/bf16_ab_decode_fp32.rs`; the extended trial requires exact FP32/BF16
bits against that oracle and the unchanged control on all bounded fixtures.
Per-site evidence and reassociation limits are in
[the FP64 section](optimizations/decode-a16-pack.md#paired-fp64-continuation).
PTX/JIT, GPU, sanitizer, and actual-model qualification are pending.

## Exact FP8 vector16 decode (unqualified)

`kernels/nvptx/fp8_linear_exact_vector16.rs` adds two inline sites:
`coordinates` reads `%laneid`, `%tid.x`, `%ctaid.x`; `load16` issues readonly
`ld.global.v4.u32` for a fully readable aligned 16-byte global span. Integer
products/adds, full-mask paired-word shuffles, rounded conversion and scale
products reuse the unchanged helpers inventoried in `fp8_linear_exact.rs`.
Independent oracle: `reference/projections.rs::linear` (decoded logical FP64).
ABI, bounds, admission, complete fixture coverage and reproduction command:
[vector16 decode](findings/fp8-exact-vector16.md). PTX compilation, JIT resources,
GPU/sanitizer and full-model evidence are pending; no measured speedup.

## Native encoded embedding and F32 GDN parameters

`fp8_embedding_gather` and `gdn_gates_f32_params` add entry points but no new
inline assembly sites. They reuse inventoried indexing/rounding helpers from
`embedding_norm.rs` and `gdn_prepare.rs`; the existing BF16 gate entry calls the
same factored arithmetic body with BF16 loads. Independent oracles are
`reference/fp8_embedding_gather.rs` and `reference/gdn_gates_f32_params.rs`.
See [consumer contract](optimizations/ninfer-parameter-consumers.md) for parameter
order, round-before-norm boundary, alias contract and the nine-case harness.
Host tests and Rust PTX compilation pass; GPU qualification is pending.

Native-consumer qualification update: direct-source-1 and native-final-checks-1 pass all nine parameter cases, same-source model checks and three sanitizers. Racecheck used forced synchronization. All1589 canonical source and GPU tensor hashes match independent evidence; old raw-mspec33-step regression also passes. See optimizations/direct-ninfer-source.md for bounds and source/PTX pins.

## Exact NVFP4 PRMT decode candidate

`nvfp4_decode_exact_prmt` changes only E2M1 packed expansion in the existing
integer-group-dot schedule. Two generic `prmt.b32` instructions per expansion;
coordinate reads and four DP4A group instructions retain the control contract.
Integer scale/reduction and FP32/BF16 epilogue use existing inventoried helpers.
Independent reference/proof: `reference/nvfp4_decode_prmt.rs`; 65,536 controls,
524,288 halfword/neighbour combinations and explicit negative-zero cases.
Exact ABI, bounds and proof hashes are in evidence/nvfp4-prmt-host-proof-20260928.json.
Host tests, Linux-target Clippy and Just PTX compilation pass; GPU/resource/model
qualification is pending. Default remains baseline.

## NVFP4 A16 fused SwiGLU candidate

`kernels/nvptx/nvfp4_swiglu_a16.rs::nvfp4_swiglu_a16` reads lane/thread/CTA
coordinates, performs FP32 `fma.rn`, `mul.rn`, and `add.rn`, reduces each warp
with full-mask `shfl.sync.down.b32`, calls the shared Rust SiLU arithmetic, and
writes separate gate/up BF16 and raw FP32 values plus the FP32 SwiGLU product
and BF16 activation. It directly reads BF16 activations, E2M1 packed weights,
and E4M3 K16 scales. The independent logical FP64 dot/SwiGLU oracle is
`reference/nvfp4_swiglu_a16.rs`; the real/synthetic harness is
`src/kernels/cuda/nvfp4_swiglu_a16_operator/`. The explicit
`MESH_SPECIALIZE_NVFP4_MLP_SCHEDULE=a16-swiglu` dispatch is legacy-only, one
row, and `past > 0`. Source/PTX inventory is not compilation or GPU
qualification. Host build, PTX, real/synthetic GPU cases, three sanitizers,
whole-model quality, and timings are pending; baseline remains the default.
See [NVFP4 A16 SwiGLU](findings/nvfp4-a16-swiglu.md).

## Native packed MTP continuation

The earlier synthetic-only FC and Q4 entries are historical. Real packed FC
now passes sparse and bounded dense, all-row BF16 comparisons with two poison
repeats under normal execution and all three sanitizers. See
[resident FC evidence](findings/native-mtp-q8-fc-resident.md). Real Q4 now passes
all 131072 proposal rows against independent scheduled FP32/BF16 and FP64
references, with two poisons and all three sanitizers. Its signed map remains
separate from the 248320-token target vocabulary. See
[resident Q4 evidence](findings/native-q4-operator-qualification.md).
Neither result qualifies arbitrary activations, complete native MTP or model
throughput.

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `native_mtp_q8_projection/instructions.rs` | CTA/thread coordinates; CTA barrier; async commit/wait; shared halfword, word and vector FP32 loads/stores; global noncoherent scale load/address conversion; full-mask indexed shuffle; signed-code/BF16 and FP16 conversions; ldmatrix.x2; BF16 m16n8k16 MMA; FP32 FMA/add; uniformly predicated odd-warp store/barrier; BF16 RNE | Independent `reference/native_mtp_q8_projection.rs`, bounded half-unit G32 dot proof, scale-FMA and ordered four/eight-split reduction; mutation tests in the host projection module | Source reviewed; host build, Rust PTX, all-row real-parent execution and sanitizers pending |
| `native_mtp_q8_projection/staging.rs` | Four/eight-split shared allocation/address conversion; aligned vector16 cp.async.cg code copies, cp.async.ca activation/scale copies and zero-fill | Same independent projection reference, complete parent planes, five distinct columns and exact all-row two-poison gates | Unqualified; physical QKV/gate-up/output/down parents, not substitute logical-row geometry |

The bounded reference proves exact representability of each unscaled G32 dot
for these fixtures. It does not certify internal MMA summation for arbitrary
BF16 activations. Scale-FMA order and split reduction remain part of the gate.

Q8 execution update: Rust PTX and host builds pass. All 16 real physical-parent
cases pass exact all-row two-poison BF16 checks under normal execution,
memcheck and synccheck. Racecheck hit the 1200-second bound without a completed
report and is not qualified. The table's original pending status is historical;
see [Q8 projection evidence](findings/native-mtp-q8-projections.md) for hashes,
retained failures and the remaining partitioned racecheck gate. No arbitrary
activation, complete native MTP or model-throughput claim follows.

Partitioned Q8 racecheck recovery now passes all original indices 0 through 15
exactly once in eight bounded processes. Every process exits 0 with zero hazards,
errors or warnings and the unchanged all-row two-poison BF16/hash gates pass.
PTX is unchanged; the case-selection executable and source are separately pinned
in trial C. The full-matrix timeout and interrupted trial remain preserved.
This closes bounded operator sanitizer coverage, not arbitrary-activation,
complete native MTP or whole-model performance qualification.

## Native MTP activation kernels (unqualified)

| Source symbol | Instructions / reused sites | Reference | Status |
| --- | --- | --- | --- |
| `kernels/nvptx/mtp_source_ops.rs::native_mtp_attention_gate` | `block_and_thread` reads `%ctaid.x` and `%tid.x`; exact BF16-to-FP32 bit decode; `source_exp` uses `ex2.approx.ftz.f32` on `value * LOG2_E`; `source_sigmoid` uses `div.rn.f32`; `multiply_rn` uses `mul.rn.f32`; `encode_bf16` uses `cvt.rn.bf16.f32`; count-guarded BF16 write | Independent mathematical FP64 sigmoid/product and direct BF16 RNE in `reference/native_mtp_activation.rs`; trial in `src/kernels/cuda/native_mtp_activation_trial.rs` | Inventory only; host build/tests, device PTX/JIT, GPU execution/resources/sanitizers, pinned-source arithmetic and full-model qualification pending |
| `kernels/nvptx/mtp_source_ops.rs::native_mtp_silu_mul` | Reuses the same coordinate, BF16 decode, exponential, multiply and BF16 RNE sites; `source_silu` uses its own `div.rn.f32` site; count-guarded BF16 write | Independent mathematical FP64 SiLU/product and direct BF16 RNE in `reference/native_mtp_activation.rs`; same activation trial | Inventory only; host build/tests, device PTX/JIT, GPU execution/resources/sanitizers, pinned-source arithmetic and full-model qualification pending |

Both entries require NVPTX with BF16 RNE conversion support and a 256-thread
CTA. Each lane uses `block * 256 + thread`, returns before input reads or output
writes when `index >= count`, and otherwise owns one output element. Equal-sized
BF16 input/output arrays must cover `count`. There are no shared-memory,
barrier, shuffle, tensor-core or async-copy sites in this source.

The device computes sigmoid or SiLU in FP32, multiplies the activation by the
decoded BF16 factor with `mul.rn.f32`, then rounds that single product to BF16.
It does not round the activation separately to BF16. The exponential scaling
multiply and denominator addition are Rust FP32 expressions, not additional
explicit rounded inline-PTX sites. `source_exp` is approximate and FTZ; its name
does not establish identity with pinned Ninfer `expf` or binary equivalence.

The independent reference instead uses stable FP64 sigmoid/SiLU, an FP64 product,
and direct FP64-to-BF16 RNE without an intermediate FP32 rounding boundary.
Its signed-zero, midpoint-sensitive and bounded finite-BF16 fixtures test the
mathematical single-product rounding contract, not the device exponential's
instruction schedule or pinned Ninfer arithmetic. The present trial requests
exact BF16 comparison with zero tolerance and two distinct output-poison repeats
for both entries. These are trial requirements, not observed results. No build,
host test, emitted PTX, GPU or sanitizer evidence was collected for these kernels
in this inventory update. Complete native MTP, full-model quality/performance,
source equivalence and runtime admission remain unqualified; prior packed
FC/Q4/Q8 operator results do not qualify these activations.
