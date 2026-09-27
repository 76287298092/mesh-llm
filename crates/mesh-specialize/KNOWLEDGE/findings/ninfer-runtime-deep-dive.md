# Why the resident runtime remains far behind Ninfer

Status: source investigation, September 27, 2026. No new GPU measurement or
implementation was made by this worker. The parent fetched the complete Ninfer
working source tree, rather than the earlier partial archive. This report uses
Ninfer `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d` and mesh-llm
`28bbea8de136e943f8f5dee3a4a1ac531e155b80`. Pending parent F03 probe edits are
excluded. Ninfer's installed binary is still not connected to either source pin.

Our runtime has working instructions and independently checked operators, but its
execution still resembles a sequence of separately completed operator trials.
Ninfer plans a whole GPU round. It keeps temporary addresses stable, combines
projections and epilogues, chooses different arithmetic and kernels for different
shapes, keeps selection on the GPU, and replays the prepared round. Its prefill
also uses large pipelined matrix tiles and parallel chunk algorithms where our
resident path still uses small tiles or serial recurrence. These are several
structural differences. A special file extension does not explain them.

The strongest immediate experiment is a resident MLP workspace and stream conversion
with unchanged arithmetic, followed by exact GPU argmax. The strongest remaining
kernel experiment for verification is small-batch A16 matrix execution, including
the output head. For prefill, the missing large NVFP4 tiles and chunked GDN deserve
as much attention as FP8. The exact control should remain available, but requiring
every experimental floating-point schedule to reproduce its integer-dot bits
would select a different optimization objective from Ninfer's.

## Reference convention and evidence limits

`N:path:lines` below means a file at the immutable
[Ninfer source pin](https://github.com/Neroued/ninfer/tree/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d).
The local read-only checkout is `target/specialize/ninfer-deep-dive`.
`M:path:lines` means a path relative to `crates/mesh-specialize` at the immutable
[mesh source pin](https://github.com/Mesh-LLM/mesh-llm/tree/28bbea8de136e943f8f5dee3a4a1ac531e155b80/crates/mesh-specialize).
Line references name the actual consumers, not only public declarations.

Direct starting points are the [Rust projection allocation/wait](https://github.com/Mesh-LLM/mesh-llm/blob/28bbea8de136e943f8f5dee3a4a1ac531e155b80/crates/mesh-specialize/src/kernels/cuda/resident_fp8.rs#L96-L161),
[Ninfer ordinary round](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/program/decode.cpp#L24-L78),
[MTP round](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/program/speculative/mtp.cpp#L71-L210),
[sliced-K FP8 consumer](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/ops/linear/fp8/fp8_a16_sliced_k_mma.cuh#L70-L245),
and [NVFP4 TMA pipeline](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L130-L312).

The earlier reports used Ninfer `9e163eee4b8acec21ab0ac765107b6a3f287b217`.
The new checkout has substantial changes to linear scheduling and program code.
For example, the current output-head kernel uses A16 sliced-K, and the current
NVFP4 fused MLP chooses a TMA route at 256 tokens. Do not silently attribute those
new-pin mechanisms to the previously measured executable.

The historical deployed Ninfer rates, 164.2–201.0 decode and 5,956–10,416 prefill
tokens/s, used MTP4, FP8 KV, capacity two, prefill chunk 2,048, and 512 requested
outputs. Our recent rates use different prompts and shorter contexts. These are
context for the investigation, not matched ratios or per-feature speedups.
See [baseline](ninfer-baseline-20260926.md) and
[format/provenance audit](ninfer-model-format.md).

## What the current measurements actually say

I read the saved JSON, grouped events by kernel name, and summed their counts
and milliseconds. These are shared-GPU diagnostic event measurements. Event
instrumentation changes execution; summed event durations are not an attribution
of an uninstrumented wall clock, and host overhead is not their simple difference.
The verification recorder calls `forward_detailed`, not `forward_recorded`. Its
counts exclude compact recovery record storage and recording overhead; actual
compact benchmark rounds do use recording. Therefore the table is a projection
execution diagnostic, not a complete compact-round cost account.

| Workload and evidence | Launches | Summed GPU event ms | Main event groups |
| --- | ---: | ---: | --- |
| One-row exact Python decode, `a16-model-2/profile-python-exact.json` | 1,476 | 31.6338 | FP8 exact 9.9059; NVFP4 decode 8.2430; BF16 A/B 2.6982; FP8 quantization 2.6074; GDN 2.4825 |
| One-row A16 Python decode, `a16-model-2/profile-python-a16-decode.json` | 1,243 | 28.6000 | FP8 A16 9.4681; NVFP4 decode 8.2644; BF16 A/B 2.7097; GDN 2.4822 |
| Five-row exact verification, split-K off, `splitk-compact-bench-1/python-off.json` | 1,476 | 86.4015 | FP8 exact4 44.9131; NVFP4 18.7004; wide FP8 verify 6.7998 |
| Five-row exact verification, split-K four, `splitk-compact-bench-1/python-4.json` | 1,692 | 63.9732 | FP8 split 21.0526 plus reduction 0.8228; NVFP4 19.2667; wide FP8 verify 6.8313; GDN 4.3306 |

Evidence paths are under `KNOWLEDGE/evidence/iterate-20260927/`. The first two
files also record unprofiled single-forward wall times 40.6652 and 37.1085 ms.
The A16 change removes 233 quantization launches, but its projection total only
falls from 9.9059 to 9.4681 ms. Thus our present A16 GEMV is not evidence that we
have copied Ninfer's entire decode design. The old 83.39% FP8 profile is obsolete
for this decision.

Compact recovery plus split-K four measured median Python 46.7013 versus 39.1273
tokens/s and prose 22.5269 versus 18.6906 over three repetitions. Parent evidence
retains exact target tokens and complete target states. Prose remains slower than
ordinary greedy decode. Native FP8 prefill measured about 445 tokens/s at 512
inputs, with unresolved numerical/quality qualification. None closes the parity
goal. See [MTP](../optimizations/mtp.md),
[F01](../optimizations/feature-f01-decode.md), and
[F02](../optimizations/feature-f02-prefill.md).

## End-to-end path map

| Stage | Ninfer current source execution | Current resident Rust execution |
| --- | --- | --- |
| Request/admission | `N:src/runtime/engine/engine_core.h:168-216,1381-1399,1999-2011` accepts prepared prompts, keeps request budgets and lanes, advances staged prefill or decode membership. `scheduler.h:240-246` alternates runnable prefill and decode rather than monopolizing all work with one prompt. | Standalone validation/benchmark commands. No equivalent model-serving request scheduler, EOS lifecycle, or concurrent lanes have been qualified. |
| Prefill | `program/prefill.cpp:1038-1100` chooses bounded prompt work; `execution/text.cpp:1167-1284` resets scratch, uploads token IDs, fills absolute positions, embeds, and runs all layers. The final chunk projects only the last normalized hidden row. One chunk completion wait is at `text.cpp:1389-1395`. | `M:src/kernels/cuda/resident_model.rs:183-278` loops all 64 blocks, then last-row head, downloads logits, and selects on CPU. Numerous operators synchronize internally. Current tested prompt lengths are much smaller. |
| Ordinary decode | `N:program/decode.cpp:24-78` stages one ingress record, runs all layers and GPU sampling, and asynchronously returns a small egress record. `decode.cpp:301-352` selects a batch/frontier graph and waits after the round. | Same general model loop as prefill, with shape-selected operators. Every step materializes a full vocabulary vector on the host for greedy selection. |
| MTP verify/propose | `N:program/speculative/mtp.cpp:71-210` captures target verification, GPU acceptance and hidden selection, batched MTP alignment, and the next autoregressive proposals. `target_verification.cpp:8-49` preserves replay records and selects accepted hidden state. | `M:src/kernels/cuda/resident_speculation.rs:296-435` forks draft and target sessions, runs proposals individually, downloads target logits, decides acceptance on CPU, and either commits or recovers. Compact recovery avoids full model replay after rejection, but full state forks remain. |
| Commit/cancel boundary | Ninfer returns licensed candidates to the frontend; `N:program/prefill.cpp:755-804` folds only the frontend-committed prefix into recurrent/conv state. EOS or cancellation can shorten that prefix. | Existing bounded trial has exact forced acceptance/rejection state evidence, but not equivalent frontend cancellation/EOS commit behavior. |
| Storage | Weights, state pool, round buffers, replay records, and workspace are planned before execution. `N:program/program_impl.cpp:154-159` binds persistent replay records/fold plan. | Weights and base state are resident. Most operator temporaries are fresh owning CUDA buffers. F07/F08 components exist but do not yet own the whole chain. |

Paths abbreviated `program/` and `execution/` in this table are beneath
`N:src/models/qwen3_5/`.

## 1. GPU work repeatedly returns control to the host

The default FP8 projection allocates activation codes, row scales, BF16 output,
and FP32 unrounded output, launches quantization and projection, then calls
`Context::synchronize`. NVFP4 allocates packed activation, group scales, FP32
effective-scale diagnostics, BF16 output, and FP32 raw output, and also completes
before returning. These are actual driver allocations, not cheap arena views.

Pinned consumers:

- `M:src/kernels/cuda/resident_fp8.rs:96-161`.
- `M:src/kernels/cuda/resident_nvfp4.rs:91-137,213-227`.
- `M:src/kernels/cuda/driver.rs:419-425,445-459,560-570` resolves these to
  `cuCtxSynchronize`, `cuMemAlloc_v2`, and buffer destruction/free.
- `M:src/kernels/cuda/resident_gdn_core.rs:264-318,346-414,439-470` adds
  allocations and synchronization at each normalization/gating/recurrence stage.

The measured five-row forward has 233 FP8 and 168 NVFP4 projections. Therefore
the split-off path performs at least 401 projection completion boundaries and
`233*4 + 168*5 = 1,772` projection-owned allocations per forward. Split four adds
one partial buffer for 216 eligible projections, raising that lower bound to
1,988 allocations. It retains 401 projection completion boundaries. These counts
exclude norms, convolution, attention, A/B, residuals, head row copies, state
forks, records, and other allocations. They are source-derived counts for the
specified profile, not measured allocator tracing and not counts for A16 mode.

Ninfer's `WorkspaceArena` aliases `DeviceArena`. Tensor allocation advances an
aligned offset into backing memory and scopes restore that offset. The enclosing
stream orders reuse. See `N:src/core/arena.h:42-90,111` and `arena.cu:130-165,196-242` allocation
implementation. `execution/ffn.cpp:67-107` borrows activation/scratch within that
arena and queues gate/up, activation, down, and residual work without operator
completion waits. `program/graph_execution.h:11-27` launches the prepared graph;
ordinary decode waits once at `program/decode.cpp:348-353`. MTP has a round wait
at lines 508-513 plus the later commit/fold boundary. It is not accurate to call
an entire frontend MTP transaction a single unconditional graph.

This is a source-proven structural mismatch. Its wall-time share needs a driver
call/timeline measurement. Graphs cannot remove tens of milliseconds of actual
matrix computation by themselves, and summed CUDA events do not establish how
much host/device idle time is recoverable.

### Exact GPU argmax and persistent output state

Our target downloads `2 * 248,320 = 496,640` bytes per logit row before CPU greedy
selection, or 2,483,200 bytes for five verification rows. Draft forward downloads
another full row per proposal. See `M:src/kernels/cuda/resident_model.rs:254-276` and
`M:src/kernels/cuda/resident_mtp.rs:172-173,348-361`, both under `src/kernels/cuda`.
Ninfer samples/argmaxes on the GPU and returns an egress record with token IDs,
counts, and next drafts, rather than all vocabulary logits. Its head output stays
resident: `N:execution/text.cpp:564-590,755-769` and
`N:program/speculative/mtp.cpp:196-199`.

Independent GPU greedy selection can preserve our exact current policy:
lowest index wins ties, signed zero ties retain first index, and any nonfinite
logit rejects the operation. That policy is at `M:src/engine/sampling.rs:3-18`.
Do not commit the session cursor until the device status confirms the entire
row is finite and selection succeeds. Retain a diagnostic command that downloads all logits. Make production execution
return device hidden/logits plus selected ID and failure status. This changes
neither weights nor arithmetic and removes an otherwise unavoidable graph break.

## 2. Exact integer FP8 and Ninfer's precision policy do different work

Our exact prefill splits each E4M3 value into three signed base-128 digits,
executes nine `m16n8k32.s32.s8.s8` MMA operations for each logical K32 product,
reconstructs exact int64 dots, and only then scales/rounds. See
`M:kernels/nvptx/fp8_prefill_exact.rs:67-127,164-168,202-241`.
Split-K changes parallelism but retains that arithmetic cost. This is an excellent
control. It is substantially more work than one native FP8 K32 MMA for the same
logical tile. Nine versus one is an instruction-count distinction, not a claimed
ninefold speed difference.

Ninfer explicitly uses different numerical routes by shape and token count:

- `[5120,17408]` uses A16 GEMV for T=1, shared-weight SIMT through T=4,
  sliced-K BF16 tensor-core paths for small larger batches, and A8 when permitted
  from T=25. `N:src/ops/linear/fp8/shapes/n5120_k17408.cu:6-52`.
- The `[248320,5120]` output head never selects A8. For T<=8 it uses a
  16-warp sliced-K A16 tile, even at one row. More rows have independently chosen
  tail and main schedules. `N:src/ops/linear/fp8/shapes/n248320_k5120.cu:12-33,38-112`.
  Our A16 experiment uses the same GEMV shape for every one-row FP8 projection,
  including that head, and leaves five-row verification exact.
- FP8 fused SwiGLU is an exception: with A8 permitted it selects A8 at T=1
  and T>=3, A16 at T=2. `N:src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:20-32`.
  A universal switch to A16 would not recreate Ninfer's dispatch.
- NVFP4 ordinary down projection uses A16 at T=1 and a dedicated five-token
  SIMT route, with A4 admitted from T=8. Fused gate/up uses A16 through T=4 and
  A4 at T=5. `N:src/ops/linear/nvfp4/shapes/n5120_k17408.cu:8-69` and
  `N:src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:23-38`.

The A16 sliced-K consumer does not decompress a full BF16 weight matrix. It
asynchronously stages packed FP8 codes and BF16 activations, swizzles shared
addresses, widens code pairs into BF16 registers, and issues BF16 MMA. Warps own
different K ranges; their FP32 partials reduce in shared memory inside the same
CTA. There is no global int64 partial matrix or second reduction launch.
`N:src/ops/linear/fp8/fp8_a16_sliced_k_mma.cuh:70-169,171-245` traces every step.
That is a concrete small-batch candidate for our remaining 21.0526 ms split
projection cost and 6.8313 ms wide verification cost. It changes numerical
association/activation precision and requires a separately named profile.

The current exact contract is not inherently required for meaningful model
quality. The existing native real-weight audit showed tiny operator error and
BF16 boundary crossings, followed by larger accumulated hidden drift. It did
not prove semantic failure, nor did short Python checks prove broad quality.
An exact-vs-native full-model logit relative L2 alone is especially weak when the
probability distribution is sharply peaked. Evaluate same-input teacher-forced
probabilities and tasks before deciding whether a changed arithmetic profile is
acceptable. Never relabel that profile exact.

## 3. Prefill still lacks the matrix reuse that dominates large batches

Ninfer's native A8 tile stages codes with aligned 16-byte `cp.async`, XOR-swizzles
16-byte shared segments by row, and reuses fragments across many token/output
subtiles. Two or more shared stages overlap fetching and computation, and
ping-pong register fragments overlap `ldmatrix` with MMA. The output epilogue
stages BF16 pairs and emits vector stores. This is directly visible at
`N:src/ops/linear/fp8/fp8_a8_mma.cuh:25-30,92-229,239-320`.
The 5120x17408 shape chooses 64-token/128-output/K128 A8 tiles beyond 128 tokens,
versus our single-buffer 32-token/64-output/K64 experimental native tile.
Our exact path is smaller still, one warp for 16x8 outputs and no shared tile.

NVFP4 is an equally serious gap. There are 56 layers with three 5120x17408-sized
MLP matrices, about 14.97 billion logical weight coefficients. Our multirow kernel
is one warp per 16-token/8-output tile, directly loading activation, weight, and
scale fragments from global memory for each K64 iteration. It has no shared
operand reuse between warps and no async staging pipeline.
`M:kernels/nvptx/nvfp4_linear.rs:3-6,200-219,233-290`.
Even a five-row verification launch computes a 16-row tile, with only five active
rows. Its measured 19.2667 ms now nearly matches the improved FP8 split group.

Ninfer's conventional A4 path stages code/scales, uses `ldmatrix`, and reuses
fragments in a multiwarp CTA, `N:src/ops/linear/nvfp4/nvfp4_a4_mma.cuh:41-157,181-274`.
Its down projection has 32x64 through 128x128 output tiles and K256 schedules.
At large rows its TMA consumer builds tensor-map descriptors, issues
`cp.async.bulk.tensor.2d` against mbarriers, and separates producer and consumer
warps with different register budgets. Fused gate/up admits that route from 256
tokens; down projection from 1024. See
`N:src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24-79,130-233,253-312` and the shape/plan
files above. This is beyond merely invoking the NVFP4 MMA instruction we already
qualified. First test ordinary multiwarp async tiles; TMA is a later independent
instruction/descriptor/lifetime gate, not a prerequisite to the first improvement.

Larger tiles reduce logical reloads, but source arithmetic does not tell us actual
DRAM traffic because L2 caching and occupancy intervene. Measure DRAM/L2 traffic,
achieved occupancy, register spills, shared usage, and kernel latency. Preserve
the exact NVFP4 K accumulation and scale boundaries where possible; tile layout
alone need not change the numerical contract.

## 4. Fused operations remove repeated work, not just calls

Ninfer's text block uses one Q/K/gate/V input entry, one GDN QKV/Z entry, a GDN
norm/control operation, fused MLP gate/up/SiLU, and output projection plus residual.
`N:src/models/qwen3_5/execution/text.cpp:848-864,922-938,970-993,1055-1074` and
`execution/ffn.cpp:67-107` identify the actual consumers.

Its Qwen27 BF16 norm/control kernel accumulates residual squared norm and A/B
projection dots together, computes inverse RMS and gate functions, and writes
disjoint normalized-hidden slices from the 48 head CTAs. It avoids a separate
normalization launch and two standalone A/B outputs.
`N:src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu:20-90`.
Crucially, it algebraically applies normalization to projection sums before
materializing normalized BF16 hidden. That is not our current BF16-boundary
contract. Copy the scheduling idea only with independent numerical qualification.

Not every Ninfer high-level fusion name means one launch. Its GDN projection+conv
plan deliberately uses fused A16 for batch 1 widths 1–3 and 7–10, materialized A16 for
widths 4–6, and A8 from width 10 when permitted.
`N:src/ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp:45-47,82-105`.
At our five-row MTP shape, blindly pursuing maximum fusion could choose the route
Ninfer itself avoids. Shape-specific measurements determine whether register
pressure and occupancy outweigh saved intermediate traffic.

Our `M:src/kernels/cuda/resident_mlp.rs:53-70` independently executes gate and up,
then activation, then down. An exact shared FP8 quantization is legal when inputs
and quantizer contracts are identical. NVFP4 additionally binds
`<projection>.input_global_scale`, `resident_nvfp4.rs:52-55`; do not share its
quantized activation unless scale bits and recipe agree. Weight global factors
remain per projection. Fusion must preserve each existing BF16 rounding boundary
in the exact experiment. The current F03 candidate covers only eight FP8 MLP
layers; it cannot on its own solve the remaining 56 NVFP4 MLPs or host round design.

## 5. Context length changes which algorithms matter

Our integrated GDN recurrence loops through every row and 128 key positions using
scalar ordered FP32 operations, `M:kernels/nvptx/gdn_recurrent.rs:141-189`.
Ninfer switches at 16 tokens to chunk 16. The preparation phase forms QK/KK products
with BF16 MMA and solves the triangular chunk interaction. The recurrence keeps
master state FP32, uses BF16 and TF32 MMA for block products, double-buffers chunk
data, and updates state once per chunk.
`N:src/ops/linear_attention/gated_delta_net/chunked/launch.h:14-41`,
`prepare.cu:83-127`, `recurrence.cu:175-335`, and
`gated_delta_net.cpp:244-275` show dispatch and arithmetic.
The F04 scalar chunk candidate is not this tensor-core implementation. A change
to chunk mathematics needs long-sequence state and quality tests; tiny recurrence
fixtures are insufficient. It cannot be assumed to reproduce sequential FP32 bits.

Our integrated causal attention still iterates visible keys one at a time per
query CTA and uses FP64 score reduction. See
`M:kernels/nvptx/causal_attention.rs:248-262,334-375`. F05 tiled attention and F06
FP8 KV are not integrated in this source pin. At long context this adds repeated
KV traversal and synchronizations inside each query CTA, even if short-prefix
profiles show only about 1.5 ms total attention.

Ninfer current FP8 prompt attention uses 64-query/64-key tiles, 512 threads, native
FP8 QK MMA, FP16/FP32 PV MMA, represented FP16 cache scales, online softmax, and
asynchronous cache loads. Its Q/K path includes a fixed D256 rotation, so simply
storing our K/V as FP8 would not reproduce this algorithm.
`N:src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh:3-5,19-44,122-218,249-409`.
Small-T attention splits the key window into partial outputs and merges them;
the dispatcher accounts for width, batch, visible-key envelope and cache format,
`causal_softmax_attention.cpp:348-379,463-487`. This exploits GQA sharing and
parallelism across a long key axis. KV format and attention algorithm need
separate ablations. Current source also has other KV formats, but their presence
is not evidence the historical FP8 baseline used them.

## Five next experiments, in recommended order

### 1. Exact MLP workspace and stream ordering

Convert one resident MLP chain first. Preallocate named regions using F07 and keep
them borrowed through the last consumer. Pass checked nonowning device views to
enqueue-only wrappers, retain the same PTX and arithmetic, and complete one lease
at the chain boundary. Keep diagnostics buffers initially to isolate allocation
and synchronization effects. Then independently remove unused raw/effective
diagnostic writes from the timed lane, retaining a diagnostic lane.

Start with one real FP8 MLP and one real NVFP4 MLP at rows 1, 5, 128, and 512. Measure current,
persistent-workspace with existing waits, and stream-ordered one-wait variants.
Instrument allocation/free counts, function lookup/argument setup, launch-submit
time, completion-wait time, host copies, and an event span around the whole chain.
Compare uninstrumented wall time in repeated paired runs separately. Require
bit-identical outputs and all retained intermediates, sanitizer coverage of reuse,
failure injection/drain, no premature region alias, stable addresses, and exact
whole-model tokens/state after integration. Sharing quantization is another
explicit ablation with the scale conditions above. This directly tests the
source-proven overhead before replacing arithmetic.

### 2. Exact GPU selection, then an ordinary one-row graph

Add independent greedy reduction with finite-status propagation and lowest-index
tie semantics. Test all-negative, ties crossing blocks, signed zero, NaN/Inf,
vocabulary tail and real logits. Retain full-logit download for qualification.
Return only ID/status in the timing lane. Measure copy bytes and CPU selection
time separately; do not claim graph gains from argmax alone.

Extend the workspace/enqueue contract through embedding, GDN, attention, norms,
residuals and head. Persistent owners are weight arena, state arena, token/position
ingress, selected-token/status egress, and workspace. Ephemeral views are projection
scratch, current/next hidden, activation, normalization and attention intermediates.
`resident_head.rs:54-56` can use a checked last-row view instead of an owning copy.
Convolution state copies must become ordered device copies or direct destination
writes, not synchronous host APIs. Diagnostic observers must stay outside capture.

First graph is ordinary single-row exact decode with fixed capacity and a bounded
position interval. Its key must include device/context/module/PTX identity,
weights, arithmetic profile and split policy, KV format, model geometry, row/batch
count, capacity/attention topology interval, and all backing addresses/lifetimes.
Dynamic token and position values live in stable input buffers. Measure eager
stream versus graph replay with identical addresses and kernels. Test interval
boundaries, repeated replay, failure poisoning, fresh sessions, and no stale KV.
Ninfer's concrete interval policy is at
`N:program/planning/graph_profiles.cpp:56-88`; its numeric thresholds are not
portable tuning constants for us.

MTP comes after ordinary replay. Keep a host acceptance boundary initially and
capture draft/verify pieces separately. Later move accepted-count/hidden selection
on device and capture a fixed maximum depth with valid-column masks. Retain the
frontend commit/fold boundary and test rejection at every position, all accepted,
budget truncation, EOS within licensed tokens, cancellation, and capacity edges.
Do not capture the current Rust `if accepted == ...` path or reuse a poisoned
workspace. Compact recovery has made progress but still allocates/forks full
state at `M:src/kernels/cuda/resident_speculation.rs:296,343` and
`M:src/kernels/cuda/resident_state.rs:36-49`, both under `src/kernels/cuda`.

### 3. Small-batch A16 matrix schedules and the vocabulary head

Implement an independent Rust FP8-code/BF16-input tensor-core tile with warp-local
K partition and CTA reduction, initially for 5 rows and the real 5120x6144,
5120x17408 and 248320x5120 shapes. Keep exact split four, existing A16 GEMV, and the new
profile separately selectable. Measure projection kernels and complete verify
rounds, resource usage, memory traffic, draft acceptance and end-to-end prose/code
rates. A head-only experiment isolates a particularly different dispatch.
Use an independent represented-A16 oracle, same-input real activations, signed
cancellation/tails, and teacher-forced logit quality. Requalify MTP within the
candidate profile before admitting it; changed shape arithmetic must not silently
break greedy target verification. Keep existing exact-profile gates unchanged.

### 4. Pipelined NVFP4 prefill and paired MLP execution

Use the current codes/scales and native NVFP4 arithmetic. Start with a multiwarp
shared tile and async double buffering at 128/512 inputs, compared against the
existing one-warp 16x8 kernel. Test the scale permutation as a separate lossless
resident-layout change with inverse-byte checks. Then pair gate/up and preserve
the exact activation boundaries. TMA is a later bounded experiment requiring
Rust driver tensor-map bindings, independent address/reference coverage, new PTX
inventory, and all sanitizers. Measure prefill separately from 5-row verification
and 1-row decode. Do not infer a decode benefit from a large-batch schedule.

### 5. Chunked GDN and tiled attention under growing context

First integrate/test the existing scalar chunk and BF16 tiled-attention candidates
to validate interfaces and state ownership. Then separately test tensor-core
chunk arithmetic and FP8 KV attention. Use prompt lengths 512, 2048, and 8192 and a
memory-admitted long target, with token/chunk partition comparisons, restart from
checkpoints, suffix equivalence, retrieval/continuation tasks, and finite bounded
state. Record throughput, TTFT, peak allocation, and quality at each length.
Increase harness limits rather than extrapolate from 512. Only after single-sequence
qualification run matched capacity-two scheduling and serving tests. Prefix reuse
must be tied to weights/profile/KV/geometry/prefix and benchmarked separately from
cold prefill. F10's generic cache is not evidence of resident suffix equivalence.

## Numerical and quality gates for a competitive profile

Keep two distinct promises. Exact mode retains current raw FP32/BF16/state checks.
An experimental floating-point mode has its own documented arithmetic and quality
contract. Its oracle checks the intended represented inputs, scaling, masking and
boundaries; it is not allowed to certify itself.

Before tuning against evaluation outcomes, freeze prompts, tokenizer/template,
weight identities, sampling/EOS policy, lengths, and an acceptance policy. Compare
teacher-forced token log probabilities/NLL, KL and probability TV over many
positions and domains, plus top-token margins and finite-state errors. Compare
exact, candidate and a pinned Ninfer run on identical token sequences where that
API is available. Free-running divergence cannot isolate a same-input operator
error. Use held-out executable code tasks, deterministic math/reference-answer
tasks, grounded prose and long-context retrieval with objective scoring; disclose
sample counts and uncertainty. A successful Fibonacci function and grammatical
prose are smoke evidence only.

Set numerical/task regression limits from the intended product requirement and
reference variability before reviewing candidate results. This report deliberately
does not invent a permissive tolerance or percentage gain. Investigate failures
by same-input layer projections and recurrence state. A small BF16 mismatch count
can amplify; a large raw logit L2 need not imply poor output. Both require evidence.

## Coverage, provenance and licensing boundaries

Read the actual text forward/ordinary/speculative paths, shape selectors, FP8
native and A16 sliced-K consumers, NVFP4 MMA/TMA scheduling, GDN chunk preparation
and recurrence, prompt/small-T attention consumers, fused plans, workspace owner,
and our current resident call chain. The engine/scheduler inspection traces the
request-to-program boundary; it is not an exhaustive audit of every cache
admission, HTTP, multimodal, MoE or DFlash branch. No exact total Ninfer launch
count, DRAM traffic, occupancy, or live selected dispatch was measured. The new
source could outperform or differ numerically from the installed baseline.

The historical artifact SHA matches Ninfer's published manifest, but its declared
quantized HF revision differs from our intake revision and historical metadata is
missing. Logical-weight equivalence remains unresolved. Its FP8 embedding,
quantized MTP and optional shortlist proposal head are representation differences from our BF16
embedding/MTP and full shared head. Resolve these before claiming matched quality
or attributing every output difference to kernels. Startup upload/format work
cannot explain already-resident steady-state timings.

Ninfer's root `LICENSE` at this pin is Apache 2.0. This task adopts techniques
through independent Rust implementations and explicit references, not copied
C++/CUDA source or a vendored runtime. Repository rules remain no C++/nvcc/CMake,
no imported Ninfer implementation, and no runtime `.ninfer` parser. A future
decision to reuse source text would need its own licensing/notice review and a
change to the current project boundary; this research makes no such change.

The objective remains matched performance and meaningful quality, including long
context, concurrent serving and request behavior. These findings identify the
next causes to isolate. They do not supply the missing final measurements.
