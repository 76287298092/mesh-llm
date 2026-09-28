# BF16 split-sequence attention v2

Status: implemented, unqualified model opt-in. Scope covers M=1 decode and M=2..8
verification; larger chunks keep exact attention. Flash prefill is deferred.
Defaults and scoring code are unchanged. Legacy and StreamForward opt-in wiring
is now implemented, described below. No Cargo, Git, SSH, GPU trial, or delegation
was performed by the worker. Standalone rustfmt parsed/formatted the changed files.
Parent reports the initial kernel delivery passed PTX compilation and Linux-target
host check/Clippy; that report predates this model wiring. Parent-supplied source
baseline: ab33f730e, not independently queried. Record the integrated revision,
PTX hash and qualification evidence separately. Existing unrelated stream/scoring
sanitizer results do not qualify the new split-attention integration.

## Design

Fixed model geometry: 24 Q heads, 4 KV heads, D=256, GQA group six. Q and output
are compact [M,24,256]. K and V stay BF16 token-major [capacity,4,256], exactly as
resident_attention_core.rs. Upstream Q/K preparation already applies partial
RoPE. Existing attention_kv_append must finish before these kernels, and caches
must be initialized through past+M. These kernels never append or modify KV.
Query row r attends positions 0..=past+r. Capacity is not initialized length.
Scale remains the existing FP32 1/sqrt(256), exactly 0.0625.

One 192-thread partial CTA owns (KV head, sequence split). Six warps each own one
Q head. Threads stage a BF16 K[16,256] and V[16,256] tile once from global memory;
each warp reuses the shared tile, including across every verification row. Each
lane holds eight Q channels and eight weighted-value accumulators per row.
A full-warp butterfly sums each FP32 dot. Online max/sum/value recurrence stays
FP32; approximate exp2 implements exp. The partial record retains unnormalized
accumulators. The reduction combines maxima, sums, and weighted accumulators
stably, then divides and rounds once to BF16. A diagnostic FP32 output is retained.
No quadratic score matrix or global atomic exists.

Dedicated M=1 entrypoints compile separately from the M<=8 dispatch, so verify's
larger register state need not determine decode's register allocation. The general
entrypoints specialize all eight row counts at compile time. JIT register counts,
stack/local storage and actual specialization/unrolling still need inspection.

Host `kernels::attention_v2_plan::split_count(L)` chooses min(85,ceil(L/64)).
At 8K and 128K this launches 4*85=340 partial CTAs, targeting two waves across
170 SMs. At short lengths it avoids very small slices. Split s owns tile indices
floor(ceil(L/16)*s/S)..floor(ceil(L/16)*(s+1)/S), so splits never overlap and
boundary KV tiles are not reread. The final tile zero-fills uninitialized positions.
A row with no visible keys in its split produces a neutral record.
This heuristic is a hypothesis; no bandwidth or speedup has been measured.

## Raw launch ABI

All pointers below are device addresses (64-bit host launch parameters); integer
parameters are u32, scale is f32. Caller owns aligned, nonoverlapping, live buffers
and must validate all extents before launch. All launches must be stream-ordered.
Out-of-contract device dimensions return without writing outputs, not an error
report, so the device checks do not replace admission validation.

### By-value position

`attention_split_decode_bf16` (M exactly one) and `attention_split_bf16` (M=1..8):

```text
q: *const u16, k: *const u16, v: *const u16, partial: *mut f32,
rows: u32, query_heads: u32, kv_heads: u32, width: u32,
past: u32, capacity: u32, split_slots: u32, scale: f32
```

Grid [4,split_slots,1], block [192,1,1], dynamic shared=0. Static shared=16,384 B.
K/V must be at least 4-byte aligned because producers load packed BF16 pairs.
Q/output BF16 alignment is two bytes; FP32/workspace alignment is four bytes.
Bounds: rows 1..8, heads exactly 24/4, width 256, capacity 1..262144,
past+rows<=capacity, positive finite scale, finite Q and initialized KV, finite
scaled scores/results. Active splits S=min(85,ceil((past+rows)/64)); require
S<=split_slots<=85.

### Device position for captured decode

`attention_split_decode_bf16_position` (M exactly one) and
`attention_split_bf16_position` (M=1..8):

```text
q: *const u16, k: *const u16, v: *const u16, partial: *mut f32,
position: *const u32,
rows: u32, query_heads: u32, kv_heads: u32, width: u32,
capacity: u32, split_slots: u32, scale: f32
```

Position is an aligned device u32 containing past, loaded at launch. Update it on
the same stream before replay; do not concurrently mutate it. Host Plan::new
accepts the replay interval's maximum length and fixes the grid/workspace stride.
Actual active split count is recomputed from the scalar. Extra CTAs explicitly
write neutral [-inf,0,zero_acc] records, including after a long-to-short rewind.
M, geometry, capacity and allocated split_slots remain fixed for a captured graph.
The existing KV append is not changed to read this scalar in this delivery; parent
integration must separately arrange append and cursor ordering. Kernels allocate
nothing and use no host data or synchronization. Actual graph capture is untested.

### Reduce and workspace

```text
attention_split_reduce_bf16(
    partial: *const f32, output: *mut u16, unrounded: *mut f32,
    rows: u32, split_slots: u32)
```

Grid [rows*24,1,1], block [256,1,1], dynamic/static shared=0.
Workspace is FP32 [rows,24,split_slots,258]: item 0 max, item 1 sum, items 2..257
unnormalized weighted values. Bytes = rows*24*split_slots*258*4, or 2,105,280 B
per query row at 85 slots (16,842,240 B at M=8). Reduce reads all allocated slots
and ignores sum==0. Outputs require rows*24*256*2 and rows*24*256*4 bytes.
No workspace initialization is needed when every grid CTA executes: all records
are overwritten. Retain arena ranges until reduce finishes. The trial adapter
allocates host argument vectors; production may prepare fixed arrays and pass
arena pointers directly. No allocator belongs in the device ABI.

## Shared memory and occupancy

K and V each take 8,192 bytes, total 16 KiB per partial CTA. This fits even the
standard 48 KiB per-block allocation allowance, without dynamic shared opt-in.
Using the task's 100 KiB/SM budget gives a shared-memory-only ceiling of six CTAs
per SM. This is not achieved occupancy: 192 threads/CTA, registers, granularity
and JIT spills may lower it. M=8 holds substantially more live state than M=1;
measure both resources. This delivery does not depend on the 99 KiB/block opt-in
limit and has not independently queried device limits. No cp.async, ldmatrix or
MMA is added here; prefill pipelining remains a separate delivery.

## Independent reference, gates, and bounded harness

`reference/attention_v2.rs` invokes the unchanged independent FP64 logical oracle
in reference/attention_online.rs. It does not replay the split recurrence. Host
fixtures hand-check grouped-head uniform means, inclusive causal tails, poisoned
unused capacity, single-key identity, invalid row counts and nonfinite results.
Host planner tests cover split boundaries, exact tile coverage, SM-target count,
workspace/ABI extents, invalid replay intervals and overflow.

Acceptance, fixed before GPU results:
- Every raw FP32 channel meets the existing 5e-6 + 3e-5*max(abs(visible V_channel))
  allowance. This preserves the prior causal attention component gate.
- Each row/head raw FP32 relative L2 <=1e-3 when reference norm >=1e-6. Near-zero
  heads use the absolute component gate; the report's L2 denominator floors at1e-6.
- BF16 output equals RNE(actual FP32 diagnostic) exactly and is finite. Differences
  from BF16(round(FP64 oracle)) remain counted. Do not apply the raw 1e-3 L2 gate
  to BF16 outputs: BF16 output rounding itself can exceed it.
- Inputs including the complete poisoned suffix remain bitwise unchanged; trailing
  64-byte canaries on both outputs and workspace remain intact.

For the bounded random fixtures, abs(V)<=1, so the maximum component allowance is
3.5e-5. This is an empirical acceptance budget, not a universal FP32 forward-error
proof. Existing full-model/partition gates and quality-gates.md are unchanged.

Parent command after its build/PTX steps:

```text
xtask specialize attention-v2-check --ptx PATH --device ORDINAL --output NEW_FILE
```

The existing run_probe wrapper uses create_new, preserves failed JSON reports,
records PTX SHA256 and exits unsuccessfully if all_passed is false. Thirteen cases:
(M,past)=(1,0),(1,1),(1,513),(1,8191),(1,32767),(5,8190),(2,14),(3,15),
(4,16),(6,62),(7,63),(8,61),(5,67). The last case has Q/K magnitude16; M8 uses
zero Q for uniform scores. Hash-generated inputs exercise distinct heads/channels;
capacity=initialized length+19 poisons uninitialized tails. Every case tests
by-value past, device past and a past=0 rewind using unchanged pointers and grids.
The rewind leaves future tokens initialized, explicitly exercising causal exclusion
and overwriting inactive workspace slots. This is scalar-path testing, not graph
capture qualification. Largest synthetic cache has 32,768 initialized positions;
128K GPU evidence is deliberately outside this first bounded delivery. The reused
oracle retains its 67,108,864-element/cache limit (capacity<=65536 at this
geometry); the wider device/planner capacity does not extend oracle admission.

Timing excludes oracle, device allocation, upload and readback. Host argument
construction/submission gaps between the two launches can appear in pair time. Three warmups and five
CUDA-event samples measure partial+reduce together, with median and raw samples.
Logical KV GB/s = (past+rows)*4*256*2*2 / pair_seconds / 1e9. It is not measured
DRAM traffic: repeated cache-hot data, Q/workspace/output traffic and two-launch
overhead matter. No target percentage is claimed. Sanitizer repetition count is
bounded but parent may need a smaller case subset if instrumentation is slow.

## Complete assembly inventory

All nine new asm sites are in kernels/nvptx/attention_split_math.rs:

| Function | Operation and proof obligation |
| --- | --- |
| coordinates | tid.x, ctaid.x/y; launch dimensions define ownership |
| shared_base | static aligned 16 KiB declaration/address; one per partial entry |
| barrier | bar.sync0; all 192 threads reach every tile publication/reuse barrier |
| store_word | st.shared.b32; unique producer word, bounded within K/V tiles |
| load_bf16 | ld.shared.b16; bounded published tile halfword |
| warp_sum | shfl.sync.bfly.b32, full mask; 32 converged lanes per Q head |
| exp | ex2.approx.f32 after log2(e); stable online/merge weights |
| round_bf16 | cvt.rn.bf16.f32; diagnostic-to-output rounding gate |
| divide | div.rn.f32; at least one visible key gives a positive denominator |

Target SM120; instruction reference: NVIDIA PTX ISA, special registers, shared
memory load/store, bar.sync, shfl.sync, ex2, cvt and div sections:
https://docs.nvidia.com/cuda/parallel-thread-execution/
Independent arithmetic reference: existing complete logical FP64 dot/softmax/value
oracle, plus hand fixtures above. attention_split.rs and attention_split_reduce.rs
add no inline asm beyond these helper calls. Compiler emission alone is not
qualification. Parent must type-check all host targets, compile/JIT PTX, inspect
resources, run memcheck/racecheck/synccheck, preserve failures, then apply model
and quality gates before any resident/default promotion.


## Decode-only model opt-in integration

Select `MESH_SPECIALIZE_ATTENTION_PROFILE=split-decode`. The recorded profile is
`bf16-split-decode-fp32-exact-prefill-v1`. Missing/`exact` selection still means
`bf16-fp64-v1`. Rows1..8 use split partial/reduce; rows>8 use the existing
causal_attention_bf16, never attention_online_bf16. Short initial prompts are
also M<=8 and therefore use split attention. The phase is selected by row count,
not by whether past is zero. The existing MTP requirement `Profile::Exact` stays
unchanged, so this profile cannot enter MTP/recovery paths.

`attention_profile.rs` adds parsing, naming, `uses_split(rows)` and
`supports_stream()` with host tests for boundaries/defaults/rejected values and
unsupported stream profiles. `attention_v2_plan.rs` adds strict24:4 D256 admission,
decode-vs-verify symbol selection, persistent workspace sizing and tests. No
unsupported small-M geometry silently falls back to a generic kernel. Larger
legacy chunks retain existing generic exact-shape admission.

### Legacy ownership and errors

`resident_attention_split.rs::Prepared` owns a temporary workspace and borrows
pre-resolved partial/reduce functions. Constructor verifies CUDA context identity,
shape geometry and capacity, before existing resident_attention_core KV append.
The original input/state context and extent checks remain in place. Append is
unchanged; the small-M branch then enqueues partial followed by reduce on the
same default stream. Slots are planned from configured capacity. Q/cache pointers
and BF16/raw output buffers remain live with Prepared through final synchronization.
Every append/partial/reduce failure drains via the existing failed-launch helper
before releasing the workspace. The helper itself does not synchronize or append.
This legacy path allocates per call, as before; the persistent executor below does
not. Larger exact attention and Online/OnlineAudit behavior are unchanged.

### StreamForward ownership and enqueue

`stream_forward/split_attention.rs::SplitAttention` is constructed only for the
explicit SplitDecode profile. It borrows the same-context module, resolves the
M1, M<=8 and reduce functions once, and owns one persistent buffer:

```text
bytes = min(max_rows,8) * 24 * split_count(configured_capacity) * 258 * 4
```

Exact construction neither loads these functions nor allocates the buffer.
`ensure_supported_profiles` admits only Exact/SplitDecode attention; existing
FP8 exact, NVFP4 baseline, MLP-workspace-off, split-K-off and no-audit restrictions
remain. The admission function remains child-visible for the concurrent chunked
benchmark worker. Host `persistent_workspace_bytes` supplies its sizing formula.

StreamForward passes optional borrowed state into layers::attention. It preserves
existing KV append, then emits two Enqueue/ActiveStream launches for M<=8 or the
original exact launch for larger chunks. Args are stack-backed, and every step
supplies current rows/past/capacity. No per-forward device allocation, module lookup,
extra context switch or wait is introduced. Workspace slots are fully overwritten
before reduction and shared sequentially by all16 attention layers. The original
arena Q/output/raw lifetimes cover the combined attention operation; persistent
scratch lives separately and cannot alias arena or cache memory.

Small-M forward admission checks the configured session capacity, row/workspace
bounds and prefix extent before any forward enqueue. ActiveStream checks function
context against its stream on every launch; construction checked the buffer's
context against those functions. Existing forward success/error synchronization
retains arena, KV and scratch owners through the drain. The cursor transaction
still commits only after successful forward/readback. No graph capture/replay is
claimed; this integration uses by-value positions, not the separate device-scalar
kernel ABI.

`StreamForward::report` adds the exact selected attention profile,
`attention_workspace_bytes` (zero for Exact), and a split-attention report with
workspace capacity/row/slot bounds and kernel names. Arena bytes remain arena-only;
consumers must include this separately reported persistent buffer in total memory.
Existing model profile/benchmark/logit manifests already record `Profile::name()`.
No scoring or comparator files were changed.

### Qualification still required

Parent must type-check/test the integration, then exercise same-profile legacy
versus stream on short prompts, exact-prefill-to-decode transitions and M2..8.
Run sanitizer checks that actually select split-decode, including failure/rewind
paths, before calling this integration qualified. The original standalone raw
component/L2/rounding gates and full-model gates remain unchanged.

A512-row teacher-forced prefill scores the unchanged exact branch and does not
exercise this decode-only profile. Quality requires teacher-forced incremental
inputs (or explicitly bounded M<=8 chunks) on identical contexts, evaluated under
the unchanged quality-gates.md thresholds. No current prefill scoring or unrelated
stream sanitizer evidence establishes split-decode quality, safety or speed.


## First whole-model ablation, September 28

At fd500c885 with PTXfc91792b..., 13 operator cases pass normal/memcheck/racecheck/
synccheck; racecheck uses forced synchronization and one worker. The selected
source is the exact pinned native Ninfer artifact, BF16KV, default FP8/NVFP4.
Balanced AB/BA whole-model trials use full-prompt warmup and two repetitions per
invocation, four samples per profile/prompt,256 fixed outputs. All within-profile
one-chunk equivalence checks pass. This is not cross-profile quality admission.

| Prompt | Exact median decode tok/s | Split median decode tok/s |
| --- | ---: | ---: |
|106 inputs|26.469|35.524|
|512 inputs|18.167|35.503|

Same-input incremental decode scoring is being added because ordinary512-row
scoring only exercises this profile's unchanged prefill fallback. No default
change or quality promotion. Evidence: evidence/reassess-20260928/decode-attention-check-1
and decode-attention-model-1. Both services restored after the trials.
