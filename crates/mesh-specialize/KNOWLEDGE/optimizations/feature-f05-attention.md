# F05: tiled FP32 online attention candidate

Status: opt-in resident integration at `238ac8a7d`. Synthetic GPU checks and
three sanitizers passed; same-input real-weight layer3 audits passed at128/512.
Two short natural prompts passed strict same-profile model checks but showed
cross-profile logit drift and a changed prose continuation. Current timings are
CPU-contended and do not establish an uncontended speedup. Default remains exact;
longer-answer quality and resident sanitizer qualification are pending.

## Candidate contract

The NVPTX entrypoint is `attention_online_bf16` in
`kernels/nvptx/attention_online.rs`:

```text
attention_online_bf16(
    q: *const u16,
    cache_k: *const u16,
    cache_v: *const u16,
    output: *mut u16,
    unrounded: *mut f32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    width: u32,
    past: u32,
    capacity: u32,
    scale: f32,
)
```

Q and output are compact `[rows, query_heads, width]` arrays. K and V are
full-capacity, token-major `[capacity, kv_heads, width]` BF16 caches. Query head
`h` maps to KV head `h / (query_heads / kv_heads)`. Query row `r` attends to
cache rows `0..=past + r`; cache capacity after `past + rows` may contain poison
and is never read. The host must reject zero extents, invalid grouping, a
nonpositive or nonfinite scale, and any `past + rows > capacity` before launch.
Q/K/V values and the scaled FP32 scores and outputs must be finite.

Launch one CTA per query/head with `grid = [rows * query_heads, 1, 1]` and
`block = [256, 1, 1]`. Each CTA owns its output row. It stages eight cache keys
at a time, decoding the global BF16 values to FP32 in shared memory. Threads
cooperatively form eight QK dots, one partial per channel and key, then reduce
each 256-element partial row with a shared-memory tree. Invalid causal tail
positions stage zeros and receive zero probability. No global score or
probability matrix is materialized.

For each key tile, thread zero computes `m' = max(m, tile_max)`, rescales the
old denominator and value accumulator by `exp(m - m')`, and adds the tile's
`exp(score - m')` weights. The kernel evaluates exponentials with PTX
`ex2.approx.f32` after multiplying by `log2(e)`. It rounds FP32 output to BF16
with `cvt.rn.bf16.f32`. This FP32 dot/softmax profile changes arithmetic from
the existing FP64 attention path. It does not claim bit equality with that path.
Each thread reaches every CTA barrier, including threads outside `width` and
the final partial key tile. Validated causal rows always include at least one
key, so an all-masked query is rejected by the host/reference contract rather
than passed to the kernel.

The kernel has no global scratch allocation. Static CTA shared memory is
24,640 bytes: 8,192 bytes for decoded K `[8, 256]`, 8,192 bytes for decoded V
`[8, 256]`, 8,192 bytes for QK partials `[8, 256]`, and 64 bytes for control
values. Each partial tile is read from its row root after reduction. The output
buffers each cover `rows * query_heads * width` elements in compact row/head/
channel order: `output` stores BF16, and `unrounded` stores the corresponding
FP32 value. The host must provide aligned, live, nonoverlapping inputs/outputs
and checked element-count arithmetic.

The independent CPU oracle is `reference/attention_online.rs`. It indexes
logical grouped heads and token-major cache rows, computes a complete FP64
logical dot/softmax/value result, then converts to FP32 and BF16. It does not
replay the kernel's shared reduction or approximate exponential. Its tests cover
past zero, nonzero past, odd capacity with a poisoned cache suffix, grouped
query heads, large positive and negative logits, zero extents, and an all-masked
softmax. A test-only FP32 recurrence checks small tiled fixtures against the
oracle; it uses ordinary CPU `exp` and sequential dots, so it is not a GPU
emulator or GPU qualification.

## Qualification gates

The existing attention component allowance remains `5e-6 + 3e-5 * max(abs(V))`
per FP32 output channel. The candidate must also round each FP32 diagnostic to
its stored BF16 output exactly. Chunked query submission has a predeclared
`1e-6` maximum absolute FP32 difference in the CPU fixture. These source-level
budgets do not relax existing baseline or full-model gates, and the new profile
must be reported separately.

Before any resident promotion, the parent must register the source and
reference, compile and JIT the PTX, inspect registers/spills/shared-memory use,
and run real-weight Q/K/V comparisons against this oracle. Exercise odd key
tiles, nonzero past, GQA ratios, score extremes, whole/chunk/token query
partitions, BF16 rounding, and poisoned cache tails. Run memcheck, racecheck,
and synccheck. Then run the unchanged whole-layer and full-model hidden/logit/
state gates, text-quality checks, and matched profile measurements. Context
splitting across multiple CTAs and FP8 KV integration are follow-up work, not
part of this candidate.

## Evidence record

- CPU tests and formatter: authored, not run. Cargo/build slots belong to the
  parent.
- PTX compile/JIT, registers, occupancy, stack/spills, and shared allocation:
  not measured.
- Device, driver, clocks, model, real weights, sanitizer reports, and timings:
  not measured.
- Reproduction command: pending parent module registration and PTX build setup.
- Source revision: the worker did not query Git; record the integration revision
  with the parent-owned qualification evidence.
- Expected result: bounded shared storage, causal/GQA-safe output for valid
  inputs, and FP32 error within the unchanged component budget on qualified
  real-weight cases.
- Observed result: source candidate and independent CPU oracle only.

Durable rule: a tiled kernel, CPU fixture, or emitted PTX does not qualify the
new arithmetic profile for resident dispatch. Keep the FP64 baseline and BF16
cache available until real-weight and full-model gates pass.

Parent registered both modules. On 2026-09-27, 239 macOS tests passed; after a test-only Clippy iterator repair, host Clippy and NVPTX compilation passed. GPU and model qualification remain pending.

Parent GPU check on Carrack RTX5090 passed four synthetic grouped-head cases with zero/nonzero past, poisoned unused cache tails and extreme scores. BF16 outputs matched; maximum raw error was 2.3842e-7 under the predeclared budgets. Memcheck, racecheck and synccheck all reported zero errors/hazards. JIT used 40 registers, no local memory and 24640 shared bytes. PTX SHA256 `7492498c60ecc890881c93f5429880da07d03e42b1df2d17d98157c13ff67d25`. Evidence: `../evidence/iterate-20260927/features-attention/`. Model integration, long-context qualification and performance remain pending.


## Parent resident audit preparation

The512-token profile identifies244.85ms of attention prefill events and22.05ms
of subsequent decode attention events. These are synchronized diagnostic totals,
not wall-time attribution. Parent prepared explicit attention selection via
`MESH_SPECIALIZE_ATTENTION_PROFILE=exact|online|online-audit`; default exact
retains the FP64 kernel. The audit runs online FP32 against identical prepared
Q/K/V and persistent KV in `layers.03` for the first >=16-row prefill and following
one-row decode with past>=16. Only exact outputs feed the model in audit mode.

Audit compares all FP32 outputs to exact GPU using causally visible per-channel
V bounds and the unchanged5e-6 +3e-5*bound allowance. It also checks selected
first/middle/final and worst-error query rows across every head against the independent logical FP64
oracle, stored BF16 rounding, finiteness and full-output drift. Audit capacity is
bounded to2048. Model-profile requires both prefill/decode reports to pass;
throughput rejects audit mode. Reports/logit manifests identify attention
arithmetic separately from projection arithmetic. MTP rejects non-exact attention
until recovery is qualified. Same-profile exact checks remain unchanged.
This is preparation; real-weight audit, model quality and performance are pending.


Real-weight audit `attention-audit-1` at source `238ac8a7d` passed at128/512
inputs with exact outputs continuing to drive the model. Layer3 prefill BF16
differences:155/786432 and636/3145728; raw relative L2:1.8451e-7 and1.8598e-7.
Following decode:1/6144 differing BF16 values at both lengths, raw relative
L2:1.5908e-7 and2.7077e-7. All-output candidate/exact GPU comparisons passed the
unchanged per-channel component budget. Independent FP64 oracle samples cover
all24heads across selected first/middle/final/worst-error rows; maximum candidate
error/budget ratios were0.01924/0.01329 for prefill and0.00641/0.01020 for decode.
Stored BF16 rounding, finiteness, exact-model partition and profile/control gates
passed. These are same-input operator diagnostics, not semantic quality or
candidate model throughput. PTX stayed `features-greedy`, SHA256
`961d1652408eeb9ec8d72aa9c14d32ced2c2f0efb3e8e2a32dd0d5cdcc0c4a05`.
Host267tests, both-host Clippy, Linux release build and no-console checks pass.
Ninfer stayed inactive and ComfyUI remained resident. Candidate-driven natural
model comparison is running separately in `attention-model-1`; no promotion.

## Short natural-prompt model comparison

`attention-model-1` passed strict same-profile whole/token partition and complete
profile/control state checks for both prompts and modes. Python32-token outputs
match; prose differs. Prefill KL(exact||online) is0.04787/0.02129 and total
variation0.05999/0.06801; one same-input teacher-decode KL is2.707e-5/5.504e-5.
The tiny audited operator differences do not bound full-model drift. Semantic
quality remains unqualified. Timings were CPU-contended (load49.44 with unrelated
CUDA compilation), fixed-order and shared-GPU; no clean speedup claim. See the
evidence README and comparison JSON. Default and exact gates remain unchanged.

`attention-prefill-1` then failed strict128-token whole/token partition equivalence,
while profile/control remained exact. The script stopped before512. First observed
last-row hidden drift is layer25; earlier rows were not captured. Preserve the
failure and investigate possible downstream NVFP4 multirow-native/single-row
integer arithmetic differences on identical inputs before attributing the failure
to attention or relaxing any gate. See failure-summary.json and evidence README.

Parent added optional `MESH_SPECIALIZE_PARTITION_AUDIT=1` to model-profile.
It downloads every BF16 layer output and records SHA256 per logical row for
whole-prefix and token submissions. The report lists every differing row by
layer, with the first differing pair of hashes. It never feeds model outputs
or changes arithmetic, and rejects incomplete/overfull diagnostic capture.
The ordinary last-row drift report and strict partition gates remain intact.
This is diagnostic instrumentation, not a performance measurement. Host tests
cover submission-independent row hashing, earlier-row localization, and invalid
capture extents;269 host tests pass. Real-model localization is pending.

Arithmetic guarantee checked against NVIDIA PTX ISA9.4, `mma` precision section:
<https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>.
For E2M1 floating-point MMA, the specification provides a minimum accumulation
precision but does not fix accumulation order or rounding. Thus native NVFP4
MMA cannot be assumed equivalent to our integer reduction for every input merely
because earlier fixtures matched. This supports measuring the downstream
hypothesis; it does not identify the cause of the observed partition failure.

`partition-audit-1` atfbf877158 localized the earliest all-row mismatch to
GDNlayer22,row101(zero-based). All rows through layer21 agree. Exact attention
agrees across every row and layer. Both modes retain prior whole/token final
state hashes. Next diagnostic `MESH_SPECIALIZE_PARTITION_STAGE_LAYER=22` records
existing layer stage outputs; observed MLP uses the ordinary allocation path,
so whole-profile/control equality and previous hashes must be checked before
using it to localize arithmetic. Captures stop before teacher decode and never
feed model values. Stage diagnostics are not throughput or qualification.

`partition-stage-1` exact mode passes every row/stage numerical check but fails
global free-memory release because another mesh-llm process appeared onGPU0
with1000MiB; preserve that failed gate. `partition-stage-2` online mode localizes
the first differing stage to layer22 NVFP4 `mlp_down`, row101. Every prior stage,
including BF16 MLPactivation inputs, matches at all128rows. Both whole/token
final-state hashes match the prior non-stage-captured trial; profile/control
outputs/state match and memory release passes in stage2. Strict partition still
fails. Next audit identical quantized inputs through native/integer projection
and independent CPU reference; no attention arithmetic change is justified yet.
