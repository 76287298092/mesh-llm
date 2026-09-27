# F05: tiled FP32 online attention candidate

Status: source candidate authored on 2026-09-27. The parent has not registered or
compiled it. Driver JIT, GPU correctness, sanitizers, resource use, timing, real
weights, and full-model qualification remain pending. Resident dispatch remains
unchanged.

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
