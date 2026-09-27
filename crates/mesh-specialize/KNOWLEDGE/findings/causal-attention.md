# Causal attention and persistent BF16 KV state

Status: real-weight core, partition/cache equivalence and all sanitizers pass. This adds the
attention core after resident layer-3 Q/K preparation and V projection. The
output sigmoid gate, output projection, full attention layer and full model
remain pending. Layer-3 trial inputs remain explicitly synthetic embedding rows;
this does not imply execution of preceding layers.

The first core uses causal grouped-query attention with a persistent token-major
BF16 K/V cache. Query head h maps to KV head h/(Q_heads/KV_heads). Appending a
chunk copies only its K/V rows at the current prefix length. A query in chunk
row r attends exactly prefix+row+1 keys, even when later chunk rows have already
been appended. Capacity is distinct from initialized length; untouched capacity
uses NaN poison in the harness and must remain bitwise unchanged.

One CTA owns a query/head; each thread owns one value channel. A shared FP32
QK reduction supplies each causal score. Stable online max/denominator/value
accumulation avoids a quadratic global score allocation. Explicit rounded FP32
arithmetic prevents unintended FMA; exponentials use approximate exp2 and may
flush negligible underflows. Output divides accumulated values by the denominator
in FP32, then rounds to BF16. This is a fused-attention-style FP32 score/softmax
profile, distinct from the eager BF16 logits/probabilities path. The scalar
oracle computes logical dots, stable softmax and weighted values in f64.

Before measurements, each FP32 output must meet 5e-6 + 3e-5*max(abs(V)) over the
causally visible values for that output channel. BF16 must exactly round the GPU
FP32 diagnostic; independent scalar BF16 differences stay visible. Whole-sequence,
[1,remainder], [2,1,remainder] and token-by-token execution must produce identical
BF16/FP32 outputs and final K/V bits. Every intermediate append must preserve the
prior prefix and poison tail exactly and match an independent CPU cache update.
Hand fixtures cover GQA mapping, causal masking, stable large logits, signed values,
zero queries and unused NaN capacity. CPU validation rejects malformed extents,
nonfinite initialized inputs, invalid scales and out-of-capacity appends.

This is a correctness and memory-layout foundation, not the final throughput
kernel. It still processes keys sequentially inside each CTA. Tiled tensor-core
attention and launch/graph tuning remain performance requirements; no harness
wall time may be reported as model prefill or decode. BF16 cache here does not
claim parity with Ninfer's deployed FP8 KV profile or usable long-context memory.
The remaining full-model comparison must use explicit matched cache/speculation
profiles and measured peak memory/context.

Workers own one two-kernel device module and one independent CPU reference. The
parent owns resident-buffer integration, persistent/chunk harness, source review,
PTX inventory and bounded Carrack trials. New kernels require host tests/Clippy,
PTX compilation and all three CUDA sanitizer checks before qualification.

Partition tests reuse identical resident prepared Q/K and projected V arrays,
then partition the core's append/query calls. They qualify core/cache behavior;
they do not yet prove that the entire projection/decoder pipeline or sampled
logits agree between prefill and decode. The prepared outputs remain device
buffers: CPU reference outputs never replace them.

The first local run caught an incorrect hand fixture expectation: 1,000,000 is
not exactly BF16-representable, so its expected uniform mean used the wrong
input. The causal exclusion test now uses exactly representable +/-2^20 and
hand-computed means. The failed log is preserved; production arithmetic and
numerical budgets did not change.

## Carrack qualification

Source `9333e09185da211ec34f2458460b63dd121dfb76` passes on RTX5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09/API 13040.
The host uses Rust 1.98.1/LLVM 22.1.8; device PTX uses nightly-2026-09-25 with
rebuilt NVPTX core. CUDA 13.4.92 offline assembly and driver JIT agree on 31
registers/1,024 bytes shared for causal attention and 14 registers/no shared
memory for append. Both have zero local memory/stack/spills; the attention
kernel uses one barrier identifier and append uses none.

Release xtask SHA-256:
`a405560abdb2fe154d36539e2398ce20b36d6067f2e8698c22a02dae71600124`.
PTX SHA-256:
`86c006bfce3f1aff27d64d4e80428304493ce8a3a1552d391ecde8381361c42d`.
Mac passes 156 library tests; Linux passes 167 library and 17 validator tests.
Both focused Clippy runs, formatting, no-console, PTX compilation and offline
assembly pass. Independent reviews cover chunk offsets, resident inputs,
reference state, GQA addressing, causal bounds, online arithmetic and barriers.

The one-token case produces 6,144 exact scalar FP32/BF16 outputs. The 17-token
case produces 104,448 outputs, with maximum FP32 absolute error
0.0000011920928955078125 and nine BF16 differences from the wide reference.
All outputs pass the original per-channel bound and exactly round their GPU
FP32 diagnostics. The [17], [1,16], [2,1,14] and [1 x 17] paths have bit-identical
FP32 outputs, BF16 outputs and final K/V caches. All 24 real-case cache boundaries
match the independently appended CPU state, including prior prefixes and every
untouched NaN-poison tail word. These tests cover a 24-query/4-KV-head ratio,
256-wide heads, and capacity distinct from initialized token count.

Four synthetic fixtures add 2,756 unique outputs, covering GQA ratios two/three,
widths 2/8/13/256, signed values, zero queries and large logits. They pass the
same numeric/rounding checks and every partition/cache check. Their maximum
FP32 absolute error is 0.0000002384185791015625. The zero-query fixture has three
BF16 differences associated with tiny f64 cancellation residuals (maximum
FP32 error 5.551115123125783e-17); these are retained diagnostics, not claimed
bitwise agreement with the wide oracle.

Normal, memcheck, racecheck and synccheck reports all pass; sanitizers report
zero errors/hazards/warnings. Their elapsed harness times are 12.290836382,
12.31510837, 17.78238325 and 12.255517099 seconds. Each used an 8 GiB user scope,
zero swap and a 240-second timeout. These are correctness harness times, not
inference throughput. Driver free memory before/after temporary allocations is
32,221,822,976 bytes, not a peak model-memory measurement. Model prefill, decode
and usable-context fields remain null.

Ninfer restarted at 04:13:00 EDT on 2026-09-27, PID 3012208, and reported engine
ready at 04:13:06. Fresh `/health` returned HTTP 200. Its sampled GPU allocation
is 30,046 MiB; ComfyUI PID 448118 remains at 498 MiB. The original Carrack branch,
Ninfer configuration/source and model assets remain preserved.

Evidence is committed under `KNOWLEDGE/evidence/qwen-causal-attention-20260927/`.
Raw files/PTX remain under `target/specialize/qwen-causal-attention-20260927/` on
both hosts. Reproduce using the existing `qwen-attention-check` command (schema
2); it persists failed numerical reports before returning failure. Next: resident
sigmoid output gate and FP8 output projection, then residual/norm/MLP and an
independent full attention-layer reference. Full model scheduling, logits, ABI
serving, cache compression and the measured Ninfer comparison remain open.
