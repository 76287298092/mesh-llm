# Complete full-attention decoder layer

Status: complete layer-3 numerical trial and all three sanitizers pass. This completes layer 3's
synthetic-input component chain with sigmoid output gating, FP8 output projection,
first residual/post-attention norm, NVFP4 MLP and final residual. It also adds an
independent whole-layer CPU comparison, starting only from original tokens,
positions and checkpoint weights. Preceding layers and full-model logits remain
outside this trial.

Pinned Transformers attention multiplies its BF16 attention result by
`sigmoid(gate)` before output projection. The chosen eager boundary rounds stable
FP32 sigmoid to BF16, multiplies the decoded BF16 values in FP32, then rounds the
product to BF16. GPU diagnostics expose sigmoid, rounded sigmoid and product.
The scalar sigmoid oracle uses f64. Before measurement, sigmoid FP32 uses the
existing 3e-6 + 5e-6*abs(reference) bound; intermediate/final rounding and product
from actual rounded sigmoid must be exact. Full-reference BF16 differences remain
visible. Negative extreme inputs use the previously qualified non-FTZ exponential
profile, including subnormal sigmoid near -90.

The existing FP8 matrix, residual/norm, NVFP4 MLP and residual-add kernels are
reused with verified layer-3 weights. The complete chain consumes resident device
buffers, including saved raw Q gates and the initial residual; references never
replace device intermediates. Per-component checks remain enabled.

The separate whole-layer reference composes only scalar operations: layer-3 input
norm, FP8 Q/K/V, Q/K norm and text RoPE, f64 causal GQA, BF16 sigmoid/product,
FP8 output, residual/post-norm, NVFP4 MLP and final residual. Its initialized K/V
states are independently derived. The existing fixed whole-layer budget applies
to every token's final hidden vector and each cached token/head K/V vector:
normalized L2 <=0.01 and cosine >=0.9999, plus aggregate checks and explicit zero
handling. No tolerances may change based on observed results. Exact-bit counts
remain diagnostics; this does not imply 64-layer quality or framework parity.

New gate launch paths require local/Linux tests and Clippy, formatting/no-console,
PTX compilation, real-weight trial and all three sanitizers. Whole-layer failures
must remain visible in persisted JSON even if all individual components pass.

Local validation caught two test-only issues before GPU deployment: dynamic array
repeat counts in the tiny fixture and a Clippy excessive-precision literal. The
fixture now uses its fixed 16-element dimensions and expresses the exact sigmoid
midpoint as 0.5 + 1/512. Failed logs are retained. The resulting 164 host tests,
focused Clippy, no-console check and Rust NVPTX compilation pass on macOS.
Linux compilation and execution remain pending at this source checkpoint.

Launch review confirmed pointer order, extents, tail guards and numeric boundaries.
It found that CUDA fixtures had negative-zero gates but no negative-zero attention
inputs. The edge fixtures now include those inputs, require exact scalar BF16
outputs, and record the negative-zero count. This tests product sign preservation
as well as the nonzero subnormal sigmoid case on the GPU.

The first complete real-weight trial, source `cffc88757`, failed the original
whole-layer gate. One token passes with normalized L2 0.00013146. Seventeen tokens
reach aggregate normalized L2 0.01345972 and worst-token L2 0.03488956, so this
layer is not qualified. Each component comparison passes and initialized K/V
comparisons pass. Ninfer restarted successfully and health returned HTTP 200.
`normal.json` and its logs are preserved. Sanitizers were not attempted after
this numerical failure. The next diagnostic trial records independently derived
scalar intermediate stages against actual GPU intermediates, without replacing
inputs or changing any arithmetic or tolerance.

The unchanged-arithmetic diagnostic source `348bc6590` reproduces the failure.
For 17 tokens, accumulated error is 0.000103819 after attention gating,
0.000787763 after output projection, 0.001021758 after post-attention norm,
0.007671315 after MLP activation and 0.021388754 at the MLP down branch.
Final hidden aggregate remains 0.013459719. The trace retains every stage and
per-token metrics. Tiny initial Q/K/V BF16 differences grow across subsequent
quantization boundaries. All individual components still pass against actual
inputs. The diagnostic fixture initially used an incorrect expected Q width of
64 rather than 32; the fixed host tests pass and the failed log is retained.

The next controlled experiment changes only attention FP8 projections to a new
wide tile-accumulation entrypoint: reset each K32 tensor-core MMA accumulator,
sum tile results in FP64, round once to FP32, then preserve the original FP32
row/channel scaling and BF16 rounding. The independent scalar reference and all
budgets stay unchanged. Per-tile MMA still rounds in FP32, so this is not a claim
of exact f64 dot products. Existing `fp8_linear` keeps its original arithmetic.
Extra FP64 work has unmeasured performance cost; this is a correctness experiment,
not a throughput optimization or a qualified model-serving path.

The wide-entrypoint trial also compares original/wide projection kernels on
1x1x1 and 3x9x35 signed fixtures, covering row/channel/K tails against the same
independent reference. Gate edge fixtures cover 1/257/513 elements, negative-zero
attention values, saturated gates and a subnormal sigmoid. These run in every
normal and sanitizer invocation.

Offline CUDA assembly rejected the first wide PTX before Ninfer was stopped.
FP32-to-FP64 widening is exact and PTX rejects a rounding modifier on that
conversion. It now uses `cvt.f64.f32`; narrowing retains `cvt.rn.f32.f64`.
The rejected PTX and assembler log remain as `probes-wide.ptx`/`ptxas-wide.log`.

Wide accumulation source `f01e2ba6b` reduces the 17-token aggregate L2 error to
0.003044232 and makes the one-token layer bit-exact. Token 12 still fails with
L2 0.014019502. Only two projected Q BF16 values now differ; K/V projections
are exact. The failing token has one Q projection difference. This remains a
failed whole-layer trial. The unchanged reference and fixed per-token gates
caught it despite an aggregate pass.

The next refinement targets BF16-ambiguous FP8 outputs. A second positive-product
MMA per tile estimates sum(abs(products)), accumulated in FP64. Before measuring,
the interval radius is fixed at 16e-6*absolute_sum*sx*sw + 1e-6*abs(scaled_output),
conservative for K32 FP32 reductions and final FP32 scale operations. If both
interval endpoints round to the same BF16 code, the original wide result stays.
Otherwise the kernel independently decodes the raw FP8 row/column and sums their
products in FP64, then applies the original FP32 scale order. For K<=32768, every
E4M3 product is a multiple of 2^-18 and the absolute sum fits FP64's significand.
The original entrypoint, reference and tolerance remain unchanged. The cost of
extra MMA and scalar fallback is unmeasured and needs later performance work.
A 35-wide cancellation fixture places a residual just above a BF16 midpoint;
refined output must match scalar BF16 exactly on this and both tail fixtures.

## Complete-layer numerical qualification

Source `8d5528f046d6535346e7dffc1c48ede01fa1671b` passes the unchanged complete-layer
numerical gates on RTX5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120,
driver 615.71.09/API 13040. Model identity, raw-v1 artifact and checkpoint revision
are unchanged from the prior attention trials. Host Rust is 1.98.1/LLVM 22.1.8;
NVPTX uses nightly-2026-09-25 with rebuilt core, and offline CUDA tools are 13.4.92.

Release xtask SHA-256:
`b75fc59b4044c05c945b43bda050aff6babf0a24074309d43431d7cd0a7cd4b0`.
PTX SHA-256:
`ee13b6bf3d34ee2ccceeaf4b9420c32ddc0fee2c97fc7f7d529f86ba06c306b9`.

| Projection profile | One-token hidden L2 | 17-token aggregate L2 | Worst token L2 | Fixed gate |
| --- | ---: | ---: | ---: | --- |
| Original running FP32 MMA | 0.000131464 | 0.013459719 | 0.034889559 | Fail |
| FP64 sum between MMA tiles | 0 | 0.003044232 | 0.014019502 | Fail |
| Tile sums plus ambiguous-output refinement | 0 | 0.0000584683 | 0.0002702340 | Pass |

The one-token case has 5,120 bit-exact final hidden values. The 17-token case has
87,040 values, 36 BF16 differences, aggregate cosine 0.9999999982907376 and worst
token cosine 0.999999963491221. Both initialized K and V caches are bit-exact
against their independently derived scalar counterparts, 18,432 values each
across the two cases. All per-token/head and aggregate budgets pass.

The resident chain covers 258,048 Q/K/V projection outputs, 110,592 sigmoid gate
values, 92,160 output-projection values, post-attention residual/norm, 718,848 NVFP4
MLP matrix outputs, SiLU product and the final residual. Q/K/V projections now
match independent BF16 results exactly. Local output-projection BF16 results also
match exactly, with maximum FP32 reference error 4.76837158203125e-7. Whole-chain
output projection retains 140 BF16 differences in the 17-token case because the
upstream attention/norm/gate path is approximate. The independent trace keeps that
distinction visible. The MLP down outputs are bit-exact against the full scalar
chain; remaining final hidden differences come from the residual branch.

Three gate fixtures cover 771 values, including 60 negative-zero attention inputs.
Their intermediate/product rounding and scalar BF16 outputs are exact, including
the sigmoid subnormal near -90. Real sigmoid maximum FP32 error is
5.960464477539063e-8; one 17-token gate BF16 boundary differs from the f64 sigmoid
reference and remains within the original component bound. Six matrix fixture
runs cover 58 outputs across original/refined entrypoints. The cancellation case
has one BF16 error with the original entrypoint and none with refinement; refined
fixtures match scalar BF16 exactly, including row/channel/K tails.

Driver JIT reports 46 registers for refined FP8 and 20 for the sigmoid gate, with
zero local/shared bytes. Offline assembly reports 50/20 registers, zero shared
memory, stack, spills and barriers for these kernels. The compiler allocation
difference is recorded rather than presented as agreement. Host tests pass:
165 on macOS, 177 library plus 17 validator tests on Linux. Focused Clippy,
formatting, no-console, PTX compilation and offline assembly pass. GPU clocks and
power were not controlled for this correctness experiment.

The full-layer trial still feeds synthetic embedding rows into layer 3. It does
not run layers 0..2, all 64 layers, final norm/logits, tokenizer/sampling or serving
ABI. Core/cache partition equivalence is retained, but whole decoder prefill/decode
partition parity still needs a resident scheduler. KV is BF16, unlike Ninfer's
FP8 baseline. Refinement frequency and performance cost are unmeasured. The next
execution work is model scheduling with persistent per-layer state and independent
logit checks; full-model prefill/decode, peak memory and usable context stay open.

Durable rule: component bounds alone cannot qualify a quantized layer. Small
upstream rounding changes can cross later FP8/NVFP4 boundaries and exceed the
whole-layer budget. Preserve independent original-input traces and per-token
checks, and improve arithmetic before considering any accuracy-budget change.

Normal, memcheck, racecheck and synccheck all pass the complete layer and fixtures.
The sanitizers report zero errors, hazards or warnings. Elapsed harness times are
69.130711567, 68.680204111, 73.553326073 and 67.049201162 seconds. Each invocation
used an 8 GiB user scope, no swap and a 240-second timeout. These CPU-reference
harness times are not model inference rates. Driver free bytes before/after
temporary allocations are 32,221,822,976, not a model peak-memory measurement.
Model prefill/decode/context fields remain null.

Ninfer restarted on 2026-09-27 at 04:55:15 EDT, PID 3047200, with engine ready at
04:55:21 and fresh `/health` HTTP 200. Its sampled allocation is 30,046 MiB.
ComfyUI PID 448118 remains at 498 MiB. The original Carrack branch, Ninfer source
and configuration, and model assets are preserved.

Committed reports and logs are under
[`evidence/qwen-full-attention-20260927`](../evidence/qwen-full-attention-20260927/).
The raw directory `target/specialize/qwen-full-attention-20260927/` on both hosts
also retains PTX/cubin artifacts; the failed original and wide-only numerical
trials and failed assembler/local-test logs are preserved. `refined.json` and
`refined-{memcheck,racecheck,synccheck}.json` are the qualified reports.
The `qwen-attention-check` command now emits schema 3. Use a fresh output file and
an appropriately bounded service window to reproduce; saved outputs are never
overwritten. All model scheduling, full-model/logit and serving/performance gates
remain open in the work ledger.
