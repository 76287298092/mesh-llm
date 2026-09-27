# Ninfer Qwen decode path comparison

Status: bounded source comparison against the Ninfer archive at
[`9e163eee4b8acec21ab0ac765107b6a3f287b217`](https://github.com/Neroued/ninfer/tree/9e163eee4b8acec21ab0ac765107b6a3f287b217).
The archive was made from Carrack's source tree; it is not a Git checkout. The
deployed service binary's source provenance is unverified. This review ran no
matched Ninfer benchmark and makes no new timing or numerical-parity claim.

## FP8 routes differ by shape and token count

Ninfer's ordinary A16 GEMV consumes the BF16 activation directly, decodes packed
E4M3 weight codes, multiplies them by the represented BF16 values using FP32
FMA, then applies the weight row scale. It does not first quantize
the activation to E4M3. The source describes the direct path and packed code
loads in [`fp8_gemv.cuh`](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_gemv.cuh#L3),
with vector loads at [lines 38-65](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_gemv.cuh#L38)
and per-lane accumulation across four FP32 chains at [lines 74-92](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_gemv.cuh#L74).
The Qwen shapes use `Fp8GemvSchedule<8, 2, 8, 4, ...>`: eight warps per block,
two output rows per warp, eight FP8 values per lane, and four accumulator chains
([schedule definition](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_config.h#L20),
[Qwen shape](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n5120_k17408.cu#L5)).

The A8 route has a different input approximation: it reduces each BF16 row to
`max(abs(x))/448`, converts scaled values to saturated E4M3, and returns that
scale for the projection ([quantizer](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_a8.cu#L20)).
Our resident FP8 path also materializes E4M3 activation codes and row scales
before its projection ([`resident_fp8.rs`](../../src/kernels/cuda/resident_fp8.rs#L75)).
These paths therefore do not have the same activation arithmetic as direct A16
GEMV; any switch to A16 needs its own output/state quality check.

Ninfer dispatch is shape-specific. For `[5120,17408]` and `[5120,6144]`, T=1
selects A16 GEMV, A16 uses tiled SIMT for short multi-token inputs, and A8 is
admitted from T=25 ([17408 shape selector](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n5120_k17408.cu#L32),
[6144 shape selector](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n5120_k6144.cu#L33)).
Other shapes use other thresholds, and the dispatch chooses A16 or A8 from the
shape policy and current token count ([dispatcher](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_dispatch.cpp#L30)).
For example, the `[14336,5120]` and `[16384,5120]` shapes switch at T=12 and
T=11 respectively ([14336](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n14336_k5120.cu#L20),
[16384](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n16384_k5120.cu#L20)).
The `[5120,17408]` A16 route has dedicated token-tile shapes for T=17-20 and
other tiled choices through T=24. Its SIMT kernel reuses decoded weight data
across a token tile ([shape selector](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/shapes/n5120_k17408.cu#L32),
[SIMT kernel](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_simt.cuh#L3)).

The fused FP8 SwiGLU path is a deliberate exception to the ordinary linear
thresholds: when the caller permits A8, it routes T=1 and T>=3 through A8, while T=2 uses
A16. An A16-only policy always stays A16 ([route](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp#L20)).
Its A16 T=1 GEMV pairs gate/up rows and writes the final BF16 SiLU(gate)*up result
from the projection kernel ([launch](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu#L27),
[epilogue](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh#L31)).
Our FP8 MLP currently runs gate and up projections separately, then a separate
activation kernel ([`resident_mlp.rs`](../../src/kernels/cuda/resident_mlp.rs#L120)).
The fused Ninfer epilogue applies SiLU to the FP32 gate result and multiplies the
FP32 up result before its BF16 store. Our local activation contract consumes
BF16 gate/up projection outputs, BF16-rounds the SiLU gate, then multiplies
([`mlp_activation.rs`](../../kernels/nvptx/mlp_activation.rs#L54)). The fused
path is a launch and intermediate-traffic candidate, not a matching numerical
contract or parity claim.

## NVFP4 decode also has a distinct A16 route

Ninfer's ordinary `[5120,17408]` NVFP4 down projection chooses A16 GEMV for one
token. Its A4 policy threshold is eight current input tokens, checked by the
dispatcher using the actual token count
([shape](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/nvfp4/shapes/n5120_k17408.cu#L43),
[dispatch](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/nvfp4/nvfp4_dispatch.cpp#L32)).
Its GEMV schedule uses eight warps, two output rows per warp, vectorized FP4-code
loads and staged raw scales with warp broadcasts
([GEMV](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/nvfp4/nvfp4_gemv.cuh#L28)).
The fused NVFP4 gate/up operation also selects A16 for single-token decode, even
when A4 is permitted
([route](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp#L28)).
Our resident MLP quantizes activations to A4 and uses a 16-by-8 warp MMA tile even
for one token. This is a different numerical and execution path. A dedicated
BF16-activation/NVFP4-weight GEMV is a strong next candidate because NVFP4 now
accounts for the largest measured kernel total. It needs a separate independent
reference and end-to-end quality qualification; replacing A4 with A16 will not
preserve current fixture bits. Increasing the current MMA block to four warps
was tested and rejected, so that geometry change alone is not the solution.

## Prefill tiling and weight reuse

Ninfer's default A8 schedule is a 64-token by 128-output tile with K128 tiles,
two shared-memory stages and a ping-pong fragment pipeline
([schedule](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_a8_schedule.cuh#L6)).
Its implementation stages inputs, loads shared matrix fragments and overlaps
subsequent tiles
([kernel](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/linear/fp8/fp8_a8_mma.cuh#L145)).
Our new exact FP8 path has no shared-memory matrix tile: the prefill experiment
reuses one decoded weight across only four tokens, and the decode variant owns
one output column per warp. This explains an architectural opportunity, not a
measured attribution of the complete Ninfer gap. A pipelined prefill GEMM is the
next substantial kernel project. The current exact integer dot removes a costly
FP64 fallback, but it does not exploit FP8 tensor cores. Adopting a different
accumulation profile must retain an independent oracle and qualify model quality;
Ninfer's FP32 accumulation cannot be assumed bit-identical to our scalar contract.

## Fused projection entrypoints

Ninfer provides a single attention input-projection entrypoint that produces
query, gate, key, and value. Its weight preparation can join contiguous rows,
or retain paired parents when the format requires that route ([attention call](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/execution/attention.cpp#L35),
[weight preparation](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/weight_input.cpp#L115)).
Its GDN projection entrypoint similarly produces QKV and Z together
([call](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/execution/gdn.cpp#L61)).
Our resident attention currently invokes Q, K, and V projections separately;
GDN invokes QKV and Z separately. A combined route could reduce repeated input
quantization and launch count, but the source comparison does not establish the
selected Ninfer kernel count or a speedup for our checkpoint.

Ninfer also exposes one GDN norm/control operation accepting A/B either as
joined weights or a paired-weight form ([weight preparation](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/ops/weight_input.cpp#L189),
[execution call](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/execution/gdn.cpp#L71)).
Our path calls input normalization and A/B projections separately. The high-level
entrypoint makes this a bounded candidate for the remaining BF16 cost; it does
not alone show that all stages fuse into one device kernel.

## Allocation, graph replay, and measured boundary

Ninfer constructs a planned workspace once in `ProgramImpl` and backs its arena
with one device allocation ([program](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/program/program_impl.cpp#L38),
[arena](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/core/arena.cu#L141)).
It can capture ordinary decode profiles during preparation and replay a prepared
graph with `cudaGraphLaunch` ([capture](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/program/graphs.cpp#L304),
[replay](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/src/models/qwen3_5/program/graph_execution.h#L11)).
Our wrappers allocate temporary CUDA buffers per projection. Arena reuse and
graph replay are plausible submission/allocation optimizations, but the current
profile does not isolate their effect. Its per-launch event waits change host
scheduling, and event intervals do not measure allocator or host cost
([profile limits](model-profile.md#full-decode-kernel-attribution)).

The first full-model profile is historical: `fp8_linear_wide` accounted for
83.39% of 633.56 ms summed event time and BF16 linear for 11.26%. After the first
exact-integer FP8 iteration, local short-prefix decode was 8.029-8.031 tokens/s
and 128-token-prefix decode was 7.242-7.244 tokens/s, versus about 1.55 tokens/s
in the earlier local trial. That iteration reported unchanged output IDs and
71.57 ms for BF16 gates as its next target. These are existing local measurements,
not Ninfer results; later optimization iterations are tracked in the [decode
projection record](../optimizations/decode-projections.md). The 2026-09-26
Ninfer baseline uses different requests, MTP4, FP8 KV, and other serving settings,
so it is not a matched benchmark ([baseline boundary](ninfer-baseline-20260926.md)).

No GPU run was made for this source comparison. Architecture, driver, clocks,
and toolchain are not applicable to the review itself; local measurement conditions
belong to the linked trial records. The source archive path is
`target/specialize/perf-20260927/ninfer-source/`; re-reading that archive at its
pinned revision is the reproduction procedure. The durable rule is to treat
Ninfer's per-shape route, activation quantization, and rounding boundaries as
separate choices, and to qualify any adopted path against the independent local
reference and model checks before making a parity or speed claim.
