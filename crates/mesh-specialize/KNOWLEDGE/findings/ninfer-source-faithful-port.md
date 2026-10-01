# Source-faithful decode port plan

Status: active plan for native MTP and decode porting. Operator evidence lives in
the native MTP Q4/Q8/residency and target-batch findings; native MTP admission,
whole-model quality and throughput remain open.

Source pin: Ninfer `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`.
The pinned repository has an Apache-2.0 LICENSE. Preserve its license,
attribution, and modification notices when adapting source. Correspondence
between this source revision and the running Ninfer service binary remains
unverified.

## Execution strategy

Port the selected source schedule, storage addressing, accumulation order,
and fused epilogue together. Translating the mathematical equation alone is
not a performance-equivalent port. Retain the independent reference and
existing model-quality thresholds; source similarity is not qualification.

For ordinary NVFP4 gate/up decode, `nvfp4_linear_swiglu_decode.cu` selects
`Nvfp4A16GemvSchedule<8,2,16,4,Direct,Default,2>` with static K=5120.
`nvfp4_a16_gemv.cuh` implements eight warps, two parent rows per warp,
16 values per lane, four FP32 accumulation chains, packed code loads,
BF16-pair activation loads, and a fused SwiGLU epilogue. The joined gate/up
parent uses M128 row permutation and tiled scales. The pending Rust A16
prototype instead uses separate canonical planes and diagnostic outputs;
it is not this source schedule and has no model-quality qualification.

FP8 single-token routes also require per-operation selection. The ordinary
source GEMV schedule uses eight warps, two rows, eight values, and four
accumulation chains. Fused GDN input/convolution, attention QKGV, and output
residual paths must retain their epilogues. FP8 MLP gate/up uses an A8 route,
and the target head uses sliced-K. Do not apply a blanket A16 replacement.

For the Q4 proposal head shape N=131072, K=5120, the source T=1 route selects
`GemvR4W1`, with packed-word decoding, FP16-mantissa scaling, asynchronous
vector loads, and shared BF16 pairs. The pending scalar Rust Q4 consumer
is a correctness prototype, not a source-faithful performance port.

Native MTP also needs the source's multi-row draft initialization and
single-row continuation paths, packed Q8 projections, the indexed Q4 head,
and target verification with rollback. The qualified synthetic Q8 operator
does not establish any of these whole-model properties. Keep native MTP
admission closed until the complete path is validated.

## Acceptance

For each port, record its source specialization and physical layout, run
independent operator checks and sanitizers, then measure the whole model.
Arithmetic-changing candidates additionally require decode-aware scoring,
the fixed quality gates, and repeated full-logit determinism checks.
Report ordinary decode and accepted-token MTP throughput separately.

The current measured exact path remains 45.10/40.53 decode tokens per second
at 106/512 input tokens. No speedup or quality result is claimed here for
the pending source-faithful ports.

## Native MTP consumer boundaries

The pinned source's `execution/text.cpp:287` normalizes the BF16 embedding
and incoming hidden state separately, packs them, projects FC to BF16, and
normalizes that result before the attention projections. Its complete
`mtp_projection` computes the full physical Q/K/gate/V parent before splitting.
Incremental prefill can instead use separate K/V and last-row Q/gate routes.
Those routes have different selected Q8 schedules and require separate checks.

The MTP tail projects attention output to BF16 before `residual_add`. Its
`ffn(..., true)` branch projects gate/up to BF16, applies `silu_mul`, projects
down to BF16, and then adds the residual. Ordinary target fused residual and
SwiGLU epilogues must not replace these MTP boundaries.

Attention gating also differs from the existing resident helper. The source's
`ops/kernel/sigmoid_gate_mul.cuh:20` multiplies the BF16 attention value by the
FP32 sigmoid and rounds only the product to BF16. The existing Rust
`attention_gate_bf16` first rounds sigmoid to BF16, then multiplies. Its stable
`ex2.approx` evaluation also differs from the source's `expf` expression.
Likewise, the existing FP64-polynomial SiLU helper is not the source's FP32
`silu` evaluation. These are observed implementation differences, not measured
model-quality failures. Native MTP integration needs independently checked
consumers; reusing the current helpers does not establish source arithmetic.

## Native MTP norm convention

The pinned `execution/text.cpp` passes `unit_offset=true` for all seven MTP
norm roles: embedding, hidden, input, query, key, post-attention, and final.
`include/ninfer/ops/rmsnorm.h` defines that flag as `gain = 1 + weight`.
Consequently, the native BF16 norm words must remain unchanged in residency;
the existing Rust kernel's FP32 `1 + BF16(weight)` is the correct gain
convention. A loader's `direct` binding does not imply plain-gamma arithmetic.

This resolves the gain convention only. The source uses BF16-pair partials,
warp/block reductions, and `rsqrtf`; `embedding_norm_bf16` currently uses
scalar partials, a shared-memory tree, and rounded square root/division.
Identical gain semantics do not establish bit-identical norm outputs or
qualify the complete native MTP step.
