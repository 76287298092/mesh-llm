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
