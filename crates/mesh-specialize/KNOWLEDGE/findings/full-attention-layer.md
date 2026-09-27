# Complete full-attention decoder layer

Status: implementation criteria before measurements. This completes layer 3's
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
