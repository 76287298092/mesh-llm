# Independent whole GDN layer reference

Status: implementation and fixed criteria recorded before the real comparison.
This gate addresses accumulated error through the entire first decoder layer.
It does not execute or qualify the full 64-layer model or its logits.

The independent Rust CPU path receives only the verified checkpoint weights,
original token IDs and zero initial convolution/recurrent state. It composes the
existing scalar references from embedding and input norm through QKV/Z/A/B,
convolution, GDN preparation, ordered-FP32 recurrence, gated norm, attention
output projection, first residual and post-attention norm, NVFP4 MLP and final
residual. Every intermediate comes from that CPU path. It never receives a GPU
output or a device intermediate as an oracle input.

The chosen arithmetic profile remains explicit: f64 dot/norm/transcendental
oracles with the documented BF16 boundaries, ordered-FP32 recurrent reductions
and FP32 activation local scales. This compares the implemented quantized model
profile, not a full BF16 checkpoint or another runtime's exact numerical profile.

Before seeing real results, the parent sets an engineering error budget of
normalized L2 at most 0.01 and cosine similarity at least 0.9999. Both must pass
for the final hidden vector, raw convolution history and recurrent state.
The same criteria also apply to each token's hidden vector, each history row
and each recurrent head separately, preventing good aggregate metrics from
masking an isolated failed partition. For a zero reference vector, only a zero
actual vector passes. Finite values and exact extents are mandatory. Reports
include maximum/RMS absolute error and exact FP32/BF16 element counts; these
counts remain diagnostics rather than an exact-parity claim.

The generic comparator uses compensated f64 energy and dot-product sums. A unit
fixture deliberately changes one small partition while keeping aggregate error
small; the partition gate must reject it. Other tests cover scaling, angle,
signed zero, zero energy and malformed/nonfinite vectors. The layer composition
has a tiny zero-branch fixture with nonzero embeddings and explicit residual
identity, plus missing/invalid connection cases.

GPU wrappers now retain the actual final hidden words, convolution history and
recurrent state readbacks for this comparison. They do not replace any device
inputs. The existing component checks remain in place. A failed layer budget
sets the case and overall report false; the command persists the full numerical
report before returning failure, so failed evidence is not lost.

Two bounded Luna-max tasks own the CPU layer composition and generic comparison.
The parent owns their interfaces, fixed thresholds, per-token/head grouping,
retained readbacks, report integration and deployment. No device kernel or
launch changes are included. Existing sanitizer evidence for the identical
kernels and launches remains in the MLP qualification entry; this change needs
fresh host tests/Clippy and a normal real-weight whole-layer trial.

Reproduce with `just specialize-tools-build`, then the existing
`qwen-projection-check` command using the unchanged MLP-trial PTX. Schema 9 adds
`whole_layer_reference`. Fresh evidence belongs under
`target/specialize/qwen-layer-reference-20260927/`. Model performance/context
fields remain null. Timing from this harness includes independent CPU execution
and cannot represent inference throughput.

Local tests (143) and Clippy passed; Linux passed 154 library and 17 validator
 tests, then its CUDA-only Clippy path rejected placing `compare_layer` after the
 test module. Moving that helper before the tests fixes source ordering without
 changing behavior. Preserve the initial Linux Clippy log alongside the rerun.
