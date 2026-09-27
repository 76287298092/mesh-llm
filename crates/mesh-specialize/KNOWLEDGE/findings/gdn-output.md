# GDN gated normalization and output projection

Status: implementation under local validation. GPU qualification is pending.
This connects the remaining layer-zero GDN attention operations. Model execution,
independent full-layer/logit parity and performance comparisons remain open.

The pinned [Transformers gated norm](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L218)
normalizes each 128-value head in FP32, rounds normalized values to BF16,
multiplies the direct BF16 norm weight, rounds that product to BF16, then gates
with FP32 SiLU of Z and rounds the final result to BF16. Gamma is not
zero-centered. One shared BF16 weight vector applies to all groups. Epsilon is
1e-6 in the pinned checkpoint. These rounding points differ from a fused path
that keeps every intermediate in FP32; this experiment preserves the fallback
semantics explicitly.

A Rust kernel uses one 256-thread block per head/token group, shared FP32 square
reduction, explicit rounded add/multiply/divide/square-root, and non-FTZ
`ex2.approx.f32` for stable SiLU. It exposes normalized FP32, weighted BF16,
SiLU FP32 and final FP32 values for independent phase checks. The scalar oracle
uses f64 norm accumulation and f64 SiLU. FP32 normalization must satisfy
`2e-6 + 2e-6 * abs(reference)`; SiLU must satisfy
`3e-6 + 5e-6 * abs(reference)`. Intermediate weighted BF16 and final BF16
rounding must exactly match their GPU FP32 precursors. Full scalar BF16
differences are reported separately, not hidden by those component bounds.

The host retains the whole-sequence recurrent BF16 output and the original
resident FP8 Z projection output. Neither is replaced by a CPU oracle result.
After gated norm, the existing qualified FP8 activation quantizer produces
per-token E4M3FN codes and FP32 scales. The output projection uses pinned
E4M3 weights shaped `[5120,6144]` with BF16 per-channel scales. Every activation
code and scale is independently compared, and the existing f64 dot-product
oracle checks output projection arithmetic and BF16 rounding.

One and 17 token sequences exercise real layer-zero weights. Dedicated norm
fixtures cover shared signed gamma, zero rows, signed and extreme gates, and
widths one, eight and 256. The recurrent kernel/state and all earlier projection,
convolution and preparation checks remain in the command. Whole/chunk state
qualification concerns recurrence; the entire layer is not yet scheduled as a
stateful serving session or compared against an independent full-layer engine.

Two bounded Luna-max workers own the device kernel and scalar reference. The
parent owns the precision contract, retained device inputs, artifact parameters,
output projection reuse, comparisons and deployment. Source review corrected
the initial reference's gamma extent/indexing to a shared width-length vector
and corrected a negative SiLU underflow test. The reference also rejects an
FP32-overflowing square sum before division by width. No serving or public ABI
change is included.

Reproduction extends `qwen-projection-check` to schema 6, using fresh evidence
under `target/specialize/qwen-gdn-output-20260927/`. Build PTX through
`just specialize-ptx` and the Linux host through `just specialize-tools-build`.
Model throughput/context fields remain null and `model_executable` remains false.

Local validation passes 112 library tests, focused all-target/all-feature Clippy
with warnings denied, the repository no-console check, formatting and Rust PTX
compilation. The initial test run caught an incorrect unsigned-zero expectation
for negative gamma. Its fixture now checks the signed BF16 zero produced by the
specified arithmetic; the kernel/reference arithmetic did not change. The failed
log is retained under the trial directory.
