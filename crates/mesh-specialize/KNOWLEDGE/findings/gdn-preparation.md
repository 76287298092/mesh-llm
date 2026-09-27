# GDN normalization and gates

Status: implementation and local host checks pass; GPU qualification is pending.
Recurrent matrix updates and complete attention/model execution remain open.

The compiled layer-zero schedule has 16 key heads, 48 value heads and width 128.
The convolution output contains all Q heads, then all K heads, then all V heads.
The [pinned Transformers reference](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L441)
converts Q/K to FP32, normalizes each vector by its L2 norm with epsilon 1e-6,
and divides queries by sqrt(head width). This implementation retains 16
unrepeated Q/K heads. The recurrent schedule must map value head `h` to key head
`h / 3`, matching repeat-interleave, rather than `h % 16`.

The same upstream forward path produces BF16 `beta = sigmoid(b)` and FP32
`g = -exp(A_log) * softplus(a + dt_bias)`. Recurrent execution multiplies state
by `exp(g)`. The Rust gate kernel emits BF16 beta, FP32 log decay and FP32 decay.
The pinned BF16 A_log/dt_bias parameters come from the verified artifact, and
A/B inputs are the resident BF16 projection buffers already checked against
their independent dense references. Q/K inputs are the retained, qualified
whole-sequence convolution output. Neither path uploads a CPU replacement.

Each Q/K block uses 256 threads and two shared reductions. The GPU accumulates
squares in FP32; the CPU oracle independently uses f64 sums with FP32 normalized
outputs. Gate math uses stable sigmoid and softplus, including the softplus
linear branch above 20 and a fifth-order log1p polynomial below 0.0625.
The remaining log1p range uses approximate PTX log2. Exponentials use
`ex2.approx.f32` without FTZ so representable subnormal sigmoid gates survive.
The [NVIDIA PTX contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2)
supports subnormal inputs/results on SM20 and later by default. The parent
replaced the initial FTZ/clamped draft before device qualification and added
fixtures for sigmoid inputs -88 through -93. Prior convolution/SiLU's separately
documented FTZ profile is unchanged. No vendor math library is introduced.

The host validates dimensions, exact extents, finite values, distinct A/B
projection indices and matching QKV/parameter shapes. A_log is restricted to
[-80,80] for this experimental kernel, and overflow is rejected. The CPU oracle
uses f64 exp/log1p and no PTX arithmetic helpers. Q/K acceptance is
`2e-6 + 2e-6 * abs(reference)`. Log decay and decay acceptance is
`3e-6 + 5e-6 * abs(reference)`. Beta must be finite, in [0,1], and within one
BF16 ULP of the scalar reference. All beta reference differences are reported.
Gate sign and decay range are checked separately. These are component-level
numeric contracts, not proof of model-output parity.

Real cases use one and 17 tokens. Dedicated fixtures cover zero Q/K vectors,
distinct heads, eight-wide reductions, tail threads, signed gates, large finite
inputs, the softplus threshold, subnormal beta and A_log endpoints. The runtime
still emits null model performance/context fields and `model_executable:false`.
The reference tests also reject malformed dimensions, ratios, extents, nonfinite
values and arithmetic overflow.

Two bounded Luna-max workers own the device kernel and independent scalar
reference. The parent owns the precision profile, retained device buffers,
model-specific loading, comparisons and deployment. Local validation passes 102
library tests, focused all-target/all-feature Clippy with warnings denied and
Rust PTX compilation. Linux and GPU evidence will be captured separately under
`target/specialize/qwen-gdn-prepare-20260927/` using `qwen-projection-check`, now
report schema 4. The previous convolution and projection reports remain intact.
