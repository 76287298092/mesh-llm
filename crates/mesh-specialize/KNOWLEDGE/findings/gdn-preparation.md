# GDN normalization and gates

Status: real-weight GPU comparisons and all three CUDA sanitizers pass.
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
Rust PTX compilation. `qwen-projection-check` now emits report schema 4. The
previous convolution and projection reports remain intact.

## Carrack qualification

Source `d802372bdc6dbd4250f73ceff6e0aa6c0b069e48` passed on RTX 5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09 and
driver API 13040. Release xtask SHA-256 is
`04a6994c42a5a3863229ef5c0331a3093fb8398dc1474bafbf8d267fdbdb3033`;
PTX SHA-256 is `2d62cdb7ed5e99736a0a4c330a512d91cbf16b4ce0a53a5db55412cb0b1f8d6c`.
The Linux host used Rust 1.98.1 and LLVM 22.1.8. Device PTX was built on the Mac
using pinned nightly-2026-09-25 and rebuilt NVPTX core, then transferred intact.
CUDA 13.4.92 ptxas reports 23 registers and 2,048 shared bytes for Q/K norm,
and 21 registers with no shared memory for gates. Neither has local memory or
spills. The kernels use one and zero barriers respectively.

The [normal report](../evidence/qwen-gdn-prepare-20260927/normal.json) passes
73,728 real normalized Q/K values and 2,592 gate values across one and 17 tokens.
Maximum Q error is `1.4901161193847656e-8`; maximum K error is
`1.1920928955078125e-7`. All 864 BF16 beta values exactly match the scalar oracle.
Maximum log-decay error is `2.384185791015625e-7`; maximum decay error is
`8.940696716308594e-8`. The resident projection and convolution regressions also
pass. These are checks against component oracles, not complete GDN/model parity.

Both dedicated fixtures pass, including exact BF16 beta at subnormal inputs.
The extreme fixture's log-decay absolute error reaches `9.903520314283042e27`
at its huge finite magnitude; it satisfies the declared relative bound. That
fixture must not be described using the much smaller real-weight error maximum.

[Memcheck](../evidence/qwen-gdn-prepare-20260927/memcheck.log),
[racecheck](../evidence/qwen-gdn-prepare-20260927/racecheck.log) and
[synccheck](../evidence/qwen-gdn-prepare-20260927/synccheck.log) each report zero
errors or hazards. All numerical checks pass in each run. Every run is bounded
by an 8 GiB host memory limit, no swap and a 240-second timeout. Linux passes
109 library tests, 17 validator tests and focused all-target/all-feature Clippy
with warnings denied. No GitHub Actions run is claimed.

Reproduce with `just specialize-ptx` on the pinned Mac toolchain and
`just specialize-tools-build` on Carrack, then `xtask specialize
qwen-projection-check` using the pinned `.mspec`, device 0 and a fresh report.
The normal command takes 12.311 seconds, including artifact verification,
uploads and CPU calculations. It is not a model or kernel throughput measurement.
Driver free memory before and after temporary allocations is 32,221,822,976 bytes.
No full-model peak memory or context capacity is inferred from these samples.

Ninfer restarted at 02:19:42 EDT on September 27 as PID 2862526, reached
engine-ready at 02:19:48 and returned HTTP 200 from `/health`. Its sampled
allocation is 30,046 MiB. ComfyUI PID 448118 remained resident at 498 MiB.
The [evidence directory](../evidence/qwen-gdn-prepare-20260927/) contains the
reports, service records and test summaries. Raw build logs and exact PTX remain
under `target/specialize/qwen-gdn-prepare-20260927/` on both hosts.

The subsequent [recurrent update trial](gdn-recurrence.md) now retains these
prepared buffers and qualifies the matrix update. Gated output normalization,
output projection, full-attention layers, MLP execution, model scheduling and
serving are still required.
