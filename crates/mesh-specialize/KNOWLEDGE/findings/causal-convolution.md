# Causal convolution and SiLU

Status: real-weight GPU execution, chunk/state equivalence and all three
sanitizers pass. No complete GDN layer,
model execution or model-performance result is claimed.

The pinned checkpoint has BF16 `linear_attn.conv1d.weight` with shape
`[10240,1,4]`, no bias and SiLU activation. The
[pinned Transformers fallback](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L249)
applies causal depthwise convolution and then SiLU to its BF16 output. This
experiment preserves that intermediate rounding. Fused third-party convolution
kernels may round at different points; parity with those paths or NInfer is not
claimed. The retained source file has SHA-256
`f567c3e4b7db57b5d04be57a8fe9788e7c19aaa01aad877ee0daf94b2dab1925`.

The Rust kernel reads the already-qualified resident QKV projection buffer. It
uses four ordered FP32 multiply/add pairs per output, rounds to BF16, evaluates
SiLU in FP32, and rounds again to BF16. The SiLU implementation uses a stable
sigmoid with `exp(-abs(x))`, PTX `ex2.approx.ftz.f32`, and explicit rounded
FP32 arithmetic. Exponents below -126 produce zero to avoid FTZ underflow;
this intentionally loses extremely small negative SiLU tails. The numerical
acceptance is absolute-plus-relative and includes an extreme-value fixture.
No CUDA C++, libdevice or vendor compute library is used.

Input and output are time-major `[tokens,channels]`. Weight taps are oldest to
newest, channel-major `[channels,4]`. Three raw pre-convolution input rows are
sufficient history. A launch reads an immutable input history and writes a
distinct output history, with one last-token owner per channel. The host swaps
two device state buffers between chunks. State is downloaded for verification
but never replaced with a host-generated result between chunks. This avoids a
read/write race while preserving the minimal convolution state. The private
three-row state is not a public cache ABI or compatibility claim.

The independent CPU reference operates on logical matrices, accumulates products
in f64, rounds convolution to BF16, and uses a stable f64 SiLU. It also derives
the next raw history independently. Each GPU convolution FP32 result must meet
`1e-6 + 2e-6 * sum(abs(products))` versus that reference. SiLU is checked
separately against f64 SiLU of the rounded GPU convolution, within
`2e-6 + 2e-6 * abs(reference)`. Final BF16 must exactly round GPU SiLU FP32.
Full scalar BF16 output differences and activation errors are reported separately
to distinguish convolution rounding boundaries from activation error. This is
component-level numerical qualification, not full-model output quality evidence.

The real-weight trial chains QKV for sequences of one and 17 token IDs. For the
17-token case it compares whole-sequence execution with `[1,16]`, `[2,1,14]`
and 17 single-token updates. GPU outputs and final histories must be exactly
equal across those partitions. Each intermediate history must also exactly
match the independent CPU state. Fixtures add 13 channels, seven tokens,
asymmetric signed taps, nonzero history, tail threads, reset behavior and extreme
finite SiLU values. Model context, prefill and decode remain unmeasured.

Two bounded Luna-max workers own the device kernel and scalar reference. The
parent owns semantics, host state lifecycle, resident chaining and qualification.
Initial parent review corrected a reference history offset: append length decides
how many old rows to drop, not how many old rows remain. The first local compile
also caught a non-constant array length in a test. Its log is retained under
`target/specialize/qwen-convolution-20260927/local-tests-initial.log`.

Reproduction uses the existing `qwen-projection-check` command, which now includes
convolution checks after QKV and emits schema version 3. Build device PTX on the
Mac through `just specialize-ptx`, build Carrack's host through
`just specialize-tools-build`, and run with the pinned `.mspec`, device 0 and a
new report path. Real GPU and sanitizer evidence is retained in a distinct
`qwen-convolution-20260927` directory without replacing prior projection trials.

Local validation passes 98 library tests, focused all-target/all-feature Clippy
with warnings denied, Rust PTX compilation and repository no-console checks.
Clippy also requested `as_chunks` for a fixed-size test array; its initial log
is retained. A read-only review found no kernel/host ABI, extent, lifetime,
history ownership or tail-masking mismatch. Linux compilation and real device
execution are recorded below.

## Carrack qualification

Source `7d81bea889d941a09cbe7d47faa95af68685924e` passed on RTX 5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09 and
driver API 13040. Release xtask SHA-256 is
`9776541379407efb24cd5dbb451ce95ba81da63559803ca90dd3178776130389`;
PTX SHA-256 is `7d06167709aaed0aa1b21f6a180bed5c034b7953f56f420da75d228fdc7839be`.
The host used Rust 1.98.1 and LLVM 22.1.8; the device build used pinned
nightly-2026-09-25 with rebuilt NVPTX core. ptxas and the driver report
26 registers, zero local/shared memory and no spills for the new kernel.

The [normal report](../evidence/qwen-convolution-20260927/normal.json) passes
184,320 real-weight convolution/SiLU outputs across one and 17 tokens. Maximum
convolution FP32 error is `1.1920928955078125e-7`; maximum SiLU error is
`9.5367431640625e-7`. The complete scalar activation comparison has the same
maximum error. Every BF16 output exactly rounds GPU SiLU FP32, while 33 outputs
differ from the scalar BF16 reference. Those counts are preserved. This does not
claim exact scalar BF16 parity. All four projection comparisons continue to pass.

Whole-sequence, `[1,16]`, `[2,1,14]`, and 17 one-token executions have exactly
equal GPU BF16 outputs and final raw histories. Every intermediate state matches
the independent reference. The signed seven-token, 13-channel fixture also
passes all partitions with nonzero initial history. Its scalar BF16 outputs
match exactly. The extreme-value fixture passes the numerical bound with one
BF16 reference difference caused by flushing a tiny negative SiLU tail.

[Memcheck](../evidence/qwen-convolution-20260927/memcheck.log),
[racecheck](../evidence/qwen-convolution-20260927/racecheck.log), and
[synccheck](../evidence/qwen-convolution-20260927/synccheck.log) each report zero
errors or hazards; all numerical and state checks also pass in each run. Every
run has an 8 GiB host memory limit, no swap and a 240-second timeout. Carrack
passes 104 library tests, 17 validator tests and focused all-target/all-feature
Clippy with warnings denied. No GitHub Actions run exists for this branch.

The normal command takes 12.193 seconds including file verification, uploads and
CPU oracle calculations. It does not time model or kernel throughput. Driver
free memory before and after temporary allocations is 32,221,822,976 bytes.
No full-model memory peak or context capacity is inferred from this operation.
Model throughput/context fields remain null and `model_executable` remains false.

Ninfer restarted at 02:02:44 EDT on September 27 as PID 2855077, reached
engine-ready at 02:02:50, and returned HTTP 200 from `/health`. It retains the
30,046 MiB allocation. ComfyUI PID 448118 remained resident at 498 MiB.
Before/during/after records, test summaries and initial failed host checks are
in the [evidence directory](../evidence/qwen-convolution-20260927/). Raw logs and
exact PTX remain in `target/specialize/qwen-convolution-20260927/` on both hosts.

One report-label clarification follows this qualified source: `reference_input`
now describes a host BF16 mirror generally. In the captured fixture reports its
older wording mentions a projection, although those dedicated fixtures use
literal input arrays. Real cases use the resident QKV projection as stated;
arithmetic, state checks and device data flow are unchanged by the label fix.

This qualifies causal convolution state only. GDN gate transforms, Q/K
normalization, the recurrent matrix update, gated output normalization and output
projection remain before complete layer-zero attention can run. Session management,
full-attention layers, MLP execution and model serving are still open.
