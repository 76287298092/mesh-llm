# Causal convolution and SiLU

Status: implementation and local host checks pass; GPU execution is pending. No complete GDN layer,
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
new report path. Real GPU and sanitizer evidence will follow in a distinct
`qwen-convolution-20260927` directory without replacing prior projection trials.

Local validation passes 98 library tests, focused all-target/all-feature Clippy
with warnings denied, Rust PTX compilation and repository no-console checks.
Clippy also requested `as_chunks` for a fixed-size test array; its initial log
is retained. A read-only review found no kernel/host ABI, extent, lifetime,
history ownership or tail-masking mismatch. Linux compilation and real device
execution remain separate qualification gates.
