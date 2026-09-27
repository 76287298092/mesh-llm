# Post-attention residual normalization and MLP input quantization

Status: real-weight component trial and all three sanitizers pass. Full layer/model
execution and performance remain unmeasured. This extends the resident layer-zero attention
chain without adding serving or public ABI changes.

The pinned Transformers decoder at revision
`96331a9f93b72697f160a958d2883d4b49a56739` adds the attention result to the
original BF16 residual, rounds that sum to BF16, then applies FP32 RMSNorm with
zero-centered `1 + weight` before rounding to BF16. The real checkpoint provides
`layers.0.post_attention_layernorm.weight` shaped `[5120]` and epsilon 1e-6.
[Source](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L920).

A Rust kernel uses one 256-thread block per row. Explicit rounded FP32 arithmetic
adds residual and attention output, stores BF16, and reduces squares of that
rounded value. The independent scalar reference uses f64 square accumulation.
Residual BF16 must match exactly. Normalized FP32 must satisfy
`2e-6 + 2e-6 * abs(reference)` and final BF16 must exactly round the device FP32
value. Scalar BF16 differences are reported separately. Fixtures cover widths
1, 71 and 5120, rounding ties, cancellation, signed zero and signed shared gamma.
The input buffers are the original resident embedding residual and actual
resident attention output, never replacement CPU-oracle buffers.

The MLP gate/up inputs use NVFP4 groups of 16. Their individual F32 checkpoint
`input_global_scale` tensors are loaded and verified; equality is not assumed.
The chosen profile computes local `(amax / 6) * global` in FP32, encodes E4M3FN
nearest-even with finite saturation, and replaces a zero encoded scale with
0.125. Effective dequantization scale is `decode(local) / global`. Payloads
encode `input / effective` as signed E2M1 nearest-even with saturation to 6.
Negative zero is preserved and consecutive elements occupy low then high nibble.

These format rules are derived from compressed-tensors revision
`47f7d42fabb314f674d259c98a176a6f01a41df8`:
[scale calculation](https://github.com/vllm-project/compressed-tensors/blob/47f7d42fabb314f674d259c98a176a6f01a41df8/src/compressed_tensors/quantization/utils/helpers.py),
[effective scale](https://github.com/vllm-project/compressed-tensors/blob/47f7d42fabb314f674d259c98a176a6f01a41df8/src/compressed_tensors/quantization/lifecycle/forward_helpers.py),
and [packing](https://github.com/vllm-project/compressed-tensors/blob/47f7d42fabb314f674d259c98a176a6f01a41df8/src/compressed_tensors/compressors/nvfp4/helpers.py).
The chosen FP32 local-scale profile does not claim bit parity with BF16
fake-quantization helpers. Full-model logit validation must assess that choice.

One full warp owns each 16-element group. All 32 lanes take part in max and
neighbor shuffles. An independent host encoder exhaustively chooses nearest
representable values; packed bytes, local scales and effective FP32 scale bits
must agree exactly. GPU fixtures include positive/negative FP4 midpoints,
negative zero, FP8 scale ties, zero groups, scale saturation and multiple rows.
Host tests additionally reject nonfinite and overflowing scales/inputs.
The quantizer currently checks and releases its output; no MLP matrix multiply
or decoder-layer completion is claimed.

Two bounded Luna-max workers supplied device kernels and scalar references.
The parent defined the arithmetic and interfaces, connected device allocations,
loaded checkpoint parameters and built the comparison harness. Static review
found no concrete residual-path defect. The first macOS test run passes 122
library tests. Initial Clippy rejected two iterator styles; they were corrected
without changing arithmetic. Initial device compilation warned about unused
width; its ABI parameter is retained and marked unused. Failed/warning logs are
retained. Linux compilation, sanitizer results and measured resource usage are
recorded below.

Reproduce with `just specialize-ptx`, `just specialize-tools-build`, then
`xtask specialize qwen-projection-check` using the pinned raw-v1 Mspec artifact.
Fresh raw evidence belongs under `target/specialize/qwen-residual-norm-20260927/`.
The JSON is schema 7 and includes residual norm and NVFP4 fixture/real-input
reports. Clocks, runtime timings, model prefill/decode/context and full-model
memory are not measured by this implementation entry.

## Carrack qualification

Source `8b56c16ca074f106cb601a0dff6ea51888c0c00c` passes the real-weight
component trial and all three CUDA sanitizers on RTX5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09,
driver API 13040. Linux host toolchain is Rust 1.98.1/LLVM 22.1.8. Device PTX
was emitted on macOS using pinned nightly-2026-09-25 and rebuilt NVPTX core.
CUDA 13.4.92 ptxas and the driver JIT agree on 22 registers/1024 shared bytes
for residual norm and 21 registers/no shared memory for NVFP4 quantization.
Neither kernel spills or uses local memory. Residual norm uses one barrier
identifier; the warp quantizer uses none.

Release xtask SHA-256:
`e806e9715ee84d7e23433beac7e0960dfd884e08d88609ffcaecf9f895245e6b`.
PTX SHA-256:
`bf15638f3f714a64601bf993056c666a2419a26fd966e48869b74d52e1e85f11`.
Mac passes 122 library tests and Linux passes 132 library plus 17 validator tests.
Focused all-target/all-feature Clippy with warnings denied, formatting, repository
no-console check and PTX compilation pass. No GitHub Actions result is claimed.
Static reviews of both paths found no concrete arithmetic or ABI mismatch.

The [normal report](../evidence/qwen-residual-norm-20260927/normal.json)
passes 92,160 real residual/norm values across one and 17 tokens. Residual and
normalized BF16 both match the independent scalar reference exactly for these
inputs. Maximum normalized FP32 difference is `4.76837158203125e-7`, within the
declared bound; all device BF16 rounding checks pass. Another 10,456 dedicated
fixture values pass, including the tail width 71 and width 5120.

The checkpoint gate/up input global multipliers both happen to be 836. Each
is loaded independently. Both sets of quantization checks pass every payload
byte, local E4M3 scale and effective FP32 scale bit: 184,320 values total across
the two projections (the same 92,160 input values quantized twice), with 11,520
scale groups. Another 10,400 dedicated fixture values pass. These comparisons
qualify the specified FP32 local-scale profile; they do not prove parity with a
BF16 fake-quantization path or establish model-logit correctness.

[Memcheck](../evidence/qwen-residual-norm-20260927/memcheck.log),
[racecheck](../evidence/qwen-residual-norm-20260927/racecheck.log), and
[synccheck](../evidence/qwen-residual-norm-20260927/synccheck.log) each report zero
errors or hazards. Every earlier chain/state comparison also passes. Each run
uses an 8 GiB host-memory systemd scope, zero swap and a 240-second timeout.
Normal command duration is 13.496 seconds including artifact checks, uploads and
CPU references. This is not model or kernel throughput. Driver free memory
before and after temporary allocations is 32,221,822,976 bytes, not a measured
full-model allocation peak. Full-model throughput/context fields remain null.

The invocation, after the Just build recipes, is:

```sh
./target/release/xtask specialize qwen-projection-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/qwen-residual-norm-20260927/probes.ptx \
  --device 0 --output NEW_REPORT.json
```

The parent runs this within the bounded systemd scope, stops Ninfer and protects
its restart with an EXIT trap. Sanitizer runs prefix the command with
`/opt/cuda/bin/compute-sanitizer --tool TOOL --error-exitcode 42` and use fresh
report paths. The [evidence directory](../evidence/qwen-residual-norm-20260927/)
contains reports, sanitizer logs, compilation/test evidence including initial
Clippy failures and PTX warning, and service/GPU records. Full build logs, PTX
and cubin remain under the target trial directory on both hosts.

Ninfer restarted at 03:07:36 EDT on September 27 as PID 2896352, reached
engine-ready at 03:07:42, and returned HTTP 200 from `/health`. Its sampled
allocation is 30,046 MiB; ComfyUI PID 448118 remains at 498 MiB.

Remaining work includes actual MLP projections/activation/down projection and
second residual, independent whole-layer/logit qualification, full-attention
layers, whole-model scheduling, stateful serving, tokenizer/sampling, ABI
integration and measured model prefill/decode/memory/context comparisons.
