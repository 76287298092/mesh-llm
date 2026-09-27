# Post-attention residual normalization and MLP input quantization

Status: implementation pending GPU qualification. Full layer/model execution and
performance remain unmeasured. This extends the resident layer-zero attention
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
retained. Linux compilation, sanitizer results and measured resource usage will
be recorded after qualification.

Reproduce with `just specialize-ptx`, `just specialize-tools-build`, then
`xtask specialize qwen-projection-check` using the pinned raw-v1 Mspec artifact.
Fresh raw evidence belongs under `target/specialize/qwen-residual-norm-20260927/`.
The JSON is schema 7 and includes residual norm and NVFP4 fixture/real-input
reports. Clocks, runtime timings, model prefill/decode/context and full-model
memory are not measured by this implementation entry.
