# Layer-zero FP8 projections

Status: implementation and host checks pass; Carrack execution is pending.
Full recurrent attention and model execution remain open.

The pinned checkpoint config specifies dynamic per-token FP8 input activation
quantization for `in_proj_qkv` and `in_proj_z`, with static per-channel FP8 weight
scales. The small `in_proj_a` and `in_proj_b` weights are BF16 and are deliberately
tracked as the next separate operation, not claimed as executed here.

The GPU chain now keeps embedding/norm output resident, quantizes it to E4M3FN,
and feeds those device bytes/scales directly to a tiled FP8 tensor-core linear
kernel. It reads the original row-major quantized weights without an offline
fragment repack or a second weight representation. The first untuned kernel uses
one warp per 16x8 output tile and 32 K values per MMA instruction. Tail rows,
columns and K values are zero-padded on load and masked on store.

## Numerical contract

This experimental execution profile computes the token absolute maximum in FP32,
uses `amax / 448` as its FP32 scale, and replaces a zero scale with one. Conversion
is round-to-nearest-even with finite saturation at +/-448. Weight scales are the
original BF16 values. FP32 accumulators receive the row scale and then the channel
scale, and the final result rounds once to BF16. The quantizer currently uses a
software binary search over finite FP8 values; hardware conversion and further
kernel tuning remain separate work.

The checkpoint's token/channel quantization semantics follow its config and the
[compressed-tensors quantization helpers at 47f7d42](https://github.com/vllm-project/compressed-tensors/blob/47f7d42fabb314f674d259c98a176a6f01a41df8/src/compressed_tensors/quantization/utils/helpers.py).
FP32 activation-scale storage is an explicit runtime choice. This is not a claim
of bitwise parity with that package's BF16 fake-quantization path or with NInfer's
conversion and kernels. Those comparisons require end-to-end reference execution.

The [NVIDIA PTX matrix-fragment contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832)
defines the register/lane mapping for the E4M3 `m16n8k32` MMA. The independent CPU
reference operates on logical matrices, accumulates products in f64, and uses a
brute-force nearest-value FP8 encoder. It shares no fragment packing or device
arithmetic. GPU quantized bytes and scales must match exactly. Projection FP32
acceptance is `1e-6 + 2e-6 * scaled_sum(abs(products))`, an explicit absolute
error allowance that remains meaningful near cancellation. BF16 must exactly
match independent rounding of the GPU FP32 output; differences from rounding the
f64-based reference are reported separately.

A signed M17/N13/K71 fixture checks all tile tails. Dedicated quantizer fixtures
cover every signed finite FP8 value and midpoint, signed zero, a zero row and
minimum BF16 subnormals. Real QKV and Z trials use M1 and M17, K5120 and
N10240/N6144. Their input normalization must match the independent BF16 reference
before any downstream comparison. These are operation fixtures, not requests,
model prefill/decode or context-capacity measurements.

Run `just specialize-ptx` on the Mac with the pinned nightly, transfer PTX to
Carrack, build there with `just specialize-tools-build`, then execute:

```sh
target/release/xtask specialize qwen-projection-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx PATH --device 0 --output NEW_FILE
```

Validation and exact trial evidence will be recorded after execution. Bounded
Luna-max workers implemented the quantizer and FP8 linear kernel; the parent
owns the numerical contract, independent reference, host validation, chain,
model-specific loading, CLI and remote run.

Local checks: 92 macOS library tests, focused all-target/all-feature Clippy with
warnings denied, Rust PTX compilation and repository no-console checks pass.
The CLI now shares report persistence between entry and projection checks; a
missing direct anyhow dependency was avoided by making the runner generic over
its error display type. No dependency was added.
