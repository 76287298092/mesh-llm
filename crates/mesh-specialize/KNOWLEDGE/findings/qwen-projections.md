# Layer-zero FP8 and BF16 projections

Status: FP8 QKV/Z passes real-weight GPU execution and all three sanitizers.
BF16 A/B implementation and host checks pass; its Carrack execution is pending.
Full recurrent attention and model execution remain open.

The pinned checkpoint config specifies dynamic per-token FP8 input activation
quantization for `in_proj_qkv` and `in_proj_z`, with static per-channel FP8 weight
scales. The small `in_proj_a` and `in_proj_b` weights are BF16 matrices with
48 output channels. Their new kernel consumes the same resident normalized BF16
activation directly. It has no activation quantization or scale factors.

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

Bounded
Luna-max workers implemented the quantizer and FP8 linear kernel; the parent
owns the numerical contract, independent reference, host validation, chain,
model-specific loading, CLI and remote run.

Local checks: 92 macOS library tests, focused all-target/all-feature Clippy with
warnings denied, Rust PTX compilation and repository no-console checks pass.
The CLI now shares report persistence between entry and projection checks; a
missing direct anyhow dependency was avoided by making the runner generic over
its error display type. No dependency was added.

Carrack's first check passed 97 library tests and 17 validator tests. Linux-only
Clippy then found a fixed-size `vec!` in the quantizer fixture; it was replaced by
a stack array without changing values or execution. The initial lint log is
retained. Offline CUDA assembly accepts the new FP8 instructions for SM120a;
actual JIT/launch and numerical qualification are recorded below.

## First FP8 GPU trial

Source `ca1b076f7d191995e953905f870ebdb7f4fe605d` passed on Carrack's RTX 5090,
UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09,
driver API 13040. The release xtask SHA-256 was
`b30bc1a5a85355f8af97242a63e844bcc1d14282e6fa1f9e4f8da83429de6b3a`;
PTX built with nightly-2026-09-25 had SHA-256
`f9e8a2b8cd25d03ec50c07bca624a931308e10256c2ac5417ebbc65eeb1d1d46`.

The [normal report](../evidence/qwen-projections-20260927/normal.json) covers
294,912 real QKV/Z outputs across M1 and M17. All FP32 values meet the declared
error bound; maximum absolute error is 0.0000457763671875. Every BF16 output
exactly rounds its GPU FP32 accumulator. Forty-four BF16 outputs differ from
rounding the f64-based reference, so this is tolerance-based numerical evidence,
not exact reference BF16 parity. Activation codes and scales match exactly.
The signed tail fixture adds 221 outputs and the quantizer fixture checks 1,518
values, including all finite FP8 codes, ties, zero and tiny BF16 values.

[Memcheck](../evidence/qwen-projections-20260927/memcheck.log),
[racecheck](../evidence/qwen-projections-20260927/racecheck.log) and
[synccheck](../evidence/qwen-projections-20260927/synccheck.log) each report zero
errors or hazards. Numerical comparisons also pass under every sanitizer. Each
command ran in an 8 GiB user scope with swap disabled and a 240-second timeout.
The linear kernel uses 35 registers and zero local/shared memory; the quantizer
uses 22 registers and 1,024 bytes of shared memory. No register spills appear.

Ninfer was restored after the trial, reached engine-ready at 01:35:57 EDT on
September 27, and returned HTTP 200 from `/health` as PID 2844092. ComfyUI PID
448118 stayed resident throughout. The 12.714-second command duration includes
hashing, upload and CPU reference work; it is not an inference throughput result.
Driver free memory returned to 32,221,822,976 bytes after trial allocations were
released. Full-model residency, context capacity, prefill and decode remain
unmeasured. Raw PTX, logs and reports remain in
`target/specialize/qwen-projections-20260927/` on both hosts.

## BF16 A/B extension

The new scale-free BF16 linear kernel uses `m16n8k16` with FP32 accumulation,
row-major input and checkpoint weights, masked M/N/K tails and one final BF16
rounding. The scalar reference operates on logical matrices with f64 products
and sums. The existing projection acceptance and separate BF16 rounding checks
apply. The arithmetic reference now lives in `reference/projections.rs`, since
both low-precision formats share the result contract.

Real tests add A/B at M1/M17, N48, K5120. A separate signed M17/N13/K71 fixture
checks all tails. A/B reads the original normalized device buffer, never the FP8
codes. Completing these projections still leaves convolution, gates, recurrent
state, GDN normalization and output projection before layer-zero attention works.
The BF16 worker owns only the device kernel; the parent owns reference arithmetic,
loading, launch/checks and integration. GPU evidence for this extension follows
in a separate directory so the first FP8 trial remains reproducible.
