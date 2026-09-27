# F03 FP8 gate/up SwiGLU fusion

Status: first bounded candidate source is written. It has not been compiled,
assembled, launched, compared on GPU, sanitized, integrated into a model path,
or benchmarked. The existing exact projection and separate activation path
remain the qualified control.

## Kernel contract

`kernels/nvptx/fp8_swiglu_exact.rs` exports `fp8_swiglu_exact` with this ordered
ABI:

1. `input_codes: *const u8` — shared row-major E4M3FN activation codes `[m, k]`.
2. `gate_weight: *const u8` — row-major E4M3FN gate weights `[n, k]`.
3. `up_weight: *const u8` — row-major E4M3FN up weights `[n, k]`.
4. `input_scales: *const f32` — one FP32 activation scale per row `[m]`.
5. `gate_scales: *const u16` — one BF16 gate scale per output channel `[n]`.
6. `up_scales: *const u16` — one BF16 up scale per output channel `[n]`.
7. `output: *mut u16` — final row-major BF16 SwiGLU values `[m, n]`.
8. `unrounded: *mut f32` — final FP32 products before BF16 rounding `[m, n]`.
9. `m: u32`, `n: u32`, `k: u32` — row, channel, and reduction dimensions.

Launch with `grid = [ceil(n / 4), m, 1]` and `block = [128, 1, 1]`. Each of the
four warps owns one output channel, and its 32 lanes divide K. Every lane reaches
both full-mask integer reductions. M/N channel tails use zero partials; K tails
stop at `index < k`, so they do not read padded storage. Gate and up consume the
same activation code and row scale, decoded/read once per lane and K iteration.

## Arithmetic profile and structural effect

The exact E4M3FN dot accumulates integer units where each code is an integer
multiple of `1/512`. Each projection converts the reduced integer dot to FP32,
then applies `2^-18`, activation row scale, and its own BF16 weight scale in that
order with rounded FP32 multiplication. Gate and up are separately rounded to
BF16 RNE. The existing `silu` implementation consumes decoded BF16 gate; its
result is rounded to BF16 before multiplying decoded BF16 up. The final product
is written as FP32 diagnostics and BF16 RNE output. This preserves the current
profile's rounding boundaries and does not adopt a Ninfer FP32 epilogue.

The kernel takes one activation-code/scale matrix for both projections, so the
host can quantize the activation once. At the source-structure level, the fused
entry can replace two exact projection launches and a separate SwiGLU launch
with one launch, while removing one duplicate activation quantization from the
current gate/up path. These are structural counts only. Saved time, bandwidth,
and end-to-end speed are **not measured**. No concatenated weight copy is needed.

## Independent reference and current qualification

`reference/fp8_swiglu_exact.rs` composes the independent logical FP8 projection
reference for gate/up and the independent BF16 SiLU/product reference. It takes
one already-quantized `QuantizedRows`, matching the shared input contract. Its
unit test compares the fused result to those logical APIs with signed values,
an exact cancellation, distinct per-row/per-projection scales, and odd
`M=3`, `N=5`, `K=7` dimensions. A second test checks invalid dimensions and
weight extents. These host tests are authored but have not been run by this
worker.

Parent qualification must register the new device and reference modules, add
this entry to `KNOWLEDGE/asm-inventory.md`, compile the pinned NVPTX target,
assemble and JIT on SM120a, and compare signed/cancellation/scale/tail cases to
both this independent composition and the existing exact component controls.
Run memory, race, and synchronization sanitizers before any resident dispatch.
For performance claims, compare matched fusion-on/off launches and model-path
impact under the same arithmetic profile. Source revision, target device,
toolchain, and evidence paths remain to be recorded by the parent integration.

Parent continuation: a standalone `feature-fusion-check` host probe is prepared
in `src/kernels/cuda/feature_fusion_trial.rs`. It compares seven signed/tail/
maximum-K/cancellation fixtures against both the independent FP64 logical
composition and separate exact GPU projections plus existing activation.
It requires exact projection controls, final BF16, and final raw FP32 for these
fixtures. Distinct gate/up scales are exercised. The device kernel is unchanged.
This probe has not run on Linux/GPU and has no performance claim. Further model
integration is held pending the requested deep runtime comparison with Ninfer;
this prepared qualification does not establish that fusion is the next bottleneck.
