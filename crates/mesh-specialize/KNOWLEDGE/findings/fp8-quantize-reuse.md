# FP8 activation quantization reuse

Status: implemented, not GPU-qualified. `MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE`
accepts `baseline` (default) or `reuse-input`. The quantizer implementation,
per-row scale formula, RNE conversion, and projection arithmetic are unchanged.
Local qualification has no GPU, so operator, whole-model, sanitizer, and timing
results remain pending.

The code-shape correction keeps the original projection entry unchanged when the
schedule is `baseline`, including its exact allocation and launch sequence. Only
the `reuse-input` branch passes caller-owned codes/scales into a projection. The
host planner covers stream grouping and lifetime; the legacy callers explicitly
create one owned quantization output per matching projection group.

## Design

`fp8_quantize_bf16` uses one 256-thread CTA per row. Each thread scans its
columns at stride 256, takes absolute values, and participates in a shared-memory
maximum tree. The per-row scale is FP32 `amax / 448`, with exactly `1.0` used
when the computed scale is zero. Codes use the existing software search over
finite E4M3FN with nearest-even tie handling. The independent reference rejects
non-finite BF16 inputs before device launch. No device arithmetic or source code
in `kernels/nvptx/fp8_quantize.rs` changed.

`reuse-input` removes duplicate quantizer launches by sharing the first
projection's codes and row scales with projections that consume the same BF16
input:

- Attention Q/K/V projections share normalized input.
- GDN QKV/Z projections share normalized input.
- FP8 MLP gate/up projections share their BF16 input.

Legacy resident execution owns one temporary quantized-input allocation for the
group. Stream, chunked, and graph execution direct each follower projection at
the first projection's existing arena code/scale slots. The arena liveness plan
extends those slots through the last follower and omits each duplicate code/
scale allocation. Different projection inputs remain separate. NVFP4 gate/up
continues to quantize separately because its per-projection input scaling
parameters differ. Standalone legacy, MLP-workspace, and prepared-workspace
projection paths do not deduplicate unless they enter the explicit legacy
projection group; workspace experimental profile combinations are rejected.

The mode is opt-in and process-fixed through `OnceLock`. It is supported by
legacy execution, stream execution, chunked prefill/decode, and graph capture.
No dynamic graph parameter behavior changes. `a16-decode` and MLP workspace
profiles reject `reuse-input` explicitly. The standalone quantization operator
check dispatches through `xtask specialize fp8-quantize-check` and compares
repeated baseline quantization, one shared quantization, and the independent
host oracle on decode widths and batch rows. Fixtures include all zeros,
all-equal values, BF16 denormals, positive/negative max BF16, mixed signs, and
the current NaN contract (host/reference rejects before launch).

## Expected savings

The supplied RTX 5090 single-token baseline has 233 quantizer launches at
2.60 ms aggregate per token. Q/K/V sharing saves two launches per full-attention
layer (16 layers); GDN QKV/Z sharing saves one per GDN layer (48 layers); the
eight FP8 MLP gate/up pairs save one per layer. In total this removes
`32 + 48 + 8 = 88` launches per token, or about 0.98 ms using the measured
2.60/233 ms average launch cost. This is an estimate, not a measurement. Stream
execution may trade some eliminated quantizer work for additional projection
reads from the shared code/scale slots; model timing must measure the net change.
The 168 NVFP4 quantizer launches are unaffected.

## Qualification

Rebuild current PTX and xtask, then run the operator check on Carrack:

```sh
just specialize-ptx
just specialize-tools-build
target/release/xtask specialize fp8-quantize-check --ptx target/specialize/probes.ptx --device 0 --output /tmp/fp8-quantize-check.json
```

Run the same check under each CUDA sanitizer with the FP8 quantizer name filter:

```sh
for tool in memcheck racecheck synccheck; do
  /opt/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 42 --kernel-name regex=fp8_quantize_bf16 target/release/xtask specialize fp8-quantize-check --ptx target/specialize/probes.ptx --device 0 --output "/tmp/fp8-quantize-${tool}.json"
done
```

The parent then qualifies output/state/logit/token equality against the unchanged
baseline for legacy and stream using `qwen-model-check` and
`qwen-stream-check`, each with `MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE=baseline`
and `reuse-input`. Check graph capture/replay with the existing
`MESH_SPECIALIZE_EXECUTION=graph qwen-stream-check` path, and run the chunked
benchmark in both modes. Profile matched whole-model decode only after exact
identity passes. Expected whole-model commands, where artifact and fixture paths
are supplied by the parent:

```sh
MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE=baseline target/release/xtask specialize qwen-stream-check --artifact ARTIFACT --tokens TOKENS --ptx target/specialize/probes.ptx --device 0 --output /tmp/fp8-reuse-baseline.json --decode-steps 2
MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE=reuse-input target/release/xtask specialize qwen-stream-check --artifact ARTIFACT --tokens TOKENS --ptx target/specialize/probes.ptx --device 0 --output /tmp/fp8-reuse-candidate.json --decode-steps 2
MESH_SPECIALIZE_EXECUTION=graph MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE=reuse-input target/release/xtask specialize qwen-stream-check --artifact ARTIFACT --tokens TOKENS --ptx target/specialize/probes.ptx --device 0 --output /tmp/fp8-reuse-graph.json --decode-steps 2
```

Model-level sanitizer scope is broader than the quantizer filter and needs a
separate deliberate full-model run. Do not treat the operator filter as whole
model sanitizer coverage.

## Source accounting

The activation-sharing seam is `resident_projection::Projection::{shared_input,
run_with_input}` in legacy execution and `Enqueue::{prepare_projection_input,
projection_with_input}` in stream execution. Stream planner behavior and its host
test live in `stream_forward/program.rs`. The new operator check is
`kernels/cuda/fp8_quantize_trial.rs`. Quantization assembly remains inventoried
under `kernels/nvptx/fp8_quantize.rs` in `KNOWLEDGE/asm-inventory.md`.
