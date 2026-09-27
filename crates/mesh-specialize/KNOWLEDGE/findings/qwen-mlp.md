# Layer-zero NVFP4 MLP and final residual

Status: implementation pending host/GPU qualification. The full model cannot
serve or provide performance/context measurements yet.

This extends the existing resident layer-zero chain with the MLP:
`gate_proj -> SiLU -> multiply up_proj -> down_proj -> second residual`.
The pinned checkpoint has hidden width 5120, intermediate width 17408 and SiLU
activation. Layer-zero gate/up/down weights use logical low-nibble-first E2M1,
per-16 E4M3 unsigned local scales and independent F32 input/weight global
multipliers. The verified artifact reader supplies all tensors. No NInfer source
or artifact format is consumed.

The direct Rust NVFP4 matrix kernel assembles registers from logical checkpoint
bytes, avoiding a second tile-packed weight allocation. Its A/B/scale fragment
coordinates are the already-qualified m16n8k64 NVFP4 instruction mapping. One
warp computes a 16-by-8 output tile, preserving FP32 accumulation across K tiles.
Partial M/N tiles and K padded from a multiple of 16 to 64 are guarded, with zero
payload and neutral scales outside the logical shape. The global output factor
is explicitly FP32 `1 / (input_global * weight_global)`, followed by rounded FP32
multiplication and BF16 output rounding.

The scalar reference decodes logical nibbles and per-group scales and sums in
f64. It has no CUDA fragment mapping. The declared output profile rounds the
raw f64 sum to FP32, multiplies the FP32 global factor and rounds to BF16.
Comparison uses the existing `1e-6 + 2e-6 * scaled sum(abs(products))` bound;
BF16 must exactly round the device's diagnostic FP32 output. Scalar BF16
mismatches remain separately visible. Hand-computed tests cover signed nibble
order, nonuniform group scales, reciprocal global direction, zeros and tails.

The MLP activation kernel evaluates stable non-FTZ FP32 SiLU, rounds it to BF16,
multiplies the decoded BF16 up projection in FP32, then rounds the product to
BF16. This preserves the default BF16 activation boundary in the pinned
[Transformers MLP](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L835).
The independent reference uses f64 exponential. SiLU is bounded by
`3e-6 + 5e-6 * abs(reference)`; its BF16 rounding, multiplication from that
rounded activation and final rounding are checked explicitly. Full-reference
BF16 differences remain diagnostic and do not silently become exact parity.

The second residual addition uses the saved BF16 sum from the post-attention
boundary plus the actual resident down-projection output. Its logical CPU
reference requires exact BF16 results. All device intermediate buffers survive
until their consumers finish. CPU oracles never replace resident inputs.

The parent designs interfaces/precision, validates ownership and integrates the
artifact loader/harness. Bounded Luna-max tasks supply the matrix device kernel,
independent matrix reference, activation kernel and activation reference. The
parent supplies the small final residual kernel/reference. No serving/ABI
integration or full-layer reference comparison is implied by these component
checks. The complete component chain still needs an independent whole-layer
and eventual model-logit comparison.

Build with `just specialize-ptx` and `just specialize-tools-build`. The existing
`qwen-projection-check` command becomes schema 8, with the real one/17-token
cases, signed M17/N13/K80 projection fixture, activation/residual tail fixtures
and earlier chain/state checks. Fresh logs belong under
`target/specialize/qwen-mlp-20260927/`. GPU timings, clocks, memory and sanitizer
results are not yet measured for this change; qualification will be recorded
below. All model performance/context fields remain null.

Initial local reference tests pass. Clippy identified an oversized validation
test's cognitive complexity; splitting dimension/scale checks from global/output
overflow checks resolves it without changing arithmetic. Its failed log is
retained. Static host-path review confirmed ABI argument order, scale direction,
lane layout and resident input/mirror coupling. A transient missing activation
reference noted during review was filled by the separate reference task before
host compilation.
