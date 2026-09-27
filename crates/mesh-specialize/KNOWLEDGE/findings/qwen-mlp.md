# Layer-zero NVFP4 MLP and final residual

Status: real-weight component chain and all three CUDA sanitizers pass. The full
model cannot serve or provide performance/context measurements yet.

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
`target/specialize/qwen-mlp-20260927/`. Qualification and its measurement limits are recorded below. All model performance/context fields remain null.

Initial local reference tests pass. Clippy identified an oversized validation
test's cognitive complexity; splitting dimension/scale checks from global/output
overflow checks resolves it without changing arithmetic. Its failed log is
retained. Static host-path review confirmed ABI argument order, scale direction,
lane layout and resident input/mirror coupling. A transient missing activation
reference noted during review was filled by the separate reference task before
host compilation.

## Carrack qualification

Source `000ae98a3a5b1fcfdae9107b526f19f33a237bc3` executes the full layer-zero
component chain on RTX5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`,
SM120, driver 615.71.09, driver API 13040. Host build uses Rust 1.98.1/LLVM
22.1.8. Device PTX uses pinned nightly-2026-09-25 with rebuilt NVPTX core.
CUDA 13.4.92 ptxas and the driver JIT report 48 registers for NVFP4 linear,
20 for MLP activation and 14 for residual addition. None uses shared/local
memory, spilling or barrier identifiers.

Release xtask SHA-256:
`d654084b64ede9c7f6cfaa5f5de7935794752216e2517d30fdd5e09a21edb6a0`.
PTX SHA-256:
`e75136f4874be3765180d3ab3c2dd1c6fd70cb8415a87f7a2fa8e77b5febcb8f`.
Mac passes 133 library tests; Linux passes 144 library and 17 validator tests.
Focused all-target/all-feature Clippy, formatting, no-console checks and PTX
compilation pass. No GitHub Actions result is claimed.

The [normal report](../evidence/qwen-mlp-20260927/normal.json) passes one and
17 real tokens, with 718,848 new matrix outputs: 313,344 each for gate and up,
and 92,160 for down. Gate/up FP32 results exactly match the scalar profile.
Down's maximum FP32 error is `4.9591064453125e-5`, with zero numerical or
rounding mismatches. Every new matrix BF16 output matches the independent
component scalar reference exactly for these inputs. The extra M1/N1/K16 and
M17/N13/K80 matrix fixtures pass all 222 outputs exactly.

The actual input global multipliers are 836 for gate/up and 161 for down.
Weight global multipliers are 6400 for gate/up and 2752 for down. All 497,664
activation values across these three quantizations have exactly matching packed
codes, local scales and effective FP32 scale bits. This is 31,104 scale groups;
gate/up quantize the same resident normalized input separately. The global factor
is reciprocal, as required by the retained checkpoint format.

Across 313,344 real SiLU/product values, maximum SiLU FP32 error is
`4.76837158203125e-7`. Twenty-one BF16 activation values, and their resulting
BF16 products, differ from the f64 reference. Maximum full-reference product
FP32 difference is `2.0563602447509766e-6`. All declared SiLU phase bounds,
intermediate BF16 rounding, exact multiplication from that rounded activation
and final BF16 rounding checks pass. Another 771 activation fixture values pass,
including tails, negative zero, extreme gates and subnormal negative SiLU.

The final residual addition passes all 92,160 real and 258 fixture BF16 values
exactly against the scalar sum of its actual two inputs. These are component
checks: the reference for each operation receives the preceding device output.
No independent whole-layer comparison has yet been performed. In particular,
matching down-projection BF16 outputs does not prove that the 21 earlier SiLU
BF16 differences disappear in an independently executed full MLP or model.
The report explicitly keeps `full_layer_reference_compared` false.

Normal command duration is 40.147 seconds including artifact checks, uploads and
CPU oracles. This is not model/kernel throughput. Driver free memory before and
after temporary allocations is 32,221,822,976 bytes, not an allocator peak for a
full model. Clocks/power are only sampled before the trial in `before.txt`.

Reproduction, after the two Just build recipes:

```sh
./target/release/xtask specialize qwen-projection-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/qwen-mlp-20260927/probes.ptx \
  --device 0 --output NEW_REPORT.json
```

The parent runs each command in a systemd scope with 8 GiB host-memory limit,
zero swap and a 240-second timeout, stopping Ninfer under an EXIT restart trap.
Sanitizer runs add `/opt/cuda/bin/compute-sanitizer --tool TOOL --error-exitcode 42`
and use distinct output paths. Raw logs/PTX/cubin remain in the target trial
folder on both hosts. The permanent evidence directory retains reports, test/
build summaries, the initial failed Clippy run and service records.

[Memcheck](../evidence/qwen-mlp-20260927/memcheck.log),
[racecheck](../evidence/qwen-mlp-20260927/racecheck.log) and
[synccheck](../evidence/qwen-mlp-20260927/synccheck.log) all pass with zero errors
or hazards. Each report passes all new and earlier component/state checks.
Ninfer restarted at 03:27:52 EDT on September 27 as PID 2946597, reached
engine-ready at 03:27:58 and returned HTTP 200 from `/health`. Its sampled
allocation is 30,046 MiB. ComfyUI PID 448118 remains at 498 MiB.

The resident layer-zero component chain now reaches the final decoder residual.
Independent whole-layer/state/logit comparisons, full-attention layers, generic
execution/session scheduling, tokenizer/sampling, ABI integration and the
requested full-model prefill/decode/memory/context comparison remain open.
The 40-second qualification command includes independent CPU oracles and cannot
be used as a runtime speed estimate.
