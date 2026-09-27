# GDN gated normalization and output projection

Status: real-weight component comparisons and all three CUDA sanitizers pass.
This connects the remaining layer-zero GDN attention operations. Model execution,
independent full-layer/logit parity and performance comparisons remain open.

The pinned [Transformers gated norm](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L218)
normalizes each 128-value head in FP32, rounds normalized values to BF16,
multiplies the direct BF16 norm weight, rounds that product to BF16, then gates
with FP32 SiLU of Z and rounds the final result to BF16. Gamma is not
zero-centered. One shared BF16 weight vector applies to all groups. Epsilon is
1e-6 in the pinned checkpoint. These rounding points differ from a fused path
that keeps every intermediate in FP32; this experiment preserves the fallback
semantics explicitly.

A Rust kernel uses one 256-thread block per head/token group, shared FP32 square
reduction, explicit rounded add/multiply/divide/square-root, and non-FTZ
`ex2.approx.f32` for stable SiLU. It exposes normalized FP32, weighted BF16,
SiLU FP32 and final FP32 values for independent phase checks. The scalar oracle
uses f64 norm accumulation and f64 SiLU. FP32 normalization must satisfy
`2e-6 + 2e-6 * abs(reference)`; SiLU must satisfy
`3e-6 + 5e-6 * abs(reference)`. Intermediate weighted BF16 and final BF16
rounding must exactly match their GPU FP32 precursors. Full scalar BF16
differences are reported separately, not hidden by those component bounds.

The host retains the whole-sequence recurrent BF16 output and the original
resident FP8 Z projection output. Neither is replaced by a CPU oracle result.
After gated norm, the existing qualified FP8 activation quantizer produces
per-token E4M3FN codes and FP32 scales. The output projection uses pinned
E4M3 weights shaped `[5120,6144]` with BF16 per-channel scales. Every activation
code and scale is independently compared, and the existing f64 dot-product
oracle checks output projection arithmetic and BF16 rounding.

One and 17 token sequences exercise real layer-zero weights. Dedicated norm
fixtures cover shared signed gamma, zero rows, signed and extreme gates, and
widths one, eight and 256. The recurrent kernel/state and all earlier projection,
convolution and preparation checks remain in the command. Whole/chunk state
qualification concerns recurrence; the entire layer is not yet scheduled as a
stateful serving session or compared against an independent full-layer engine.

Two bounded Luna-max workers own the device kernel and scalar reference. The
parent owns the precision contract, retained device inputs, artifact parameters,
output projection reuse, comparisons and deployment. Source review corrected
the initial reference's gamma extent/indexing to a shared width-length vector
and corrected a negative SiLU underflow test. The reference also rejects an
FP32-overflowing square sum before division by width. No serving or public ABI
change is included.

Reproduction extends `qwen-projection-check` to schema 6, using fresh evidence
under `target/specialize/qwen-gdn-output-20260927/`. Build PTX through
`just specialize-ptx` and the Linux host through `just specialize-tools-build`.
Model throughput/context fields remain null and `model_executable` remains false.

Local validation passes 112 library tests, focused all-target/all-feature Clippy
with warnings denied, the repository no-console check, formatting and Rust PTX
compilation. The initial test run caught an incorrect unsigned-zero expectation
for negative gamma. Its fixture now checks the signed BF16 zero produced by the
specified arithmetic; the kernel/reference arithmetic did not change. The failed
log is retained under the trial directory.

Linux passes 121 library tests, 17 validator tests and focused all-target/all-feature
Clippy with warnings denied. No GitHub Actions result is claimed. The real trial
and sanitizer evidence follow below.

## Carrack qualification

Source `73bc627efaf90cff97d8f499cccf218baad14d35` passes on RTX 5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09 and
driver API 13040. Release xtask SHA-256 is
`5758d1b0ee252e2290b296fe6f1698ba0af4a312758ff001f2ef59fe3dfe33c0`;
PTX SHA-256 is `87d25d625bcb45df3ce529a2847ef407f55ea9149cab1b74c02abec622a177f6`.
The host used Rust 1.98.1 and LLVM 22.1.8. Device PTX was built on the Mac
with pinned nightly-2026-09-25 and rebuilt NVPTX core. CUDA 13.4.92 ptxas and
the driver JIT both report 23 registers and 1,024 shared bytes for gated norm.
There is no local memory or spilling, and the kernel uses one barrier identifier.

The [normal report](../evidence/qwen-gdn-output-20260927/normal.json) passes
110,592 real gated-norm values across one and 17 tokens. Maximum normalized
FP32 error is `1.9073486328125e-6`; maximum SiLU error is
`9.5367431640625e-7`. Normalized BF16 and gamma-weighted BF16 both match the
scalar oracle exactly for these inputs. The final gated BF16 comparison has
one difference; maximum full-reference FP32 difference is
`4.76837158203125e-7`. Every intermediate/final device rounding check passes.
The 827 dedicated fixture values also pass, with no final BF16 differences.

The resident gated output quantizes to exactly the expected FP8 codes and token
scales. The resulting `[1,5120,6144]` and `[17,5120,6144]` projections pass
92,160 outputs with maximum FP32 error `5.53131103515625e-5`. The existing
absolute-product-sum tolerance accepts every value, and every BF16 output
exactly rounds its GPU FP32 precursor. There are 29 BF16 differences from the
independent f64 projection reference. No exact scalar-output parity is claimed.

These are component arithmetic gates. The scalar final-error and BF16-difference
counts are diagnostics; the declared norm/SiLU phase bounds, projection bound,
quantization equality and explicit rounding checks decide pass/fail. The report
is explicit that an independent complete-layer/model-logit comparison is still
missing. A source-only review found no ABI, retained-buffer, extent, shared-gamma,
head grouping or output-projection shape mismatch and confirmed this distinction.

[Memcheck](../evidence/qwen-gdn-output-20260927/memcheck.log),
[racecheck](../evidence/qwen-gdn-output-20260927/racecheck.log) and
[synccheck](../evidence/qwen-gdn-output-20260927/synccheck.log) each report zero
errors or hazards. Every component and earlier recurrent/state check passes in
all three runs. Each command is bounded by 8 GiB host memory, zero swap and a
240-second timeout. Normal execution takes 12.903 seconds including artifact
verification, uploads and CPU oracles. That is not kernel or model throughput.
Driver free memory before and after temporary allocations is 32,221,822,976 bytes,
not a full-model allocation peak. Model throughput/context fields remain null.

The qualified normal invocation, after the two Just build recipes, is:

```sh
./target/release/xtask specialize qwen-projection-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/qwen-gdn-output-20260927/probes.ptx \
  --device 0 --output NEW_REPORT.json
```

The parent runs this inside the bounded systemd scope with Ninfer stopped and a
restart trap. Sanitizer runs prefix it with
`/opt/cuda/bin/compute-sanitizer --tool TOOL --error-exitcode 42` and a fresh
report path. Raw build logs and PTX remain in the target trial directory on both
hosts; the [evidence directory](../evidence/qwen-gdn-output-20260927/) preserves
reports, test summaries, the initial failed fixture test and service records.

Ninfer restarted at 02:48:35 EDT on September 27 as PID 2878542, reached
engine-ready at 02:48:41 and returned HTTP 200 from `/health`. Its sampled
allocation is 30,046 MiB. ComfyUI PID 448118 remained resident at 498 MiB.

The layer-zero GDN attention component chain is now connected from embedding
through output projection. Residual/post-attention normalization and MLPs,
full-attention layers, whole-model scheduling, stateful serving, tokenizer and
sampling, ABI integration, full-model correctness and the requested performance/
context comparison remain open.
