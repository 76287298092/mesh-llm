# Full-attention Q/K preparation

Status: real-weight preparation and all three CUDA sanitizers pass. This advances Q02 with
layer-3 Q/K/V projections, per-head Q/gate split, zero-centered Q/K RMSNorm and
partial text RoPE. Attention scores, softmax, KV cache, sigmoid output gate and
output projection remain separate tasks. Layers 0..2 are not run: embedding rows
serve as explicit synthetic hidden input to layer 3's input norm/projections.

Pinned checkpoint config/header: `KNOWLEDGE/evidence/checkpoint-intake-20260927/`.
Pinned Transformers revision `96331a9f93b72697f160a958d2883d4b49a56739`,
`src/transformers/models/qwen3_5/modeling_qwen3_5.py` (cached in the entry trial).
Layer 3 is full attention: Q/gate channels 12288, K/V channels 1024, input 5120,
24 Q heads, 4 KV heads, width 256. Q projection layout is [token,24,512], with Q
then gate in EACH head. Q/K norm weights have width 256 and zero-centered gamma.
Only the first 64 head channels rotate, paired across two halves of width 32.
All four attention projections use E4M3 channel weights with BF16 row scales.

The kernel receives explicit BF16 cosine/sine tables and projected BF16 inputs.
Its per-head norm uses FP32 squares, tree reduction and reciprocal square root,
then FP32 multiply by 1+BF16 weight, then BF16 rounding. Diagnostics preserve
unrounded FP32 and BF16 norm values. RoPE matches the eager BF16 operator
boundaries: both products round to BF16 before their sum rounds to BF16. The
remaining channels pass through; Q gate words are copied unchanged.

The CPU oracle uses f64 norm accumulation independently of the GPU reduction.
Before measurements, FP32 norm uses 1e-6 + 3e-6*abs(reference); norm BF16 must
exactly round the actual GPU diagnostic. Rotation must exactly match a scalar
rotation of actual GPU norm words. A separate full-operation comparison uses
the original inputs and CPU norm throughout, reporting BF16 differences and
applying the established 1% normalized-L2 / 0.9999 cosine budget per head and in
aggregate. Gate copy is bitwise exact. All nonfinite/invalid shapes fail.

Text RoPE has identical T/H/W position IDs, so the MRoPE interleaving is identity.
The initial reusable table builder computes theta^(2*i/dim) in f64 rounded to
FP32, takes FP32 reciprocal, then FP32 position multiplication; f64 sin/cos of
that FP32 angle round through FP32 to BF16. This explicit CPU table profile is
not a claim of exact CUDA/PyTorch trig parity. Hand anchors check zero/one and
rotation signs/pairs; positions through 262143 exercise coefficient generation
only, not usable model context. GPU kernels perform no trig approximation.

Workers own the independent reference, one device kernel and the checkpoint
loader. Parent owns the design, harness, wiring, validation, PTX inventory and
Carrack deployment. New kernel/launch paths require host tests/Clippy, PTX build,
normal real-weight trial and memcheck/racecheck/synccheck before qualification.

Local 149 tests, Clippy, formatting, no-console and PTX compilation pass. The
first Linux compile caught a parent integration mistake: the new wrappers used
a nonexistent `Context::allocate` method instead of the existing `Buffer::new`.
Both allocation helpers now use the established RAII API. Preserve the initial
failed Linux test/build log. No GPU trial ran at the failed revision.

## Carrack qualification

Source `295cb5e8c95155f476e61659920cae3c0f3a70d2` passes real-weight input
preparation on RTX5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`,
SM120, driver 615.71.09/API 13040. The host uses Rust 1.98.1/LLVM 22.1.8;
Rust device PTX uses nightly-2026-09-25 with rebuilt NVPTX core. CUDA 13.4.92
ptxas and the driver JIT report 32 registers and 1,024 bytes shared memory for
`attention_qk_prepare`; offline assembly has zero stack/spills and one barrier.

Release xtask SHA-256:
`4165b33adc569d9a2cc39894cb0b0cf63ed9bfc24ad8b8a7c5513b8f7cdaa66c`.
PTX SHA-256:
`817564d59ea188ab9af4cbc0fb5170dec6a750e91b9041650c5bd5c36d101b2f`.
Mac passes 149 library tests and Clippy; Linux passes 160 library and 17 validator
tests plus Clippy. Formatting, no-console and PTX compilation pass. Read-only
cross-checks cover kernel/reference layout and arithmetic and resident-buffer
integration. The initial Linux compile failure remains in evidence.

The one- and 17-token cases validate 258,048 real Q/gate, K and V projection
outputs. The largest FP32 projection error is 0.000057220458984375, within the
existing absolute-product-sum bound. There are 34 BF16 scalar-reference
projection differences (26 Q/gate, six K and two V in the 17-token case), with
exact rounding from each device FP32 diagnostic. These are reported component
differences, not hidden by replacing the resident projection input.

Q/K preparation checks 129,024 real values. Norm FP32 maximum absolute error is
0.000000476837158203125. One 17-token Q value differs from the independent BF16
norm/rotation reference; all other prepared values match. The worst per-head
normalized L2 is 0.0001990355183777659 and the minimum cosine is
0.9999999802228831, passing the fixed 0.01/0.9999 gates. Every rotary output
matches the scalar BF16 operator sequence from the actual GPU norm exactly.
All 110,592 Q gate words copy exactly; K gate buffers remain untouched.

Five additional fixtures check 10,566 prepared values, including width 8, real
width 256, strided tail width 258, width 1024, multiple heads, zero input, weight
-1 and positions 0/1/131071/131072/262143. Their prepared BF16 values match the
independent reference exactly. The reference unit tests separately prove
midpoint product rounding differs from single final rounding, quarter-turn
pair/sign layout, and the first two frequency anchors.

Normal, memcheck, racecheck and synccheck reports all pass. Memcheck and
synccheck report zero errors; racecheck reports zero hazards/errors/warnings.
Each run uses a user scope with 8 GiB RAM, zero swap and a 240-second timeout.
Elapsed times are 12.284807581, 12.297463947, 12.617530541 and 12.209262198 seconds,
respectively. These include artifact verification, uploads and CPU references;
they are not inference throughput. Driver free memory before/after temporary
allocations is 32,221,822,976 bytes, not a measured full-model peak. Model prefill,
decode and usable context fields remain null.

Ninfer restarted at 03:58:35 EDT on 2026-09-27, PID 3006534, and reported engine
ready at 03:58:40. Fresh health returns HTTP 200 and GPU sampling shows its
30,046 MiB allocation. ComfyUI PID 448118 remains at 498 MiB. No original model
assets, Ninfer source or service configuration changed.

Evidence: `KNOWLEDGE/evidence/qwen-attention-prepare-20260927/`. Raw files/PTX
remain under `target/specialize/qwen-attention-prepare-20260927/` on both hosts.
Reproduce by building with `just specialize-ptx` and `just specialize-tools-build`,
then running `target/release/xtask specialize qwen-attention-check --artifact
PATH --ptx PATH --device 0 --output NEW_FILE`. The schema is 1 and preserves
failed numerical reports before returning failure. Next: causal attention and
KV state/chunk equivalence, then gate/output projection and full attention-layer
qualification before the full model schedule and serving ABI.
