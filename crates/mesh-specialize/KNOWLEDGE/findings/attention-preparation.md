# Full-attention Q/K preparation

Status: implementation criteria, before measurements. This advances Q02 with
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
