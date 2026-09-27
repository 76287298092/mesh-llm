# Causal attention and persistent BF16 KV state

Status: parent design and fixed criteria before measurements. This adds the
attention core after resident layer-3 Q/K preparation and V projection. The
output sigmoid gate, output projection, full attention layer and full model
remain pending. Layer-3 trial inputs remain explicitly synthetic embedding rows;
this does not imply execution of preceding layers.

The first core uses causal grouped-query attention with a persistent token-major
BF16 K/V cache. Query head h maps to KV head h/(Q_heads/KV_heads). Appending a
chunk copies only its K/V rows at the current prefix length. A query in chunk
row r attends exactly prefix+row+1 keys, even when later chunk rows have already
been appended. Capacity is distinct from initialized length; untouched capacity
uses NaN poison in the harness and must remain bitwise unchanged.

One CTA owns a query/head; each thread owns one value channel. A shared FP32
QK reduction supplies each causal score. Stable online max/denominator/value
accumulation avoids a quadratic global score allocation. Explicit rounded FP32
arithmetic prevents unintended FMA; exponentials use approximate exp2 and may
flush negligible underflows. Output divides accumulated values by the denominator
in FP32, then rounds to BF16. This is a fused-attention-style FP32 score/softmax
profile, distinct from the eager BF16 logits/probabilities path. The scalar
oracle computes logical dots, stable softmax and weighted values in f64.

Before measurements, each FP32 output must meet 5e-6 + 3e-5*max(abs(V)) over the
causally visible values for that output channel. BF16 must exactly round the GPU
FP32 diagnostic; independent scalar BF16 differences stay visible. Whole-sequence,
[1,remainder], [2,1,remainder] and token-by-token execution must produce identical
BF16/FP32 outputs and final K/V bits. Every intermediate append must preserve the
prior prefix and poison tail exactly and match an independent CPU cache update.
Hand fixtures cover GQA mapping, causal masking, stable large logits, signed values,
zero queries and unused NaN capacity. CPU validation rejects malformed extents,
nonfinite initialized inputs, invalid scales and out-of-capacity appends.

This is a correctness and memory-layout foundation, not the final throughput
kernel. It still processes keys sequentially inside each CTA. Tiled tensor-core
attention and launch/graph tuning remain performance requirements; no harness
wall time may be reported as model prefill or decode. BF16 cache here does not
claim parity with Ninfer's deployed FP8 KV profile or usable long-context memory.
The remaining full-model comparison must use explicit matched cache/speculation
profiles and measured peak memory/context.

Workers own one two-kernel device module and one independent CPU reference. The
parent owns resident-buffer integration, persistent/chunk harness, source review,
PTX inventory and bounded Carrack trials. New kernels require host tests/Clippy,
PTX compilation and all three CUDA sanitizer checks before qualification.

Partition tests reuse identical resident prepared Q/K and projected V arrays,
then partition the core's append/query calls. They qualify core/cache behavior;
they do not yet prove that the entire projection/decoder pipeline or sampled
logits agree between prefill and decode. The prepared outputs remain device
buffers: CPU reference outputs never replace them.

The first local run caught an incorrect hand fixture expectation: 1,000,000 is
not exactly BF16-representable, so its expected uniform mean used the wrong
input. The causal exclusion test now uses exactly representable +/-2^20 and
hand-computed means. The failed log is preserved; production arithmetic and
numerical budgets did not change.
