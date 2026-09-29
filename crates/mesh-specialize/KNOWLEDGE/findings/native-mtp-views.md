# Native MTP packed views

Status: host metadata views and proposal token-ID validation implemented. CPU
Q8/Q4 numeric row decode and the indexed Q4 proposal-head step are now executable.
Native NInfer model admission remains unchanged; there is still no complete native
MTP block, artifact-weight loader, or GPU packed consumer in mesh-specialize.

## Verified layout facts

Codec and MTP source citations below refer to Ninfer commit `e31bc99b` in the
read-only reference tree `/var/folders/5q/y9dmlwq11tqd74j_17t5p5ym0000gn/T/opencode/ninfer-ref-e31bc99b/`.

- The independent v3 schema represents tensor shape, format, layout, payload
  offset and byte count separately. Binding part ranges count logical elements,
  not packed bytes: `src/artifact/ninfer_schema.rs:26-39,116-131`.
- `row_split_k128_v1` uses `K_pad = align_up(K,128)`; the base plane precedes
  256-byte alignment and the FP16 scale plane follows. Row/group traversal is
  `row * (K_pad/G) + group`; the even Q4 code is low nibble and odd code high
  nibble; Q8 stores one signed byte/lane. Ninfer spec citations:
  `docs/maintainer/storage-layouts.md:93-107,110-155,177-217,223-249`.
- Q4 words are b-bit two's-complement (`q=u` if `u<2^(b-1)`, else `q=u-2^b`);
  Q8 legal values are `[-127,127]` and `0x80` is invalid. Ninfer numeric citations:
  `docs/maintainer/tensor-formats.md:357-395`.
- Scales are positive zero or finite positive FP16 normal/subnormal; negative
  zero/values and infinities/NaNs are invalid. A positive-zero scale requires
  zero codes. `0x0001 = 2^-24` is valid. Decode is
  `g=floor(k/G); s32=exact_binary16_to_binary32(s); w_hat=binary32(s32*binary32(q))`.
  Citations: `docs/maintainer/tensor-formats.md:397-441`.
- Row-split padding has zero code for the final partial group and later complete
  groups have zero code and scale `0x0000`: `docs/maintainer/storage-layouts.md:102-108`.
- The schema supports contiguous little-endian INT32 as four bytes per element:
  `src/artifact/ninfer_schema.rs:287-288`. The artifact evidence declares the
  proposal map as `int32`, `contiguous_le_v1`, shape `[131072]`, 524288 bytes;
  proposal metadata declares indexed rows 131072, distinct from target vocab
  248320: `KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-artifact-inspect.json:6-17,110-113,23970-23975`.
- The inspected bindings select the Q8 MTP parents and proposal tensors.
  Q/K/gate/V ranges use Q rows `[0,6144)`, K `[6144,7168)`, gate
  `[7168,13312)`, V `[13312,14336)` from one `[14336,5120]` parent. The source
  rows also record the MLP gate/up split at 17408. Evidence:
  `KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-selected-bindings.json:96-194,249-298,313-339`.
- Existing target-text mapping interleaves 256 Q rows and 256 gate rows per
  head and applies the same selected row order to code and scale rows:
  `src/packages/qwen3_8_27b/native_views/mapping.rs:180-208` and
  `KNOWLEDGE/findings/ninfer-import-contract.md:80-85`. The new MTP view uses
  that row mapping while retaining parent plane offsets and bytes.
- Source inventory declares 12 MTP objects / 451267584 bytes and two proposal
  objects / 357040128 bytes in `HANDOFF_NINFER_DIRECT_DECODE.md:225-231`.
  The source artifact object records and selected binding evidence independently
  confirm the formats, shapes, parent IDs and sizes; see object records at
  `KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-artifact-inspect.json:8620-8730`
  and proposal records at `:12547-12567` (proposal-head and token-map objects).
- `KNOWLEDGE/findings/ninfer-op-dispatch-e31bc99.md:51-53` records Q8 MTP
  signed INT8 codes with FP16 group scales and Q4 shortlist shape/format as
  source investigation. It labels the dispatch/recipe association as inference,
  not a full quantization codec definition.

## Numeric reference

The test-only reference at `reference/native_mtp_quantized.rs` decodes Q8/Q4
rows from their original code/scale planes and selected-parent row maps. It
requires exact row-split plane sizes and offsets, zero 256-byte alignment padding,
zero physical tail codes/scales, rejects Q8 `0x80`, invalid FP16 scale words, and
nonzero codes under zero scales. Binary16 scales expand directly from their
IEEE bits, including `0x0001`; no half-precision dependency or intermediate
conversion is involved.

The test-only Q4 proposal-head step consumes a `[K]` BF16 hidden state, dequantizes
each selected shortlist row to FP32, accumulates the row dot product with FP32
fused multiply-adds in increasing K order, rounds each logit to BF16 with
round-to-nearest-even, selects the greatest BF16 logit (first row wins a tie),
then maps that shortlist row through the validated signed INT32 token map. The
linear contract defines the FP32 dequantized dot product and BF16 output
(`include/ninfer/ops/linear.h:52-68`); the Q4 GEMV uses FP32 accumulation
(`src/ops/linear/q4/q4_a16_gemv.cuh:58-101`) and its output helper uses BF16 RNE
(`src/ops/linear/common/output.cuh:12-22`). The argmax tie rule selects the
lower row (`src/ops/kernel/argmax.cuh:22-25`). The separate `linear_topk` API
returns FP32 score/ID pairs (`include/ninfer/ops/linear_topk.h:24-63`); Qwen MTP
uses BF16 indexed-head logits followed by argmax/remap
(`src/models/qwen3_5/execution/text.cpp:564-590`).

Hand-built spec-derived tests cover Q4 nibble ordering and signed endpoints, Q8
signed byte endpoints and invalid `0x80`, scales `0x0001` and positive zero,
invalid scale classes, group transitions at K32/64/128, full rows crossing the
K128 split, logical-tail padding, selected parent-row addressing across aligned
planes, nonzero alignment-padding rejection and Q4 BF16-logit/remap.
Ninfer Python `tools/artifact/codecs/row_split.py:30-38,65-79,82-117,429-472`
implements encoding and dequantization; it was inspected, but runtime round-trip
was unavailable because Python Torch is not installed here. Rust expectations are
literal values derived from the published format rules, independent of the decoder.

## Ninfer MTP execution and remaining work

The production request is `--spec mtp --draft-tokens 4 --lm-head-draft` (startup
range permits 1..5: `docs/serving.md:778-779`). One MTP step consumes the next
token embedding plus target final hidden: RMSNorm embedding/hidden independently,
concatenate embedding then hidden, Q8 FC `[5120,10240]`, input RMSNorm; Q/K/gate/V
Q8 projection, Q/K per-head RMSNorm and RoPE, causal GQA attention over MTP KV,
sigmoid gate, Q8 output projection and residual, post-attention RMSNorm, gate/up
Q8 projections, SwiGLU, down Q8 projection and residual, final RMSNorm. These
steps and Q/K/V/gate shapes/order are in
`src/models/qwen3_5/execution/text.cpp:287-323,325-414` and
`src/ops/kernel/mtp_pack.cuh:13-54`; the high-level formulation and recursive
draft hidden behavior are in `docs/maintainer/qwen3_5-model.md:223-255`.

The proposal head is Q4 `[131072,5120]` plus an indexed shortlist map; logits are
projected, local argmax selected and row IDs remapped to target vocabulary, while
verification uses the full target head. `tools/convert/proposal.py:20-95`
selects shortlist IDs and stores the head plus INT32 row map. The separate
Q4 `linear_topk` API returns stable top-16 scores/IDs mapped to global IDs
(`include/ninfer/ops/linear_topk.h:24-63`); MTP itself uses argmax/remap
(`src/models/qwen3_5/execution/text.cpp:564-590`). Head loading is in
`src/models/qwen3_5/load.cpp:64-86`.

For four draft tokens, the first MTP forward produces draft 0 and recursive AR
steps produce drafts 1..3 (`src/models/qwen3_5/program/speculative/mtp.cpp:13-68`).
The decode round verifies target tokens for the anchor plus drafts, accepts the
longest prefix whose greedy target argmax matches, commits the divergence token,
then re-aligns and drafts the next round (`src/models/qwen3_5/program/speculative/mtp.cpp:70-193`,
`src/ops/kernel/speculative_round.cuh:216-255`). Target verification selects and
stores the continuation hidden state (`src/models/qwen3_5/program/speculative/target_verification.cpp:8-45`).

Remaining mesh-specialize work: load MTP's 12 packed matrices, BF16 norms, shortlist
head, and token map from Ninfer object ranges; compose the complete target-conditioned
MTP block with real weights and alignments; compare against Ninfer fixtures; then
add a GPU implementation consuming the original packed planes. Only after those
steps should model-source MTP admission change. Next GPU step is a row-split-aware
Q8 GEMV/linear consumer and Q4 shortlist-head top-k kernel preserving 256-byte
plane alignment, parent row maps, FP16 scale handling and FP32 accumulation; never
materialize a dense proposal vocabulary matrix.

- Token-map values are checked for target-vocabulary bounds, but uniqueness,
  shortlist ordering, and whether duplicate target IDs are legal are not
  established here.
- `NativeMtpViews::resolve` still validates metadata/token IDs only. Model-source
  admission and resident native MTP execution remain explicitly outside this change.
