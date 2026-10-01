# Native MTP packed views

Status: host metadata views and proposal token-ID validation implemented. CPU
Q8/Q4 numeric row decode, indexed Q4 proposal-head step, and a bounded synthetic
Q4 GPU operator candidate are present. Host-only physical-parent loading and
hash qualification now pass on the pinned artifact. Native NInfer model admission
remains unchanged; there is still no complete native MTP block or GPU/model
qualification for the Q4 operator candidate.

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

### Token-map streaming append fix

The parent reported `native proposal token map byte length mismatch` during the
real pinned parent check. The complete failed report is preserved at
[`failed-a.json`](../evidence/native-mtp-parents-20260930/failed-a.json), copied
from the supplied local report without rerunning qualification. It records
`all_passed: false` and 44.69598951 elapsed seconds. Source inspection confirmed that
`NinferArtifact::read_object_range` streams through `Write::write_all`, so a
`Vec<u8>` destination appends rather than overwriting existing elements.
`NativeMtpViews::resolve` reserved 524288 bytes, then incorrectly resized the
vector to that length before reading another 524288 bytes. The resulting
1048576-byte map failed the unchanged strict parser length check.

The fix retains the fallible reservation but leaves the vector empty for the
read. The regression
`resolve_reads_exact_token_map_when_artifact_has_complete_native_mtp_metadata`
uses the existing complete synthetic MTP directory in a temporary sparse v3
artifact, with compact aligned object offsets and a real 524288-byte map payload.
It calls the real reader through `resolve` and checks all 131072 distinct,
nonzero little-endian token IDs and the out-of-bounds lookup. Restoring the
resize makes resolution fail the parser's byte-length check before assertions.
Metadata, range, signed-ID checks, thresholds and admission are unchanged.

### Parent-run host-only qualification, 2026-09-30

The complete successful parent report is preserved at
[`passed-b.json`](../evidence/native-mtp-parents-20260930/passed-b.json). Both
reports were read in full from the supplied `native-mtp-parents-20260930-a.json`
and `native-mtp-parents-20260930-b.json` files in the local temporary directory.
The qualification used baseline `b188c2925` plus an uncommitted scoped snapshot.
That baseline is not a commit identity for the tested changes; neither JSON
report records a source-code revision.

The successful report records `all_passed: true`, 14 physical parents and
808307712 physical-parent bytes. Every parent's `copied_bytes` equals its
`expected_bytes`; per-parent SHA-256 values are retained verbatim in the report.
Summing those recorded fields gives 451267584 bytes for the 12 MTP parents and
357040128 bytes for the two proposal parents. There are eight logical Q8 views,
seven logical norm views, and 131072 proposal rows. The Q4 head is 356515840
bytes; the INT32 map is exactly 524288 bytes after the append fix.

The source artifact is `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`,
23719715844 bytes, with artifact ID `19c9ec11085642f1bbe340cd7cf6c207` and
SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
Source verification records `source_dirty: false`, two whole-file hash passes,
1589 canonical tensors and 20375588160 canonical tensor bytes. Total elapsed
time is 54.179 seconds, with the exact recorded value 54.179423618 seconds;
startup verification accounts for 32.123175529 seconds. This is host qualification
wall time, not GPU operator timing or model prefill/decode performance.

The parent reports that native Linux validation passed 575 unit tests and 26
integration tests, and original-worktree validation passed 432 host tests and
26 Python tests. These are parent-run results supplied with this handoff, not
worker-run tests or fields in the qualification JSON. No test transcripts were
supplied or fabricated. This worker only read evidence and source and preserved
the reports; it ran no Cargo, Git, SSH or GPU operations. Parent owns integration.

The passing result is explicitly host-only: `host_only: true`,
`gpu_execution: false`, `dense_dequantization: false`, and
`text_tensors_loaded: false`. `native_mtp_admitted`, `model_executable`, and
`full_ninfer_arithmetic_parity` remain false. Physical-parent copying and hashes
do not prove resident GPU execution, a complete MTP block, model correctness,
or completed MTP integration. Metadata checks, strict parser thresholds, and
admission remain unchanged. The synthetic Q4 candidate findings below are
preserved and gain no GPU/model qualification from this host-only result.

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

### Source-faithful Q4 indexed-head port map, audit only

Pinned source is Ninfer `e31bc99b` at the reference tree named in this entry's
layout citations. The production Q4 GEMV implementation is
`src/ops/linear/q4/q4_a16_gemv.cuh::q4_a16_gemv_kernel`, selected through
`src/ops/linear/q4/q4_dispatch.cpp::q4_dispatch/select_q4_a16_launch`, shape
table `shapes/n131072_k5120.cu::select_q4_n131072_k5120`, then wrapped by
`q4_a16_gemv.cu::launch_q4_a16_gemv_r4_w1_direct` and
`q4_instance_launch.cuh::launch_q4_a16_gemv_instance`. `text.cpp::project`
calls `ops::linear`; `TextContext::proposal_argmax` allocates BF16
`[proposal_head_n_, T]`, projects, invokes `ops::argmax`, and, when indexed,
calls `ops::proposal_remap_token_ids` (`text.cpp:564-583`). `load/text.cpp`
binds the indexed `[rows, hidden]` proposal head and I32 token map and checks
the map is unique and in the public domain (`load/text.cpp:119-141`).

For this head, the production shape is `[131072,5120]`. At `T=1`, shape
dispatch chooses `GemvR4W1` (`q4_instances.cuh:5-8`): four output rows per
CTA, one warp per row, 128 threads/CTA, direct BF16 activation reads, 16
Q4 groups per warp tile, one stage, async 16-byte vector code copies, paired
32-bit scale copies, and `PackedWord8`/`Fp16Mantissa` decoding. The kernel
assigns CTA warp `threadIdx.x >> 5` to one of four rows and lane to eight
codes per packed word (`q4_a16_gemv.cuh:314-318,320-337`). Each lane loads
one 32-bit word containing eight nibbles for one 64-value group, decodes
through `Q4SimtDecodeAtom::decode_eight`, reads eight BF16 activations as
four `uint4` values, and performs eight ordered `fmaf`s
(`q4_a16_gemv.cuh:67-99`). The XOR/magic FP16-mantissa decode is in
`q4_rowsplit_storage.cuh::Q4SimtDecodeAtom::decode_eight:18-32`. It computes
the same signed values as `(n ^ 8) - 8`; the Rust candidate uses an independent
FP16-mantissa PTX decode. Warp sums use `warp_reduce_sum`,
which adds in shuffle-down offsets 16,8,4,2,1 (`ops/common/warp.cuh:20-28`).
With one warp per row there is no inter-warp row reduction. Output uses
`linear_finish_row` then `__float2bfloat16_rn` (`linear/common/output.cuh:20-22`).

The indexed row map and final token remap are not fused into Q4 GEMV or an
indexed-specialized kernel. GEMV outputs one BF16 logit per local head row;
generic `ops::argmax` uses 512 threads for one-column direct reduction and
chooses the lower row ID for equal values (`ops/kernel/argmax.cuh:22-24,57-93`,
`ops/launcher/argmax.cu:40-58`). `proposal_remap_token_ids` applies the I32
local-row-to-public-ID map (`text.cpp:581-583`, `speculative_round.cuh:702-708`).
`linear_topk.h` also exposes a separate Q4 indexed Top-16 API, but MTP's
`proposal_argmax` path uses BF16 projection, argmax, then remap, not that API.

License provenance available in this reference tree is limited: the tree has
`tools/chat_templates/LICENSE`, but the inspected Q4/model/operator files did
not show a file-specific SPDX/license header. Do not copy this implementation
verbatim or assume a license for these sources; port the described schedule
independently and have the parent resolve provenance before any direct code reuse.

The current `kernels/nvptx/native_mtp_q4_head_gemv` candidate now follows the
selected `GemvR4W1` schedule: four rows per 128-thread CTA, one warp per row,
eight codes per packed-word lane, 16-group tiles, 16-byte code and 4-byte scale
`cp.async.ca` copies, FP16-mantissa decode, ordered FP32 FMA, five-step warp
reduction, and BF16 RNE output. Its scalar BF16 activation fallback and zero
physical Q4 padding preserve bounded logical-K tails. The separate FP64
projection oracle remains independent; `native_mtp_q4_operator/schedule_reference.rs`
also models the exact FP32 lane and reduction order. Host fixtures include row,
group, and K tails, but the CUDA schedule tests have not run and no GPU/JIT or
sanitizer qualification is established. This is a source-faithful schedule
candidate, not performance or model-parity evidence; parent GPU qualification
remains required before promotion.

Remaining mesh-specialize work: qualify the device PTX, execute the
synthetic operator check and GPU sanitizers, connect the host-qualified 12 MTP
physical parents, shortlist head, and token map to resident GPU execution, then compare a
complete target-conditioned MTP block against Ninfer fixtures. Only after those
steps should model-source MTP admission change. Never materialize a dense proposal
vocabulary matrix.

- Token-map values are checked for target-vocabulary bounds, but uniqueness,
  shortlist ordering, and whether duplicate target IDs are legal are not
  established here.
- `NativeMtpViews::resolve` still validates metadata/token IDs only. Model-source
  admission and resident native MTP execution remain explicitly outside this change.
