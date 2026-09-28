# Native MTP packed views

Status: host metadata views and proposal token-ID validation implemented. Native
MTP execution stays rejected. No GPU code or numeric Q4 packed-weight decoder was
added because the local contract does not establish Q4 code interpretation. The
CPU reference rejects numeric decoding instead of applying a guessed equation.

## Verified layout facts

- The independent v3 schema represents tensor shape, format, layout, payload
  offset and byte count separately. Binding part ranges count logical elements,
  not packed bytes: `src/artifact/ninfer_schema.rs:26-39,116-131`.
- `row_split_k128_v1` accepts Q4_g64_FP16 and Q8_g32_FP16 as distinct formats.
  Its size formula pads K to 128, computes code bytes, aligns planes to 256,
   then accounts for FP16 scales (one 16-bit scale per padded K128 block element
   group): `src/artifact/ninfer_schema.rs:301-305,317-332`.
  The format summary says the representation pads K to 128 and separates code,
  high-bit and scale planes: `KNOWLEDGE/findings/ninfer-model-format.md:68-79`.
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

## Implemented checks

`src/packages/qwen3_8_27b/native_mtp_views.rs` exposes separate packed Q8/Q4
views, parent row maps, byte-plane offsets and unchanged scale-byte extents.
Selection checks exact formats, layouts, shapes, byte counts, binding ranges,
row-aligned parent offsets, complete MTP/proposal object totals, and preserves
the ordinary 248320-row FP8 target output head separately from the 131072-row
Q4 proposal head. The signed token map is parsed with `i32::from_le_bytes`; all
131072 entries must be in `0..248320` before a view is returned.

Synthetic tests assert exact geometry, Q/gate interleave row indices, parent
plane offsets, unsupported codec rejection, and valid/negative/out-of-range
signed token IDs. The test-only CPU reference in
`reference/native_mtp_quantized.rs` sign-extends Q8 bytes and preserves the raw
FP16 scale bits. It validates row geometry and scale count but explicitly
rejects numeric Q8 decode/dot because the local reader/schema/import contract
does not establish the equation. Q4 entry points check malformed geometry and
reject numeric decoding because nibble/sign and scale equations remain unknown.
This is an independent format-boundary reference, not a numerical dot oracle or
execution consumer.

## Open codec questions

- The local reader computes encoded total sizes but does not specify full Q4
  nibble-to-signed-code interpretation, nibble ordering, zero point, or whether
  FP16 scales multiply or divide the integer code. The row-split layout summary
  establishes planes and K padding, not the missing equations.
- The separate Ninfer dispatch note records Q8 as `int8 codes + FP16 scale / 32`
  at `KNOWLEDGE/findings/ninfer-op-dispatch-e31bc99.md:51`, but this task's
  authority gate is the reader/schema/import contract. The numeric decoder stays
  disabled until that equation is incorporated into the authoritative local
  contract and independently checked against source.
- No Q8/Q4 numeric row decoder or dot-product expectations are claimed until the
  local authoritative contract establishes equations, scale bit handling and
  code-coordinate rules.
- Token-map values are checked for target-vocabulary bounds, but uniqueness,
  shortlist ordering, and whether duplicate target IDs are legal are not
  established here.
- `NativeMtpViews::resolve` is a prerequisite only. `ModelArtifact::require_mtp`
  remains unchanged and rejects native MTP; resident execution and weights
  loading are not wired to these views.
