# Offline Ninfer artifact preservation

Status: stage-one assembler implemented; integration tests and real-payload import
belong to the parent validation pass. This is byte preservation, not a runtime
consumer or a performance result. The preserved profile is explicitly
`model_executable: false`.

## Boundary and provenance

The user-approved offline exporter is
`validation/scripts/export_ninfer_bundle.py`. It uses Ninfer's official reader
outside Rust, with reader revision
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`. Rust reads only that exporter's
version-one `manifest.json` and `payload.bin`. No `.ninfer` framing parser, codec,
imported compute implementation, or runtime graph interpreter is added.

The assembler requires these exact source declarations:

- Artifact SHA-256:
  `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
- Original artifact length: 23,719,715,844 bytes.
- Profile/model ID: `qwen3.8-27b:ninfer-preserved-v1`.
- Repository: `Neroued/ninfer:artifact`; the `.mspec` source revision is the
  original artifact's 64-hex SHA-256, not the reader checkout revision.

The original source hash is verified by the offline exporter, not independently
reconstructed by Rust. The assembler verifies the bundle's own payload and every
selected object's bytes. Claimed source metadata is not a signature: a fabricated
bundle can repeat these source declarations. Keep the approved exporter and its
before/after source-hash evidence with the import report.

The locally copied parent export manifest at
`target/specialize/reassess-20260927/ninfer-preserved-manifest.json` contains
845 storage entries, 658,154 manifest bytes, and a declared 21,196,809,984-byte
payload. These are inspected manifest declarations, not this worker's verification
of the real payload. Resources use `encoding: raw_bytes_v1`; tensor metadata
includes scalar `shape: []`, BF16/FP32/INT32, FP8 row scales, NVFP4, and Q4/Q8
formats. No original-object shape-to-byte calculation is attempted: these are
packed physical objects whose codec semantics belong to future consumers.

## Validation and assembly

`checkpoint::ninfer_bundle::convert(input_directory, output)` requires a new
output path and uses the existing `artifact::writer::write_artifact` unchanged.
The crate-local `parse_manifest(bytes)` API applies metadata checks independently
of the filesystem so a later reviewed adapter can validate `recipe.json` through
the same parser. It does not verify payload bytes or authorize execution.

The import contract enforces:

- A 4 MiB manifest limit before parsing, serde's default nesting bound, strict
  version-one envelope fields, exact pins/profile, and fixed `payload.bin` path.
  Source files must be regular files, not symlinks, inside the resolved input
  directory. Tensor/resource metadata extensions remain opaque.
- Exactly `text` and `mtp` components; required text embedding/output-head and
  `proposal/head` / `proposal/token_ids` bindings; every binding, nested use
  auxiliary, and component resource reference resolves to an exported source ID.
  Unreferenced exported storage is rejected. Use parameters must name bindings.
- Equal storage/object cardinality and matching source IDs, storage names, and
  byte lengths. Names are `storage/000000` onward; source IDs are unique.
- Nonempty, checked, nonoverlapping ranges with minimal 256-byte alignment and
  no trailing payload. Bundle gaps must be zero. The selected byte total need
  not equal the original artifact's length, which includes excluded components
  and original padding.
- Payload byte length, streaming whole-payload SHA-256, and per-object SHA-256
  before artifact writing. The writer checks expected hashes again during its
  own hashing and assembly passes.
- Preservation flags require no requantization or layout transformation, every
  selected byte copied, `vision`/`dflash2` excluded, and no execution claim.

Every selected physical object becomes a `.mspec` Tensor with dtype U8, shape
`[bytes]`, layout `ninfer-preserved-storage-v1`, a source payload range, and its
expected hash. This includes resource blobs; the original resource/tensor kind
stays in the recipe. The exact original manifest bytes become `recipe.json`, a
Recipe/U8 object with `raw-v1` layout. Configuration, MTP/proposal metadata,
bindings, source IDs, provenance, and extensions therefore participate in the
content identity without executing any of them.

After writing, `VerifiedArtifact::open_for_identity` validates the complete
artifact. Each object is additionally copied to a sink through its digest-checking
copy API. Both passes use bounded buffers. No storage-sized host allocation is
needed. Import streaming buffers are 64 KiB, and object count is capped at 65,535
plus the recipe. The existing container's 128 GiB cap and stricter directory-size
limits still apply; nothing relaxes the container schema or raw-v1 identity.

The report includes artifact identity/size, original source declarations, recipe
and payload hashes, per-format object/byte totals, quantized totals, the complete
source-ID-to-mspec-name/hash mapping, and verified object/byte counts. Quantized
report labels are the pinned export's `fp8_e4m3fn_row_bf16`, `nvfp4`,
`q4_g64_fp16`, and `q8_g32_fp16`; these labels do not activate codecs.

## Command and failure handling

After the parent builds the workspace through its supported Just workflow:

```sh
target/debug/xtask specialize ninfer-bundle-import \
  --input-directory /path/to/offline-bundle \
  --output /path/to/NEW-preserved.mspec \
  --report /path/to/NEW-import-report.json
```

Both output files refuse replacement. Failed imports produce a report with
`all_passed: false` when the report path can be created. A failure during final
readback may leave the published artifact in place for diagnosis. Do not infer
success from file existence. Keep failed evidence and retry with new paths.

No runtime consumers change in this step. The existing compiled raw-v1 inventory
rejects this profile. Conversion success cannot establish model execution,
quality, throughput, MTP correctness, or performance parity. Future consumers
need separate shape/layout, numerical, and execution qualification.

## Validation scope and durable rule

Focused Rust tests cover a small multi-buffer synthetic closure with the claimed
source pins, exact object/recipe preservation, provenance, identity changes from
metadata changes, raw-v1 rejection, overwrite refusal, wrong pins/schema/path,
unknown and missing references, metadata mismatches, bad/overflowing ranges,
payload/object tamper, nonzero padding, oversize manifests, and nonregular or
symlink sources. CLI tests cover bad arguments and report overwrite refusal.

Standalone `rustfmt --edition 2024 --config skip_children=true` completed for the
new Rust files on macOS. This worker did not run Cargo, Rust tests, Clippy, Git,
SSH, GPU jobs, or a real-payload import. Parent integration/build/readback evidence
must be recorded separately. GPU architecture, driver, clocks, and model timings
are not applicable to this CPU-only implementation step. Before/after model
performance and import resource measurements are not measured.

Durable rule: a content-verified preserved storage inventory is not an executable
model. Keep source provenance, physical-byte preservation, and runtime support as
separate gates, and retain the exact metadata bytes in artifact identity.
