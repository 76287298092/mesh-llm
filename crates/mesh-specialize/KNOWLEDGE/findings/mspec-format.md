# Internal `.mspec` placement format, version 1

Status: container reader and writer pass macOS and Linux tests. No model conversion,
compiled Qwen package or real specialized inference is qualified by this format.

The fixed 64-byte header is little-endian:

| Bytes | Meaning |
| --- | --- |
| 0–8 | Nine bytes `MESHSPEC\0` |
| 9–11 | Reserved zero |
| 12–15 | Format version, u32, currently 1 |
| 16–23 | JSON directory byte length, u64 |
| 24–31 | Absolute payload start, u64 |
| 32–63 | SHA-256 of the exact directory bytes |

The directory follows immediately. Payload start is exactly the next 256-byte
boundary; padding is zero. Object offsets are **relative to payload start**, so
their placement does not depend on the number of digits in the JSON directory.
Objects have strictly increasing names and occur in that same physical order.
Each starts at the next 256-byte boundary after its predecessor; the file ends
exactly at the final object's end. No arbitrary holes or trailing data are allowed.

The directory binds a model/weight identity, a pinned source repository/revision,
a recipe digest, and objects with name, kind, dtype, shape, layout, relative offset,
length and SHA-256. One object must carry the recipe itself and match its digest.
Tensor layout strings describe storage only; a compiled package must recognize
them. Tokenizer/config/recipe objects use raw U8 vectors. The generic reader never
interprets directory content as an execution graph or dynamically dispatches code.

The prototype bounds directories to 16 MiB, objects to 65,536, shapes to eight
dimensions, identifiers/layouts to 1,024 bytes, and complete files to 128 GiB.
All dimensions and byte lengths are checked for overflow. Byte lengths must agree
with the dtype and shape, including rounded-up nibble storage. These limits cover
the target's BF16 reference and quantized artifacts; they are not model support.

## Content-derived identity

`weights_id` is `sha256:` followed by the lowercase digest of a deterministic
logical inventory. Hash input is the domain
`mesh-specialize/mspec/weights/v1\0`, schema version as u32 LE, then model ID,
source repository, source revision and recipe SHA-256 as length-prefixed UTF-8.
Each string length is u64 LE. Next comes the object count as u64 LE and, in name
order, each object's name, kind tag, dtype tag and layout string; its shape count
and dimensions as u64 LE; its length as u64 LE; and its SHA-256 as a length-prefixed
lowercase string. Placement offsets and the self-referential `weights_id` are
excluded. Repacking alignment alone does not change logical weight identity.

The canonical unit-test inventory has an independently constructed Node.js
`crypto` SHA-256 golden value:
`04a341e6de69e56815fd2920f0a8b300baf300c150186d372f7e2ec8688fb726`.
This pins the domain, field order, little-endian integers and length framing;
mutation tests separately check that every bound field affects the result.

The reader validates the schema and this declared identity, then streams through
every payload and checks its digest before producing `VerifiedArtifact`. It keeps
the file descriptor open and rechecks an object's digest whenever copying it.
Consumers must discard a partial destination on failure and must not execute
weights before all required copies succeed. Renaming a file does not change the
retained descriptor, but in-place modification can still occur; this API does not
claim an immutable filesystem snapshot.

Integrity is distinct from trust and numerical qualification. The recorded source
repository/revision is provenance metadata; the reader does not contact Hugging
Face or prove that a converter used those upstream bytes. The approved runtime
must require an exact supported identity and validate its compiled tensor inventory.
`open_for_identity` rejects a different model or weight identity explicitly.
Unknown layouts, quantization recipes and model schedules remain package concerns.
All version-1 directory metadata rejects unknown fields, including the embedded
identity. The shared runtime-manifest identity type keeps its additive behavior;
the container applies its stricter deserialization locally.

## Ownership

`mesh-specialize::artifact` owns the experimental format, identity, resident
reader and streaming object writer. Keeping the writer here avoids a dependency
from published `model-package` to unpublished `mesh-specialize`. Shipping format
ownership must be settled before integrating conversion into the existing
`model-package` workflow. No second job scheduler, GPU converter dependency or
NInfer container import is introduced. The future host discovery path must use a
verified resident artifact, then apply exact identity and selected-device policy
before ABI startup.

The writer accepts explicit local object files and metadata, validates the whole
directory before hashing large inputs, derives checksums and identity, then copies
one source at a time into a temporary file while rechecking lengths and hashes.
It reopens each source for assembly and rejects changed or replaced content.
It publishes the completed file without replacing an existing destination. This
is container assembly, not Safetensors conversion or Qwen graph construction.

Local validation on September 27, 2026: 65 `mesh-specialize` tests pass with all
features, including 39 artifact checks. All-target/all-feature Clippy with warnings
denied, focused rustfmt and repository no-console-print checks pass. The commands
are `MACOSX_DEPLOYMENT_TARGET=26.0 just with-lld cargo test -p mesh-specialize
--all-features` and the corresponding `cargo clippy -p mesh-specialize
--all-targets --all-features -- -D warnings`. Raw local logs are under
`target/specialize/mspec-local-*` and `target/specialize/mspec-console-check.log`.
No GPU timing, memory-capacity or model correctness claim follows from container
tests.

Review found that the first writer retained one descriptor per object, which
could exhaust process limits for a full checkpoint. It now holds one source open
at a time. A 1,025-object round trip passes in a separate test process with
`ulimit -n 64`; in-place changes, changed lengths and pathname replacement between
passes all reject. An oversized sparse source rejects before hashing. The raw
descriptor-limit test is `target/specialize/mspec-local-low-fd.log`.

Carrack initially passed 66 library and 17 validator tests at `21dd509e3`, plus
all-target/all-feature Clippy. That revision predates the descriptor-limit fix;
its raw evidence is preserved in `target/specialize/mspec-20260927/`. Ninfer's
PID 2697705 and ComfyUI's PID 448118 were unchanged throughout these CPU checks.

Final Linux validation at source `889a3a7879a388c6f8d612fad7afe19aef94f5cd`,
completed September 27 at 00:36:37 EDT: 68 library tests and 17 validator tests
pass, plus all-target/all-feature Clippy with warnings denied. Rust is 1.98.1,
LLVM 22.1.8 on `x86_64-unknown-linux-gnu`. Cargo commands ran serially in temporary
8 GiB user scopes with no swap and 180-second timeouts. The 1,025-object test also
passes under a 64-descriptor limit on Linux. A bounded diff review found no new
integrity issue after the descriptor fix.

[Recorded evidence](../evidence/mspec-20260927/) includes the exact source,
toolchain, test and Clippy logs, descriptor-limit check and service/process state.
The same raw Linux logs remain on both hosts under
`target/specialize/mspec-bounded-20260927/`. Ninfer is the **user** service
`ninfer-qwen38.service`; it remained active as PID 2697705 with 30,046 MiB allocated.
ComfyUI PID 448118 remained at 498 MiB. No service stop or GPU kernel trial was
needed. No Actions run was returned for the source push; this is focused local
and Carrack evidence, not a full product build or CI qualification.

Integration findings: the first compile failed because workspace `sha2` 0.11
digest arrays do not implement `LowerHex`; explicit `hex::encode` resolves that
API difference. The first Clippy run rejected manual evenness handling in nibble
length calculation; checked `u64::div_ceil(2)` expresses the intended operation.
Neither failed attempt ran or qualified model inference.
