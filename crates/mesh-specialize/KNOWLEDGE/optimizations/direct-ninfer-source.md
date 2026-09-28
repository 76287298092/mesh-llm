# Direct NInfer model source

Status: implemented and qualified for the bounded target-text cases documented
below. No offline conversion or `.mspec` intermediate is required. Real-file
hashes, independent canonical hashes, GPU readback, model checks, sanitizers,
quality and throughput measurements are recorded; general performance parity
and native speculation remain open. The user explicitly
authorized native reading for this continuation; the earlier plan's prohibition
on a native-container parser does not apply to this task. No upstream reader or
compute implementation is imported.

## Fixed source and model identity

Only the observed single-file v3 artifact is executable through this source:

- Whole-file SHA256: `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`
- Source bytes: `23,719,715,844`
- Model ID: `qwen3.8-27b:text:ninfer-v3-control-v1`
- Weights ID: `sha256:` followed by that whole-file SHA256
- Source repository: `Neroued/ninfer:artifact`
- Source revision: the whole-file SHA256, not an inferred upstream checkpoint commit
- Canonical target-text inventory: `1,589` tensors, `20,375,588,160` bytes

The opaque source artifact ID is reported for diagnostics but is not treated as a
checksum. Native `weights_id` binds the original file, including its directory and
non-text payload. It is intentionally not the `.mspec` logical-inventory digest.

## API and ownership

`artifact::model_source::ModelArtifact` selects `.mspec` or `.ninfer` by magic,
not the path suffix. Its methods are `open`, `open_for_identity`, `identity`,
`directory`, `require_identity`, `copy_object`, and `verification_report`.
Both variants retain their backend descriptor. Each backend validates the actual
file it opens after the dispatch sniff. A path replacement after opening does not
redirect subsequent reads.

The original `artifact::reader::VerifiedArtifact`, its identity calculation, and
`.mspec` writer/framing are unchanged. Existing import/readback tools continue to
use that reader. The native branch owns `packages/qwen3_8_27b/native_source.rs`;
framing and strict directory validation belong to `artifact/ninfer.rs`, and the
fixed mapping/permutations belong to `packages/qwen3_8_27b/native_views`.

Native opening performs these steps before returning a usable model source:

1. Validate single-file v3 framing and the raw directory. Reject the wrong total
   size before hashing.
2. Stream the whole file and require the exact source SHA256.
3. Plan canonical views from the validated native JSON. Compute every canonical
   tensor digest from actual output bytes, including the checked permutations.
4. Build and validate an aligned, sorted **virtual** `Directory`. A single virtual
   recipe object contains the exact raw directory JSON string, source pin, model
   mapping-contract identity, and transform/consumer ABI contract. Its SHA256 is
   computed from its real serialized bytes.
5. Require the strict native inventory and rehash the entire retained source to
   detect modifications during canonical verification.

Virtual offsets describe an internal canonical address space. They are never
passed to the native file reader as physical offsets. No virtual tensor digest
is fabricated from zeros or inherited from a fused source object. The historical
`safetensors-row-major-v1` layout is a consumer ABI, not a claim that the source
was a Safetensors file.

## Copy and failure contract

`Copy` views stream through the native reader's 64 KiB buffer. Large embedding or
vocabulary-head tensors are never materialized as complete host arrays by this
adapter. Other views cap each source/output allocation at 128 MiB, with a 256 MiB
aggregate source/output ceiling. Mapping validation checks inverse permutations
without a third complete inverse buffer. Smaller schema/recipe allocations are
separate from this tensor scratch bound.

Every tensor copy uses a checked range inside its original storage object, applies
the fixed transform, and verifies its canonical digest again. Any copy error,
including a checksum, length, missing-object, or destination-write error, latches
the native source dirty and rejects further copies. A streaming destination may
already contain partial or invalid bytes when the error arrives. It must be
discarded and never executed. The resident loader already keeps the arena local
and drops it if any object copy fails; no partial `ResidentWeights` is returned.
The file descriptor is retained, not an immutable filesystem snapshot.

The `model_source` report records startup verification wall time, source identity,
two whole-file hash passes, canonical tensor count/bytes, buffer bounds, and dirty
state. This startup I/O is outside model prefill/decode timing. No startup rate or
memory-peak measurement has been made here.

## Supported and rejected entrypoints

The following existing commands now route ordinary target-text weights through
`ModelArtifact` and can open the pinned native source without conversion:

- `qwen-model-bench`
- `qwen-model-profile`
- `qwen-model-score`
- `qwen-stream-check`
- `qwen-chunked-bench`

Their existing Linux, SM120a, PTX, context, arithmetic-profile, and request bounds
remain in force. The resident loader and public artifact-taking kernel wrappers
use the same generic source. Mechanical type routing does not alter kernel math.
The encoded embedding/F32 GDN consumers are described separately in
[ninfer-parameter-consumers.md](ninfer-parameter-consumers.md).

The native strict gate requires FP8 E4M3 embedding plus BF16 row scales, and F32
`A_log`/`dt_bias` on all 48 GDN layers. It omits the raw checkpoint's 32 unused
`k_scale`/`v_scale` tensors and all MTP tensors. The raw-v1 profile keeps its exact
old identity, dtype, shape, tensor-count, and byte-count checks. The separately
preserved `qwen3.8-27b:ninfer-preserved-v1` `.mspec` profile remains non-executable.

`qwen-mtp-reference` and `qwen-mtp-check`, including the public MTP kernel entry,
explicitly reject native sources until Q8/Q4 MTP consumers exist. Packed MTP bytes
are not interpreted as the old BF16 head.

The full-model CPU reference, layer tracing, and corresponding legacy CPU-reference
check explicitly reject native sources. The standalone embedding/projection and
attention reference-input loaders also reject them before large BF16-table
allocation. Reference-based residency and GDN/attention trials reject through
those same guarded helpers. Native target-text GPU execution does not certify
these old BF16-only reference paths. Independent native model-oracle support is
separate work.

## Tests and validation limits

Added bounded tests cover streamed subranges, transformed canonical digests,
wrong digests, range overflow, allocation limits, exact source pin checks, virtual
recipe/tensor metadata, dirty latching, destination failure, and retained-file
behavior after pathname replacement. Tiny native fixtures deliberately bypass
model admission only inside tests; public opening rejects them. They do not
represent the 23.7 GB pinned model.

A tiny real `.mspec` writer/reader test verifies magic dispatch under a misleading
`.ninfer` suffix, unchanged directory and identity, object copies, and expected
identity rejection. Independent captured upstream metadata tests adapt only the
specified native encoding differences, verify exact native totals, and reject
missing/extra tensors, wrong dtypes/shapes/source identity, changed raw metadata,
and the preserved non-executable profile. Parent-owned planner tests separately
cover the actual native directory fixture and mapping transforms.

Worker validation: Rust 2024 rustfmt on the touched files, with child traversal
disabled. No Cargo, Git, SSH, GPU execution, service control, or delegation was
performed by this worker. Parent owns compilation, tests, Clippy, actual-source
opening, device qualification, and subsequent evidence. Source revision: current
uncommitted parent worktree; not queried. GPU driver/toolchain/clocks and
performance before/after: not measured. No arithmetic parity, speedup, quality,
long-context, or production-serving claim is made.

## First direct-file qualification, September 28

At `9b627a3d6`, independent Rust inspection matches all 845 selected physical
object hashes against the official-reader export. Nine synthetic GPU parameter
cases pass. Both 106-input/two-decode-step and 512-input/two-decode-step same-source
legacy/stream checks pass complete state and BF16-logit identity. Three repetitions
of 256 fixed outputs measure stream decode26.48/18.16 tok/s, respectively; these
are not Ninfer throughput parity. No EOS token occurs in those generated runs.

The 49,148-position, four-domain 512/256 scoring campaign passes its sampled
independent scorer/head checks. Native-file control mean NLL1.62073568 versus
Ninfer BF16-KV1.620777 is a corpus-level comparison, not full arithmetic parity.
Both services were restored. Raw evidence: target/specialize/reassess-20260927/
direct-source-1 on Carrack; compact durable evidence follows after qualification.

Follow-up diagnostics add independent NumPy canonical-plane hashing and complete
GPU resident-weight readback before stream-check execution. These do not change
inference arithmetic or timed benchmark execution. `ninfer-inspect --canonical`
exposes the virtual tensor hashes for cross-checking. Tests cover the reference
transformations independently. Full canonical cross-check/readback and bounded
native-file sanitizers are still pending at this checkpoint.


## Completed bounded qualification

Implementation `9b627a3d6`; full diagnostics `fd500c885`. The current local
checkout is `/Users/ndizazzo/dev/worktrees/ninfer-direct-runtime` after verified
archive-snapshot recovery. Carrack retained the implementation and evidence.

Independent NumPy reshape/transpose hashing matches every one of the **1,589**
canonical arrays, **20,375,588,160 bytes**, including source dtype and shape.
GPU readback under each sanitizer matches all 1,589 canonical hashes before
model execution. The old `.mspec` control also matches its captured `ab33f730e`
logits, selected tokens and all state hashes over 33 steps; no mismatch.

Parameter and direct-model memcheck/synccheck report zero errors. Racecheck
reports zero hazards with `--force-synchronization-limit 1` and one worker.
The instrumented model case is two inputs plus one decode step; it is not full
long-context sanitizer coverage. Noninstrumented native model equivalence
covers 106/512 inputs plus two decode steps. All three 256-output benchmark
repetitions complete without EOS before the budget. Services were restored.

| Native-file stream workload | Prefill tokens/s, median | Decode tokens/s, median |
| --- | ---: | ---: |
| 106 input / 256 output | 306.41 | 26.48 |
| 512 input / 256 output | 327.62 | 18.16 |

The corresponding installed Ninfer MTP0 BF16-KV decode reference is about76.3
at both prompt lengths. Native loading removes the source-weight ambiguity;
it does not establish throughput parity or erase arithmetic/algorithm gaps.

Scoring:49,148 positions, context512/stride256, four domains. Native control
mean NLL1.6207356816 versus Ninfer1.6207768509 (about−0.00254% relative).
This is average teacher-forced likelihood, not full-logit agreement, reasoning
quality, long-context quality, or serving equivalence. Source arithmetic
mismatches are retained in the import contract.

Validation:353 macOS library tests;461 Linux library tests plus26 validator
tests; Clippy on both host and Linux target; Just PTX/no-console checks;
38 Python validation tests. Counts describe separate scopes, not one universal
qualification total.

Durable evidence: `evidence/reassess-20260928/direct-source-summary.json`,
`direct-source-1/`, `native-final-checks-1/`, and `mspec-regression-1/`.
Large raw score records remain under the corresponding target directories,
identified by their `raw-sha256.txt` manifests.

### Usage and limitations

Ordinary `qwen-model-bench`, `qwen-model-score`, `qwen-stream-check`, and the
bounded `qwen-chunked-bench` entrypoints accept the original `.ninfer` file as
`--artifact`; they require no conversion or intermediate `.mspec` file.
`ninfer-inspect --artifact FILE --output NEW_JSON --canonical` emits verified
logical tensor hashes without GPU execution. `--hash-objects` instead inspects
physical source objects. Output paths must be new.

Only the observed **single-file v3, pinned Qwen3.8-27B target-text artifact** is
admitted for execution. Other native models and multi-file artifacts are not
silently accepted. MTP/Q4 proposal execution, vision/DFlash2, and old BF16-only
CPU model references remain explicitly unsupported. Chunked/long-context and
serving/concurrency performance are not qualified by the short trials above.
Full-file and canonical hashing currently add roughly31seconds of verification
at startup in the scoring campaign; this is outside model throughput timing.
