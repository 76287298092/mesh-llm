# Decode scorer qualification

Status: completed scorer-only GPU smoke; full-corpus model quality remains open.
Date: evidence label 2026-09-30; retained runs occurred 2026-09-29 EDT.
Source baseline: `b188c2925aa05108ebb5b1fe7afd195c60254e6c`, with uncommitted
snapshot changes, not a clean build of that commit. The retained
[source manifest](../evidence/decode-scorer-20260930/decode-scorer-source-20260930.sha256)
has SHA-256 `afcc16df6154f738ebed5f006559f7067917f803011218a52222649603c54c98`.
Audit context: `ses_f103b2b19ffePPqhjjfVuHiZaz`.

Decode scoring creates a fresh session for each window. Compact hidden row 0
comes from prefill; compact rows 1 onward come from single-token decode.
The retained `check` still checks the first scored window. A separate
`decode_check` starts at compact row 1 with `targets[1..]` in the first window
with at least two scored rows. Both results contribute to scorer pass status.
The additional result records the hidden-row start, target start, and input
window bounds. One-row windows do not consume the decode check.

The comparator checks integer, nonnegative, bounded execution ranges against
the exact window plan. All four ranges must match between control/candidate
and candidate/repeat, including compact-record repeatability. Legacy prefill
manifests without any execution ranges remain readable; partially reported
ranges and decode manifests without complete ranges are rejected.

The existing cumulative `full_logits_sha256` mechanism remains unchanged.
Use `MESH_SPECIALIZE_SCORE_LOGITS_HASH=on` for full-logit repeat evidence.
No record-format, arithmetic, threshold, admission, or A16 change is included.
This checks scorer reduction and head consistency on decode-produced hidden
rows. It does not independently prove decoder arithmetic.

## Retained GPU smoke

[Trial A](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-a/runs.log)
failed its first command with exit 1 before inference. Its
[CLI log](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-a/score-default.log)
reports `score flags must follow documented order`. No scoring manifest was
produced. This failed attempt remains retained, not counted as a numerical failure.

[Trial B's run ledger](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/runs.log)
records all eight steps exiting 0: default prefill, explicit prefill, prefill
assertion, decode control, candidate, repeat, comparator, and decode assertion.
The [result](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/results.txt)
is `SCORER-SMOKE-PASS`, limited to four fixed streams.

The reports identify device 0 as RTX 5090, SM 12.0, CUDA driver API version
13040, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`. The target-only
Ninfer-v3 artifact SHA-256 is
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
All runs use legacy execution, `bf16-fp64-v1` attention, `exact-a8-v1` FP8,
`nvfp4-native-prefill-integer-decode-v1`, no FP8 split-K, and no MLP workspace.
The candidate label is a comparator role, not an arithmetic candidate.
Full-logit hashing is enabled. The retained
[build hashes](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/expected-hashes.txt)
and [hash check](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/hash-check.log)
identify executable SHA-256
`d670b5e31fce8be5c511d5fc456884d27789a47d40f8bc26d71d56ffae604bfb`
and PTX SHA-256
`3727e477d0691a3359e87de0ecdffc3248883890432958cb247c2db33c32f997`.
Toolchain and clocks were not established by this documentation review.

Default and explicit prefill use 512 input tokens per stream, context 512,
stride 256, and 511 scored targets per stream. Their
[manifests](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/score-default/manifest.json)
and [explicit report](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/score-explicit/manifest.json)
agree exactly on raw score records and full-logit hashes. The
[prefill assertion](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/prefill-assert.log)
also passes eligible historical `direct-source-1` raw-record prefixes on every
stream. Historical full-logit comparison is **NOT RUN** on all four streams
because the old hash is absent or covers a different span.

Decode [control](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/decode-score-control/manifest.json),
[candidate](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/decode-score-candidate/manifest.json),
and [repeat](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/decode-score-repeat/manifest.json)
use identical exact profiles, 257 input tokens per stream, context 128, stride
64, and 256 scored targets per stream. Each stream has the following half-open
ranges. Input execution ranges are window-local; hidden rows index the compact
scored buffer.

| Input window | Target range | Prefill context | Prefill scored input | Decode input | Decode scored hidden |
| --- | --- | --- | --- | --- | --- |
| [0,128) | [1,128) | [0,1) | [0,1) | [1,127) | [1,127) |
| [64,192) | [128,192) | [0,64) | [63,64) | [64,127) | [1,64) |
| [128,256) | [192,256) | [0,64) | [63,64) | [64,127) | [1,64) |
| [129,257) | [256,257) | [0,127) | [126,127) | [127,127) | [1,1) |

The separate `decode_check` passes on compact hidden row 1, target start 2,
input window [0,128), with 126 rows, bit-exact head checks, and maximum host
reference absolute error `3.5650037588652594e-7`. The final window has no decode
rows; the earlier windows establish actual single-token decode execution.

Full-logit SHA-256 values below are equal within each comparison group. Raw
record equality is retained in the assertions and
[artifact hash ledger](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/hashes.txt).
Hashes cover every little-endian BF16 vocabulary value at scored positions in
window order, not just top-k records.

| Stream | Default/explicit prefill | Decode control/candidate/repeat |
| --- | --- | --- |
| wikitext-00 | `5e50e2b9bee6a60f457d9794597a3446f981bc84edfe392362e606cb374590a2` | `86f46ad37826b2f26ab4c2a19bfa8263ea1ad54789d2db38ece240f49196a59b` |
| pg19-00 | `b4c9b9c16c842985311eb071437781e6c8f008fb928a7ddce9b4f7fe57dbbd80` | `c7ff91c5dbefc90ce7e56593a2a877a97acf515811aa3ae5b70636cdf7606308` |
| ninfer-00 | `dcb01f8e8d28da2c3ce7b906ea2d3de058d67c0a1100f8be92890c5f4f2fdda8` | `e02466f97526cf7dd403fe91f52df73d30bfed12582c00f939b54e77a7368e7b` |
| zhwiki-00 | `1a63531c6e4ca1a9ce9853e2cb687ad9246d6c11937f6689e30a785f3c5e6c27` | `24ff6dc4228042d35a9b2859058e19552c935db45aedbac3acec9c82078205f3` |

The [decode assertion](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/decode-assert.log)
passes actual execution ranges, scorer/decode checks, raw records, and full
logits on every stream. The
[same-profile comparator](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/decode-compare.log)
reports zero NLL increase, top-1 agreement 1, zero KL, exact record repeatability,
and full-logit determinism. These are expected consistency results for identical
arithmetic, not qualification of a new model profile.

## Restoration and limits

Both attempts retain service restoration records. Trial A's
[units-after](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-a/units-after.txt)
and Trial B's
[units-after](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/units-after.txt)
show Ninfer and ComfyUI initially active and active afterward. Each retained
helper health file contains `1`, including
[Trial B's helper result](../evidence/decode-scorer-20260930/decode-scorer-trial-20260930-b/ninfer-health-restored.txt).
Separately, the parent independently observed HTTP 200 after restoration.
That parent observation is not an HTTP status preserved in the helper file.

No arithmetic candidate, full-corpus quality completion, default promotion,
speedup, or MTP execution is established or claimed. Individual scoring manifests
retain `model_quality_passed: null` and `quality_gate_status: NOT RUN`.
This smoke qualifies scorer consistency on the tested decode-produced hidden
rows, not independent decoder arithmetic or full Ninfer parity. Regression
fixtures cover malformed bounds, incorrect plans, one-row windows, and comparison
range mismatches; this documentation worker did not rerun tests or GPU work.
