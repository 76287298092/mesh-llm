# Native Q4 operator qualification

The uncommitted snapshot passes synthetic indexed proposal-head qualification
and the real packed 131,072-row proposal head with two bounded dense inputs,
normal execution and all three sanitizers. This qualifies the tested
`GemvR4W1` schedule and inputs only. Arbitrary activations and complete native
MTP remain open; this is not performance, model-quality, or promotion evidence.

## Source and environment

Baseline commit: `b188c2925aa05108ebb5b1fe7afd195c60254e6c`. The tested source is
an **uncommitted snapshot, not a commit**, as recorded in
[scope.txt](../evidence/q4-snapshot-20260930/scope.txt).
The parent-owned [source manifest](../evidence/q4-snapshot-20260930/source.sha256)
pins the snapshot files, including the Q4 kernel, staging/decode helpers,
independent FP64 reference, and exact FP32 schedule reference.

| Artifact | SHA256 |
| --- | --- |
| `source.sha256` manifest | `17f9c1b24bcb7f01b050868a60dcf4f082ef251c758d94edd7c441c9ef7eb4cc` |
| `probes.ptx` | `3727e477d0691a3359e87de0ecdffc3248883890432958cb247c2db33c32f997` |
| `target/release/xtask` | `49db1be2b8008ca870dcef243cdfe98405172f246497cddb66124dd68ec3ff6b` |

PTX and executable hashes are retained in
[hashes.txt](../evidence/q4-snapshot-20260930/hashes.txt).
[versions.log](../evidence/q4-snapshot-20260930/versions.log) records Linux
x86_64 on carrack, GPU 0, NVIDIA GeForce RTX 5090,
UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver `615.71.09`, compute
capability 12.0, CUDA tools 13.4.92, Compute Sanitizer 2026.3.0.0, and Rust
1.98.1 with LLVM 22.1.8. The kernel requires SM120a.

## Synthetic results

Each of the four retained runs passes all seven cases: `k128`, `k160-tail`,
`k163-vector-tail`, `k256-two-splits`, `k5120`, `k128-first-row-tie`, and
`dense-n9-k2179`.

| Run | Case results | Sanitizer result |
| --- | --- | --- |
| [Normal operator](../evidence/q4-snapshot-20260930/operator.json) | 7/7 pass | Not a sanitizer run |
| [Memcheck](../evidence/q4-snapshot-20260930/memcheck.json) | 7/7 pass | [0 errors](../evidence/q4-snapshot-20260930/memcheck.log) |
| [Racecheck](../evidence/q4-snapshot-20260930/racecheck.json) | 7/7 pass | [0 hazards, 0 errors, 0 warnings](../evidence/q4-snapshot-20260930/racecheck.log) |
| [Synccheck](../evidence/q4-snapshot-20260930/synccheck.json) | 7/7 pass | [0 errors](../evidence/q4-snapshot-20260930/synccheck.log) |

Raw FP32 and BF16 logits match the separate lane-accumulation/warp-reduction
schedule reference exactly. Repeated raw FP32 and BF16 outputs and indexed
parent-row mapping checks pass. The independent mathematical FP64 projection
remains a separate check with the unchanged scaled-error bound `2e-4`.
Observed maximum scaled error is `5.960464477539063e-8`; the tested cases also
have zero BF16 mismatches to FP64-oracle rounding. These fixture results do not
establish universal bit equality with the FP64 oracle.

The JSON's static `scope` string still says GPU qualification is not established.
The retained successful runs supersede that pending status only for these seven
synthetic cases, not for MTP integration.

## Host-test history and restoration

The parent reports the completed native Linux run passed 569 unit tests and 26
integration tests. Two earlier exact-result test expectations failed before
correction. Their hand-derived rounded FP32 results are `3 - 2^-21` and
`1 - 2^-21`, rather than idealized exact-arithmetic expectations. This was a
test-expectation correction, not a relaxation of the independent FP64 bound.
The earlier failure remains part of the qualification history; no failed
evidence is removed. The GPU evidence directory does not contain the native
Linux host-test logs, so those counts and corrections are parent-reported,
not independently revalidated by this documentation update.

[units-after.txt](../evidence/q4-snapshot-20260930/units-after.txt) records both
`ninfer-qwen38.service` and `battlecity-comfy.service` restored to their initial
`active` state. The parent reports restored NInfer HTTP 200; the retained
[health-restored flag](../evidence/q4-snapshot-20260930/ninfer-health-restored.txt)
is `1`, not an HTTP response transcript.

The synthetic runs above do not qualify real weights, complete native-MTP
execution, model quality, throughput, or default dispatch. The broader
[native MTP packed-view finding](native-mtp-views.md) is unchanged.

## Real resident head continuation

The parent-owned continuation used the same baseline and unchanged Q4 PTX,
with resident packed storage rather than a replacement or dequantized head.
[The source manifest](../evidence/q4-resident-trial-20260930-a/source.sha256)
pins this uncommitted export. PTX SHA256 remains
`3727e477d0691a3359e87de0ecdffc3248883890432958cb247c2db33c32f997`;
the tested executable is
`c030a65b89225527fbb14009b351c7a35259cfa98730a636b98a38d3da4e03ad`.

The real head is `weight/001070`, shape `131072 x 5120`, group size 64,
layout `row_split_k128_v1`. Its 356,515,840 bytes contain 335,544,320 packed
code bytes and 20,971,520 FP16-scale bytes. Source, expected and fresh resident
readback hashes all equal
`388c9862e28fbba581e7ba2ea43cd9d62b6c0ad5d13966192c66a2f2cd17cbe8`.
The separate signed INT32-LE map `weight/001071` has 131,072 entries and
524,288 bytes, with SHA256
`c348bf8d2e70d502718ebb8eeefdc794936dc8462ae357b802ba0a987fad756c`.
All source/resident map bytes match; observed target IDs span 0 through
248,076 inside the separate 248,320-token target vocabulary.

| Run | Real-weight cases | Sanitizer result |
| --- | --- | --- |
| [Normal](../evidence/q4-resident-trial-20260930-a/normal.json) | 2/2 pass | Not a sanitizer run |
| [Memcheck](../evidence/q4-resident-trial-20260930-a/memcheck.json) | 2/2 pass | [0 errors](../evidence/q4-resident-trial-20260930-a/memcheck.log) |
| [Racecheck](../evidence/q4-resident-trial-20260930-a/racecheck.json) | 2/2 pass | [0 hazards, errors or warnings](../evidence/q4-resident-trial-20260930-a/racecheck.log) |
| [Synccheck](../evidence/q4-resident-trial-20260930-a/synccheck.json) | 2/2 pass | [0 errors](../evidence/q4-resident-trial-20260930-a/synccheck.log) |

Both `alternating-plus-minus-one` and `dense-signed-dyadic` check every raw
FP32 and BF16 output against the independent scheduled reference on two
distinct poison repeats. All 131,072 rows have zero bit mismatches,
nonfinite values and repeat mismatches. The separate mathematical FP64 gate
keeps its `2e-4` scaled-error ceiling; maximum observed error is zero for
these inputs. First-tied BF16 selection agrees at proposal row 62,132 mapped
to target token 838, and row 22,992 mapped to target token 96,795.

Serialized Just checks passed 22 focused Q4 host/reference tests, six CLI
failure-retention tests, Clippy for mesh-specialize and xtask, and the release
build. The full regressions passed
[621 library tests](../evidence/q4-resident-trial-20260930-a/q4-resident-full-regression-20260930.log)
and [84 xtask tests](../evidence/q4-resident-trial-20260930-a/q4-resident-xtask-regression-20260930.log).
Both services were restored to their initial active state as recorded in
[units-after.txt](../evidence/q4-resident-trial-20260930-a/units-after.txt).
The restoration flag is 1; a subsequent parent health request returned HTTP
200. GPU execution used Carrack GPU 0 only, with serial 1,200-second,
32-GiB host-memory bounds. No GPU 1 workload was changed.

Durable rule: full-shortlist arithmetic and map checks must precede native
draft integration. These bounded inputs do not qualify arbitrary activation
arithmetic, a complete MTP step, recovery, target batching, quality or model
throughput. All reports retain `native_mtp_admitted`, `model_executable` and
`timing_claim` as false. No default dispatch or quality threshold changed.
