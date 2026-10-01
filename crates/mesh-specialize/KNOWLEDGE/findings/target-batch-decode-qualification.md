# Target batch versus ordinary decode

Status: all 20 bounded N=1..5 normal/memcheck/racecheck/synccheck cells pass,
combining 16 trial B passes with four racecheck retry C passes. Prior timeouts
remain retained below.
No native MTP admission, whole-model quality pass, or throughput claim follows.

## Scope and identities

The target checker executes the complete 64-layer model on GPU0, comparing a
recorded N-row verification forward against N ordinary decode steps. Both paths
start from a 32-token prefix at capacity 40, then execute the same three-token
ordinary continuation. Each phase compares every layer's 5,120 hidden words per
row, final hidden words, all 248,320 vocabulary words per row, all bytes of the
128 named state regions including inactive capacity, and the unpoisoned cursor.
Selected-token agreement alone cannot pass.

The tested pending export is
`/home/ndizazzo/dev/mesh/issue1393-q4-snapshot-20260930` on Carrack. It is not a
clean committed revision. The selected-row executable SHA256 is
`6846136ff113b66924091b94ad2b74bb2fdce80ea075c5482050afc211eb626d`.
The artifact SHA256 is
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
The full PTX SHA256 is
`c2640bef7583ee620044304f5c24595d79308cf19c576e53ba18b8dc60cd1839`.
The fixture SHA256 is
`c5a18c194df1cde101455a3b0167097962c8c81c76e6502593c79e4044f7806f`.
GPU0 is RTX 5090, SM120, UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver `615.71.09`.

Source archives, the source-hash manifest, and host build/test logs are retained
in [trial A](../evidence/target-batch-decode-trial-20260930-a/checkpoint.md).
Source hashes passed before and after trial B. No source or PTX changed during
the trial. Device clocks and power samples are in each trial's retained logs;
performance and peak memory were not measured.

## Selected-row trial B

[Results](../evidence/target-batch-selected-trial-20260930-b/results.txt) retain
all 20 row/mode outcomes. The trial ran September 30 at 22:55 through October 1
at 01:13 EDT. Every case uses `--rows N`, the same fixture and arithmetic profiles,
32 GiB host memory limit, no swap, and a 1,200-second timeout with a 30-second
kill grace. Racecheck uses one worker and synchronization limit one.

| N | Normal | Memcheck | Racecheck | Synccheck |
| --- | --- | --- | --- | --- |
| 1 | Pass | Pass | Pass | Pass |
| 2 | Pass | Pass | Timeout | Pass |
| 3 | Pass | Pass | Timeout | Pass |
| 4 | Pass | Pass | Timeout | Pass |
| 5 | Pass | Pass | Timeout | Pass |

Every passing cell exits zero and passes the full exact JSON filter. Sanitizer
passes additionally require a final zero-error or zero-hazard summary. N=1
racecheck completes in 1,147 seconds. N=2..5 racecheck exits 124, produces empty
JSON files, and has no final sanitizer summary. Those cells are failures, not
clean racecheck evidence. The earlier full-matrix memcheck/racecheck timeouts in
trial A remain retained and are not overwritten by partitioned passes.

The EXIT restoration records both `ninfer-qwen38.service` and
`battlecity-comfy.service` active again. A subsequent Ninfer health request
returns HTTP 200 with `{"status":"ok"}`. No GPU1 workload was changed.

## Racecheck retry C

[Results](../evidence/target-batch-racecheck-trial-20261001-c/results.txt) and
[run timestamps](../evidence/target-batch-racecheck-trial-20261001-c/runs.log)
record four successful retries on October 1, 01:25 through 02:49 EDT.
Each exits zero with `gate_failed=0` and zero errors, warnings, and hazards.

| N | Racecheck | Duration, seconds | Exit | gate_failed |
| --- | --- | --- | --- | --- |
| 2 | Pass | 1220 | 0 | 0 |
| 3 | Pass | 1242 | 0 | 0 |
| 4 | Pass | 1257 | 0 | 0 |
| 5 | Pass | 1307 | 0 | 0 |

Only the per-case timeout changed, from 1,200 to 2,400 seconds. The
[scope](../evidence/target-batch-racecheck-trial-20261001-c/scope.txt) retains
32 GiB host memory, no swap, a 30-second kill grace, one CPU worker, and
`--force-synchronization-limit 1`. The fixture, arithmetic profiles, and exact
filter are unchanged. Coverage still includes all 64 layers' 5,120 hidden words
per row, all 248,320 vocabulary words per row, all 128 state regions including
inactive capacity, expected unpoisoned cursors, and the three-token continuation.

The [C hashes](../evidence/target-batch-racecheck-trial-20261001-c/hashes.txt)
match the artifact, executable, PTX, and fixture identities above. The unchanged
filter SHA256 is
`8047824100860987b8c168833c59aab093342a880617a102d62fb8d3f25a17f1`.
Source-manifest checks pass before C and in the
[verified postcheck](../evidence/target-batch-racecheck-trial-20261001-c/target-batch-racecheck-source-check-after-verified-20261001.log).
The [initial postcheck](../evidence/target-batch-racecheck-trial-20261001-c/target-batch-racecheck-source-check-after-20261001.log)
used the wrong manifest path and records a missing file, not a source mismatch.
That failed check remains retained. The
[versions log](../evidence/target-batch-racecheck-trial-20261001-c/versions.log)
records Compute Sanitizer 2026.3.0.0.

The parent independently revalidated all 20 reports with the unchanged retained
filter and fixture: 16 passing B cells and four C retries. Combined coverage is:

| N | Normal | Memcheck | Racecheck | Synccheck |
| --- | --- | --- | --- | --- |
| 1 | Pass B | Pass B | Pass B | Pass B |
| 2 | Pass B | Pass B | Pass C | Pass B |
| 3 | Pass B | Pass B | Pass C | Pass B |
| 4 | Pass B | Pass B | Pass C | Pass B |
| 5 | Pass B | Pass B | Pass C | Pass B |

This closes the bounded target-batch correctness/sanitizer gate only. Trial B's
timeout table and trial A's failed evidence remain unchanged. Retry durations
are sanitizer runtime, not model-performance measurements.

The [runner log](../evidence/target-batch-racecheck-trial-20261001-c/runner.log)
records both authorized services restored active. The parent subsequently
rechecked Ninfer health, receiving HTTP 200 with `{"status":"ok"}`.

## Remaining gates

Native draft arbitrary-activation arithmetic, real native MTP at depths one/four,
full-state rejection and EOS recovery, unchanged model-quality gates, and matched
whole-model timings remain open. The target-batch pass does not admit native MTP,
establish whole-model quality, or qualify performance.

Durable rule: partitioning changes the scope of each report, not the acceptance
criteria. Empty reports, missing sanitizer summaries, and timeout exits never
qualify a case, even if another execution mode passed the same numerical inputs.
