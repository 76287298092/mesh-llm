# Native MTP Q8 FC resident qualification

Status: eight sparse and bounded dense-input cases pass against the real resident FC parent under normal execution and all three sanitizers. This qualifies C4/T1 and C8/T5 on the recorded patterns only. Arbitrary activations, native MTP admission, model execution, and throughput remain unqualified. The original sparse evidence below is retained.

## Scope and exactness

The isolated, uncommitted Linux snapshot ran the resident native Q8 FC candidate against the saved binding for `weight/000725`, shape `[5120,10240]`, with 32-element groups and an identity row map. Its physical parent is 55,705,600 bytes: 52,428,800 code bytes followed by 3,276,800 scale bytes. The source copy and chunked resident readback both matched SHA-256 `7b3005eafd7bd7025cc8e7e145ed390f9879828073b166269ec2472f23072c69`.

Each candidate ran two sparse patterns. The group sweep places one nonzero BF16 activation in each of 320 groups, at lane `group % 32`, with factors `+1,-1,+0.5,-0.5,+2` rotated by group and token. The last-K probe places the sole nonzero at K=10239, group 319, iteration 19, split 7, second group, lane 31. Each run initializes the output to a different poisoned BF16 value (`0x7fc1` and `0xffc2`) before launching.

| Candidate | Pattern | Tokens | Outputs per repeat | Result |
| --- | --- | ---: | ---: | --- |
| C4/T1 | Group sweep | 1 | 5,120 | Both repeats exactly match all rows of the independent scheduled-FP32 BF16 reference |
| C4/T1 | Last-K | 1 | 5,120 | Both repeats exactly match all rows of the independent scheduled-FP32 BF16 reference |
| C8/T5 | Group sweep | 5 | 25,600 | Both repeats exactly match all rows of the independent scheduled-FP32 BF16 reference |
| C8/T5 | Last-K | 5 | 25,600 | Both repeats exactly match all rows of the independent scheduled-FP32 BF16 reference |

Every case had zero BF16 mismatches, repeat mismatches, or non-finite outputs. Comparison is exact; no GPU tolerance was added or changed. FP64 mathematical values and bounds are diagnostics only.

## Trial identity and verification

The trial used an uncommitted isolated source snapshot based on baseline `b188c2925aa05108ebb5b1fe7afd195c60254e6c`. Its source manifest is [`fc-resident-source-20260930-a.sha256`](../evidence/fc-q8-20260930/fc-resident-source-20260930-a.sha256). PTX SHA-256: `850a8b0e8e93957631938b065d201122d85db6489737dc7b5a3f977f8391af88`; `target/release/xtask` SHA-256: `edc1fe14479427503ae62cbc63fb5e1b88e8ce0a955e236e0e57d5da601b4a6b`. The resident trial used an RTX 5090 (SM 12.0, driver 615.71.09), CUDA toolkit 13.4.92, Rust 1.98.1, and Compute Sanitizer 2026.3.0.0.

Normal execution, memcheck, racecheck, and synccheck passed all four cases. Memcheck reported 0 errors, racecheck 0 hazards (0 errors and warnings), and synccheck 0 errors. The focused Linux operator tests (5), CLI tests (6), Clippy, and release build passed on the isolated snapshot. Ninfer and ComfyUI services were restored active, and Ninfer health returned HTTP 200.

Evidence: [scope](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/scope.txt), [hashes](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/hashes.txt), [normal report](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/normal.json), [pass summary](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/results.txt), [memcheck summary](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/memcheck.log), [racecheck summary](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/racecheck.log), [synccheck summary](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/synccheck.log), [service states](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/units-after.txt), and [restored service health](../evidence/fc-q8-20260930/fc-resident-trial-20260930-a/ninfer-health-restored.txt).

Reproduce normal execution on the recorded Linux snapshot with:

```sh
target/release/xtask specialize native-mtp-q8-fc-resident-check --artifact /data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer --ptx /home/ndizazzo/dev/mesh/issue1393-q4-snapshot-20260930/fc-q8.ptx --device 0 --output /tmp/fc-q8-resident-normal.json
```

Run the same command under Compute Sanitizer `memcheck`, `racecheck`, and `synccheck` for the recorded sanitizer passes. The reports explicitly keep `native_mtp_admitted=false`, `model_executable=false`, and `timing_claim=false`; this is sparse real-weight operator evidence, not dense activation, complete MTP, model, or performance qualification.

## Dense continuation, 2026-09-30

The same physical parent and unchanged PTX now pass two additional dense patterns for each candidate. Alternating units change sign by lane parity and a token-specific group bit. The signed-dyadic pattern rotates `+1,-1,+0.5,-0.5,+2` by K position and token. Both patterns populate every K lane; all five C8 input columns and their complete reference output columns are pairwise distinct.

Each G32 code dot is a half-integer with every partial sum bounded by `32*128*2 = 8192`. These values are exactly representable in FP32 regardless of MMA's internal summation order. The independent reference still preserves each scale FMA and the prescribed eight-split reduction. This proof applies to the bounded fixtures, not arbitrary model activations. FP64 values remain diagnostics, never a GPU tolerance.

All eight cases compared every BF16 output in both distinct poison repeats. Normal, memcheck, racecheck and synccheck each exited 0, with zero exact BF16 mismatches, non-finite outputs or repeat mismatches. Memcheck and synccheck reported 0 errors; racecheck reported 0 hazards, errors and warnings. Both authorized services returned to their initial active state and Ninfer health returned HTTP 200. GPU1 had no trial workload.

The uncommitted exported snapshot remains based on `b188c2925aa05108ebb5b1fe7afd195c60254e6c`. PTX is unchanged at `850a8b0e8e93957631938b065d201122d85db6489737dc7b5a3f977f8391af88`; the trial executable hashes to `292eecd128954eb2277fbcd7b2a3bec1072b7a3ee44f34fffb220acba470f186`. The [scoped source manifest](../evidence/fc-q8-20260930/fc-resident-dense-source-20260930-a.sha256) supplements the original full export manifest. Thirteen focused Linux tests and the release build passed. Clippy initially rejected a test-only `chunks_exact_mut` call; that failure is retained. After replacing it with `as_chunks_mut`, all thirteen tests and Clippy for both mesh-specialize and xtask passed. The correction did not change the release trial arithmetic.

Evidence: [normal report](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/normal.json), [all four runs](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/runs.log), [pass summary](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/results.txt), [sanitizer logs](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/memcheck.log), [failed Clippy](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/fc-dense-clippy-20260930.log), [corrected Clippy](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/fc-dense-clippy-fixed-20260930.log), and [restored service states](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/units-after.txt).

Durable rule: dense exactness needs a proof for the code dot and the actual scale/reduction schedule, plus all-row checks in every live column. Repeat agreement or sparse probes alone do not establish it. All admission and timing flags remain false; complete native MTP and whole-model competitiveness are still open.

The post-change isolated host regression passed all 608 mesh-specialize library tests and all 78 xtask tests. The Just no-console-print gate also passed. These host checks supplement the recorded operator GPU gates; they do not establish model parity. Logs: [library regression](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/fc-dense-full-host-regression-20260930.log) and [xtask regression](../evidence/fc-q8-20260930/fc-resident-dense-trial-20260930-a/fc-dense-xtask-regression-20260930.log).
