# Native packed Q8 attention and MLP projections

Status: all 16 bounded real-parent cases pass normal, memcheck, synccheck and
partitioned racecheck, with exact all-row two-poison BF16 checks. The original
full-matrix racecheck timeout remains retained. Complete native MTP, arbitrary
activations and whole-model performance remain open.

## Contract

The Rust device implementation uses the full physical packed parents, not
dequantized substitute weights or logical query/gate row geometry. Each parent
runs T1 and T5 with alternating unit and deterministic signed-dyadic inputs.
All output rows must exactly match independent scheduled BF16 values for both
distinct output poisons. Repeat differences and nonfinite values also fail.
T5 input and reference output columns must be pairwise distinct.

Unscaled G32 dot terms lie on a half-unit lattice with absolute sum at most
8192, so every intermediate dot is exactly representable in FP32. FP16 scale
FMA order and ordered four/eight-split reduction remain mandatory. This proof
does not establish internal MMA summation bits for arbitrary BF16 activations.
The independent FP64 mathematical comparison is diagnostic, not a GPU tolerance.

The source schedule is pinned to Ninfer
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`, with Apache-2.0 attribution and
modification notices retained. Correspondence with the running Ninfer binary
has not been established.

| Physical parent | Shape | Source and resident SHA256 |
| --- | --- | --- |
| `weight/000176`, QKV | 14336 x 5120 | `4e0ce18b9abf4e421a07b48c323ddb4c355360212010c51772f2274e69ed08fd` |
| `weight/000177`, gate/up | 34816 x 5120 | `3f909296f5109118596c3af0802d20d451b042a31e1b818c13ae944844ce472e` |
| `weight/000732`, attention output | 5120 x 6144 | `6f409e2b87dfbdf37c0ab22b99945be38e9cbcb3a6a7a30f7000a955ec612caf` |
| `weight/000734`, down | 5120 x 17408 | `53fe6c872e15e1125a96df790783607907deecace4cafc05f7c48d601e7630dd` |

QKV uses eight splits at T1 and four at T5. Gate/up uses four at both sizes;
attention output and down use eight. MTP rounds linear outputs to BF16 before
the separate split, SiLU product and residual operations. Ordinary target fused
epilogues are not interchangeable with these MTP boundaries.

## Execution evidence

All execution used Carrack GPU0, RTX5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`. The supplied exported snapshot is
based on `b188c2925aa05108ebb5b1fe7afd195c60254e6c` plus uncommitted changes,
not a clean commit. Its [source manifest](../evidence/q8-projection-resident-trial-20260930-b/source.sha256)
pins the tested production source, PTX and executable.

PTX SHA256 is
`c2640bef7583ee620044304f5c24595d79308cf19c576e53ba18b8dc60cd1839`;
executable SHA256 is
`51a59e5658cf06addf2ef638b2725e30da8a5e61a6dd9d164dc41ca6e1d93604`.
Rust PTX was built through Just using isolated nightly-2026-09-25, rust-src,
llvm-tools and llvm-bitcode-linker. Original toolchain/build failures are retained.

Trial A passed normal execution but its SSH command timed out. The bounded
remote normal process finished, then the parent sequence stopped without
sanitizer runs. [Partial A evidence](../evidence/q8-projection-resident-trial-20260930-a/parent-status.txt)
is retained separately.

Trial B used a longer foreground SSH timeout and kept each pass bounded at
1200 seconds, MemoryMax 32G and MemorySwapMax 0. On 2026-09-30 EDT:

| Pass | Start / end | Result |
| --- | --- | --- |
| Normal | 14:54:44 / 15:09:25 | Exit 0; all 16 cases pass |
| Memcheck | 15:09:25 / 15:24:07 | Exit 0; all 16 cases pass; zero errors |
| Racecheck | 15:24:07 / 15:44:07 | Exit 124 at the bound; no completed report or hazard summary |
| Synccheck | 15:44:07 / 15:58:53 | Exit 0; all 16 cases pass; zero errors |

Normal, memcheck and synccheck report zero exact BF16, repeat and nonfinite
mismatches over every row and both poisons. The [failed sequence](../evidence/q8-projection-resident-trial-20260930-b/results.txt)
and incomplete racecheck report are preserved. A header-only racecheck log is
not evidence of zero hazards. The subsequent partitioned recovery is recorded below.

Both initially active authorized services were restored active/running and
Ninfer health returned HTTP 200. GPU1 and unrelated workloads were not changed.
Thirteen focused Q8 host tests and six CLI retention tests pass. Subsequent
Clippy-only fixes pass Linux Clippy with warnings denied and the same 13 host
tests; these fixes were not part of the pinned trial-B executable.

The subsequent full host regression passed 634 library tests and 90 xtask tests,
and the no-console-print gate passed. Their logs are retained in the trial-B
evidence directory. These checks precede integration of case-range selection
and native MTP forward; they do not certify either later change or racecheck.

## Partitioned racecheck recovery

Trial C retained the same PTX and arithmetic gates. The CLI gained validated
half-open case ranges and stable original case indices; default execution still
selects all 16 cases. Three case-selection host tests, eight CLI tests and Linux
Clippy with warnings denied pass. The parent corrected a stale test closure
before these checks, without changing its error-retention assertions.

The full post-selection regression passes 637 library tests and 92 xtask tests;
the no-console-print gate also passes. Logs are retained with trial C. Native
forward and target-batch gate changes are not integrated into this snapshot yet,
so these counts do not qualify either pending subsystem.

Eight serial racecheck processes covered `[0,2)` through `[14,16)`, each under
the unchanged 1200-second bound, MemoryMax 32G and MemorySwapMax 0. They ran from
16:36:59 to 17:00:06 EDT on 2026-09-30. Every process exited 0 and reported
`RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`. All original
case indices 0 through 15 appeared exactly once. Every output row and both
poisons passed with zero BF16, repeat or nonfinite mismatches; all source and
resident hashes and T5 column-distinctness checks passed.

The [complete recovery evidence](../evidence/q8-projection-race-groups-20260930-c/results.txt)
and [source manifest](../evidence/q8-projection-race-groups-20260930-c/source.sha256)
pin executable SHA256
`de9da4aa5d022f0c1a2233e0e866ea1aaf7042d489d16ccc8aa4a92a80308738`.
PTX SHA256 remains `c2640bef7583ee620044304f5c24595d79308cf19c576e53ba18b8dc60cd1839`.
The new executable includes case selection and the previously recorded host
Clippy fixes. Normal, memcheck and synccheck evidence remains the separately
pinned trial-B execution, not a claim that those passes used trial C's executable.
Both initially active services were restored active/running and health returned
HTTP 200. No GPU1 workload was changed. Complete native MTP and throughput
admission remain false.

## Durable rule

Retain failed and interrupted trials. Reduce the number of independently
qualified cases per bounded sanitizer process rather than widening numerical
budgets or claiming an incomplete sanitizer run passed. Do not infer native MTP
admission or model throughput from these operator results. All admission,
model-execution and timing-claim flags remain false.
