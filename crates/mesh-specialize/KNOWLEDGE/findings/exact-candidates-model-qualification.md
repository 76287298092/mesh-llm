# Prefix attention and FP8 reuse qualification

Source `fcff4173a39f41eed673fb476a8b30a3c8a0fa33` passed the Carrack GPU0
qualification sequence in `exact-candidates-2`. PTX SHA-256 was
`1453bc8f6f0c376852580127e37bf40baf1ecd108b47806a0c7eb46b0d5c1c9f`.

All modes used stream execution with vector16 FP8 projections, paired-fp64
A/B projections, PRMT NVFP4 projections, and staged FP64 attention.
Baseline used serial attention coefficients and separate FP8 quantization.
The candidates added `reuse-input`, `prefix-parallel-v2`, or both.

## Measured decode

Each rate is the median of four measurements: 256 output tokens, two
repetitions, forward and reverse mode order on RTX 5090.

| Candidate | 106 input tokens, tok/s | 512 input tokens, tok/s |
| --- | ---: | ---: |
| Baseline | 38.66 | 29.75 |
| FP8 reuse only | 39.80 | 30.42 |
| Prefix attention only | 43.63 | 39.36 |
| Both | 45.10 | 40.53 |

The separate Ninfer ordinary-decode reference remains 76.3 tok/s. These are
not synchronized matched serving measurements. Native MTP throughput remains
unmeasured for this implementation.

## Qualification and limits

FP8 reuse operator checks and memcheck, racecheck, and synccheck passed.
Prefix attention short/full operator checks, short memcheck/racecheck/synccheck,
and full memcheck/synccheck passed. Combined model memcheck passed.
All three candidate modes at both input lengths matched saved direct-source-1
logits, token IDs, and all checked state. All 256 generated tokens matched
across modes and repetitions. The script completed `ALL-DONE`.

The first attempt, `exact-candidates-1`, failed before model testing because
the FP8 operator test used prefill launch geometry for a decode kernel.
Commit `fcff4173a` corrected geometry and made FP32 equality explicitly bitwise.
The successful run is a new evidence directory, not an overwrite of that failure.

This does not qualify longer-context combined execution, native MTP, or
nonexact A16 arithmetic. Both initially active services were restored;
Ninfer health returned HTTP 200.

Evidence: [exact-candidates-2](../evidence/reassess-20260928/exact-candidates-2/).
