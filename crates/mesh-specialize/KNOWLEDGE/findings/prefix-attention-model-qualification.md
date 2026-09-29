# Prefix attention: model qualification

Status: `prefix-parallel-v2` measured faster than `serial-v1` on both inputs in
the completed Carrack model trial, with
exact agreement against the saved direct-source-1 checks and across all 256
generated tokens. This result does not qualify long-context execution, MTP, or
model quality beyond those exact checks.

## Identity and method

The trial used source `40ab690df` and isolated PTX SHA-256
`1453bc8f6f0c376852580127e37bf40baf1ecd108b47806a0c7eb46b0d5c1c9f`. It ran on
Carrack from `target/specialize/reassess-20260927/prefix-candidate-1`. Both
modes used the `vector16 paired-fp64 prmt staged-fp64` base. The compared
schedules were `serial-v1` and `prefix-parallel-v2`.

The balanced runs generated 256 output tokens, used two repetitions per mode,
and ran in forward and reverse mode order. Rates below are the median of the
four measurements per mode and input.

## Correctness and safety

The short and full operator checks passed. Short memcheck, racecheck, and
synccheck, plus full memcheck and synccheck, all exited 0. The combined model
memcheck passed.

The `qwen-stream-check` runs for both schedules and inputs matched the saved direct-source-1 logits, tokens,
and all-state checks. All 256 generated tokens were identical across modes and
repetitions. Both services were restored to active, and health returned HTTP
200.

## Whole-model result

| Schedule | 106-token story (tokens/s) | 512-token pg19 (tokens/s) |
| --- | ---: | ---: |
| `serial-v1` | 38.66148246866766 | 29.755325723761366 |
| `prefix-parallel-v2` | 43.63685801548224 | 39.364233385719565 |

The separate Ninfer ordinary-decode reference was 76.3 tokens/s. It was not a
matched, synchronized serving comparison. This trial adds no long-context or
MTP result and makes no non-exact model-quality claim.

Evidence: [prefix-candidate-1](../evidence/reassess-20260928/prefix-candidate-1/).
