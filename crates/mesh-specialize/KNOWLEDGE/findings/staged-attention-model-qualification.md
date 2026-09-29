# Staged FP64 attention: model qualification

Status: qualified as an opt-in ordinary-decode schedule at source
`1f520b1aeb1b0101946ad916e0838d5852890bd6` on 2026-09-28. It is not
the default, and neither graph decode nor native MTP is qualified here.
The exact three-schedule combination plus staged attention remains substantially
slower than Ninfer's matched ordinary decode.

## Identity and method

The original `.ninfer` artifact is
`/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`, SHA-256
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
Runtime identity is `qwen3.8-27b:text:ninfer-v3-control-v1`. GPU 0 was an RTX
5090, SM120, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver
615.71.09. The `xtask` SHA-256 was
`548f9a0a4b687491c3faa11836042086cc1dbe14e6bbc22a94f80d4e442a6ae6`;
the PTX SHA-256 was
`47a6cb946ab572809c1af7117c6f3070043a91384fdce7da0174a06a801ae762`.
The trial used the pinned Carrack checkout and artifact, a single GPU-exclusive
script, and two active user services stopped and restored by an EXIT trap.
`gpu-before.csv`, process samples, and before/after service states are retained.
Clocks were observed, not fixed. The RTX 3080 was not used.

Reproduce with the source/PTX hashes above, the fixture files
`target/specialize/reassess-20260927/tokens-{story-128,pg19-512}.txt`, and
`target/specialize/reassess-20260927/run-staged-attention.sh` on Carrack. The
script requires a clean checkout at the expected HEAD and the exact PTX hash;
it runs `xtask specialize attention-staged-check`, `qwen-stream-check`, and
`qwen-model-bench`, including bounded sanitizer runs. Saved outputs are under
`KNOWLEDGE/evidence/reassess-20260928/staged-attention-1/`. Its terminal log
ended with `ALL-DONE`. Both units were initially active and restored active;
Ninfer health returned HTTP 200 after the trial.

## Correctness and safety

The staged operator's short nine-case and full eleven-case matrices passed
strict control FP32/BF16 bit equality, the independent FP64 oracle budgets,
repeat/rewind/poisoned-prefix checks, guards, and immutable input/cache checks.
Short timing-off memcheck, racecheck, and synccheck passed with zero reported
errors or hazards. Full timing-off memcheck and synccheck passed. The short
racecheck is bounded coverage, not a full-context racecheck claim. Combined
staged model memcheck passed on a two-token prompt plus two decode steps.

At both 106- and 512-token inputs, staged alone and combined staged
`qwen-stream-check` matched the saved native control's legacy/stream logits,
selected tokens, and all state hashes at every checked step. The balanced
256-output runs generated the same complete token sequence in every mode and
repetition. This is exact agreement with this runtime's native control, not
proof of bit-identical Ninfer arithmetic or comprehensive model quality.
The separate native source assessment measured mean NLL 1.6207356816 versus
Ninfer BF16-KV 1.6207768509 on 49,148 target positions; this trial did not
repeat that quality evaluation.

## Whole-model result

`MESH_SPECIALIZE_EXECUTION=stream`, fixed 256 generated tokens, two repetitions
per run, forward then reverse mode order. The table reports the median of four
decode rates in tokens/s. The 106-token story and 512-token pg19 inputs use the
same token-ID fixtures and artifact in all modes.

| Schedule | 106 inputs | 512 inputs |
| --- | ---: | ---: |
| Exact baseline | 26.47 | 18.17 |
| Staged attention alone | 31.02 | 25.01 |
| FP8 vector16 + paired FP64 A/B + NVFP4 PRMT | 31.84 | 20.55 |
| Three schedules + staged attention | 38.66 | 29.75 |
| Ninfer MTP0, BF16 KV, separate shared-GPU reference | 76.3 | 76.3 |

The best tested combination is about 51%/39% of Ninfer's ordinary-decode
rate, respectively. It is an exclusive-GPU engineering comparison against a
separate shared-GPU Ninfer run, not a synchronized serving benchmark. No MTP4
comparison can be inferred from ordinary decode; Ninfer's separate MTP4 FP8-KV
reference is 141.2/213.5 tokens/s at these inputs.

## Remaining cost

`staged-profile-1/` records a single profiled legacy decode step with the same
combined schedules and hashes. At 107/513 past tokens, 1,461 synchronized
kernel-event launches totaled 25.93/33.92 ms. At 513 past tokens the staged
coefficient stage consumed 7.97 ms over 16 launches, the staged values stage
1.66 ms, FP8 vector16 projections 8.30 ms over 233 launches, NVFP4 PRMT
5.97 ms over 168, FP8 activation quantization 2.62 ms over 233, and GDN
recurrence 2.50 ms over 48. The profile uses per-launch synchronization and
does not measure stream throughput or establish a host-overhead percentage.
Coefficient scaling and FP8 quantizer reuse are subsequent candidates, not
included in these rates. Long-context chunked execution, graph capture, native
MTP, and any new profile combination need their own correctness and timing runs.

The durable rule is to retain the exact baseline and raw evidence, require
whole-model state/logit equivalence before timing, and label every profile and
sanitizer scope. Operator speed alone does not close the Ninfer gap.
