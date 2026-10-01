# Split-decode quality failure

Status: execution verified; arithmetic quality verdict **FAIL**. The candidate
stays opt-in. Exact attention remains the control and default; no thresholds or
profiles were promoted.

## Trial and identity

Evidence is retained under
[`split-decode-quality-20260930-a`](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/).
The [scope](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/scope.txt)
identifies an uncommitted snapshot at
`/home/ndizazzo/dev/mesh/issue1393-q4-snapshot-20260930`; source provenance is
parent-owned, not a newly qualified commit.

Pinned SHA-256 identities:

| Item | SHA-256 |
| --- | --- |
| Executable `target/release/xtask` | `4682e811bd081b05a4908bbe2073c44a33ba90090a552a67465f0256afff1256` |
| `probes.ptx` | `3727e477d0691a3359e87de0ecdffc3248883890432958cb247c2db33c32f997` |
| Current resident source manifest, parent-provided | `84377dcfb93ee50618ff1e3bd2daa8128d0b1ee64d7bf22f9135167178ae2efa` |
| Qwen3.8 27B `.ninfer` artifact | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |

The source is unchanged from the residency trial, per parent provenance. Executable
and PTX pins are in [hashes.txt](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/hashes.txt).
The model is `qwen3.8-27b:text:ninfer-v3-control-v1`, loaded from
`/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`. The manifests identify device 0
as an RTX 5090, compute capability 12.0, driver version 13040, UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`. Clocks and compiler/toolchain version
are not measured by these manifests.

## Recipe and bounded scope

Read the [control manifest](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/decode-score-control/manifest.json),
[candidate manifest](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/decode-score-candidate/manifest.json),
and [repeat manifest](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/decode-score-repeat/manifest.json).
All use explicit `MESH_SPECIALIZE_SCORE_MODE=decode`,
`MESH_SPECIALIZE_SCORE_FORWARD_ROWS=1`, and
`MESH_SPECIALIZE_SCORE_LOGITS_HASH=on`, with context 128 and stride 64.
Control uses `MESH_SPECIALIZE_ATTENTION_PROFILE=exact`; candidate and repeat use
`MESH_SPECIALIZE_ATTENTION_PROFILE=split-decode`. Execution is legacy. The
attention profiles are `bf16-fp64-v1` and
`bf16-split-decode-fp32-exact-prefill-v1`, respectively. Both retain
`exact-a8-v1` FP8 and `nvfp4-native-prefill-integer-decode-v1`, with no FP8 split-K
or MLP workspace.

This is a bounded four-stream teacher-forced decode smoke,
`decode-quality-smoke-257`: each stream has 257 input tokens and 256 scored
targets, for 1,024 scored targets total. Domains are `chinese_reference`,
`english_long_form`, `english_reference`, and `ninfer_code`. Each window starts
with fresh zeroed resident state and cursor; state does not carry across windows
or streams. Context prefill and subsequent one-row decode are explicit in each
manifest's window ranges. This is not a whole-12K quality run or a generation
test. Generation checks are **NOT RUN**. No MTP execution or MTP quality claim
follows from this trial, and scoring time is not a model throughput benchmark.

The retained [runs.log](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/runs.log)
records control, candidate, repeat, comparison, and assertion completion with
exit 0. The bounded recipe specifies 1,800 seconds per run, `MemoryMax=32G`,
and `MemorySwapMax=0`. A literal launch command is not recorded in `scope.txt`;
the environment, profiles, model, and scoring parameters above record the recipe
without inventing a command.

## Unchanged gate results

The authoritative [split-decode-gates.json](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/split-decode-gates.json)
has verdict `FAIL`. Its fixed thresholds remain NLL relative increase at most
0.005 overall and 0.01 per domain, Top-1 agreement at least 0.98, mean KL at most
0.02, and 99.9th percentile KL at most 1.0.

| Gate or measurement | Observed | Required | Result |
| --- | --- | --- | --- |
| Overall Top-1 agreement | 0.921875 | >= 0.98 | FAIL |
| Mean KL, `chinese_reference` | 0.022695166390312375 | <= 0.02 | FAIL |
| Mean KL, `english_long_form` | 0.020892322533617766 | <= 0.02 | FAIL |
| Mean KL, overall | 0.017492985252349384 | <= 0.02 | PASS for recorded lower-bound metric only |
| Mean KL, `english_reference` | 0.014333015117460705 | <= 0.02 | PASS for recorded lower-bound metric only |
| Mean KL, `ninfer_code` | 0.012051436968006693 | <= 0.02 | PASS for recorded lower-bound metric only |
| Overall 99.9th percentile KL | 0.4117509911762525 | <= 1.0 | PASS for recorded lower-bound metric only |
| Candidate/repeat score-record repeatability | true | true | PASS |
| Candidate/repeat full-logit determinism | true | true | PASS |

Overall mean NLL improves from `2.1134277532640597` to
`2.1021163471501167`; relative increase is `-0.0053521612444396155`.
Overall and all four domain NLL gates pass. All domain 99.9th percentile KL gates
also pass. Lower NLL does not override the failed distribution-agreement gates.

As documented in [quality gates](quality-gates.md), the common-partition KL is a
lower bound on full-vocabulary KL, not full KL. In particular, the overall
`0.017492985252349384` does not prove full KL is below 0.02. A failed lower-bound
gate remains a failure. Full-logit hashes cover every little-endian BF16
vocabulary value at scored positions in window order; matching candidate/repeat
hashes prove determinism for those positions, not equality with the exact control
or distribution quality.

The [assertion log](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/decode-assert.log)
preserves the distinction exactly:

```text
EXECUTION-VERIFIED; arithmetic-verdict=FAIL; bounded four-stream decode smoke; not whole-12K quality; not promotion; generation checks NOT RUN
```

Its exit 0 verifies that the expected failed verdict was recorded. Individual
manifests' scorer-check passes and `quality_gate_status: NOT RUN` do not replace
the separate comparison's `FAIL` verdict.

## Restoration and durable rule

[units-after.txt](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/units-after.txt)
records both `ninfer-qwen38.service` and `battlecity-comfy.service` as
`initial=active after=active`. The retained
[ninfer-health-restored.txt](../evidence/split-decode-quality-20260930/split-decode-quality-20260930-a/ninfer-health-restored.txt)
is an empty health marker, not an HTTP response transcript. The parent separately
observed restored Ninfer HTTP 200. No service operations were performed for this
documentation update.

Keep split-decode opt-in after this failed real-decode candidate comparison.
Successful execution, deterministic logits, and improved NLL cannot substitute
for passing the unchanged distribution gates. Whole-corpus and generation
qualification remain unperformed by this bounded trial.
