# NVFP4 model scheduling comparison

Source e68e93d95, RTX5090, driver615.71.09. Exact FP8 and attention, MLP workspace
on, GPU greedy and FP8 split-K off. Eight generated tokens, three repetitions.
Baseline and original tiled use PTX e8beed4f...; fixed producers use83f52bb4....
Full hashes are in the reports. The same executable runs each retained PTX.

| Input tokens | Baseline prefill tokens/s | Original tiled | Fixed producers |
| --- | ---: | ---: | ---: |
| 128 | 277.41424 | 215.60535 | 278.19249 |
| 512 | 305.58046 | 233.38908 | 306.60139 |

Original tiled regresses at both sizes. Fixed producers recover to baseline, but
0.28% and0.33% differences do not establish a useful speedup. Shared GPU with
ComfyUI498MiB, fixed order, three repetitions, no matched Ninfer claim. Both
candidates remain explicit experiments; the default is unchanged.

All six model profile reports pass strict within-profile whole/token partition,
profile/control logits and complete state, and memory release. Across schedules,
all four captured BF16 logit files at each input size match baseline hashes,
complete same-input state matches, and generated tokens match across every
repetition. Comparison code verifies dump hashes, metadata and gates before
asserting equivalence. This supports the tested scheduling change, not universal
numerical proof, long-context or serving readiness. Full-model sanitizers remain
pending. Both versions have separate19-case operator and three-sanitizer evidence.

Single-run instrumented NVFP4 prefill event sums, baseline/original/fixed:
128 inputs56.781/186.274/52.148ms;512 inputs195.920/719.774/191.334ms.
At512 inputs the baseline event sums rank exact FP8 prefill723.972ms, causal
attention243.719ms, GDN recurrence237.565ms and NVFP4 linear195.920ms. These
instrumented values identify costs; do not subtract them from uninstrumented
wall time to infer CPU overhead, or treat them as separate model throughputs.

Ninfer remained inactive and ComfyUI was preserved. Trial finished with no trial
GPU process remaining. Complete BF16 dumps remain in the matching ignored
trial directory on both hosts; committed manifests contain their hashes.

Next experiment: a separate wider NVFP4 tile with multiple output fragments per
warp, reusing A fragments across them. Ninfer's pinned source uses this technique
in nvfp4_a4_mma.cuh:191-275 and nvfp4_a4_tma.cuh:250-309 at e31bc99b13.
A larger runtime gain also requires addressing the measured FP8, attention and
GDN costs while retaining independent arithmetic and meaningful quality checks.
