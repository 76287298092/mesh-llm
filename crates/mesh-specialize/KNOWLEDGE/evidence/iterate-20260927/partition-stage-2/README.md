# Layer22 stage localization: NVFP4 down projection

Sourceff387d8ce, same features-greedy PTX and128-token raw fixture. FP8exact,
attentiononline, MLPworkspaceon, split-Koff, GPUgreedyoff. RTX5090,driver615.71.09.

Across whole-prefix and one-token submissions, every layer22 captured BF16
stage agrees at all128rows through `mlp_activation`: normalized input, QKV/Z/A/B,
convolution, gated recurrence, output projection, residual, post-norm, MLPgate
andup, and MLPactivation. The first differing stage is `mlp_down`, row101.
The final hidden output differs only at that row.

Source dispatch uses native FP32 NVFP4 MMA for multirow down projections and
integer NVFP4 reduction for one row. This localizes the observed mismatch to
that projection with identical BF16 input. The capture did not separately
compare quantized codes/scales or raw FP32 projection outputs; an identical
quantized-input GPU/native/integer/independent-CPU audit is the next test.
Do not claim the exact numerical cause or difference magnitude before that audit.

Although stage observation uses ordinary allocation for layer22's MLP, whole
and token final-state hashes exactly match partition-audit-1. Profile/control
outputs and complete state remain exact. Memory release passed in this run.
The overall strict partition result remains FAILED; no gate was relaxed.

Ninfer stayed inactive and ComfyUI remained resident. Separate mesh-llm processes
were observed on the shared host and left untouched. This is diagnostic evidence,
not throughput, semantic quality or promotion evidence.
