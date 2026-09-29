# Combined exact schedules at long context

Source `48cbc42e74bdf99ffe3c901adf91fdc1c9f62bbd`, PTX SHA-256
`e482363228aa8eaedb146d12d5263edd4aabab6245e93d252dad0888f7eb80b9`.
Evidence: [combined-long-context-1](../evidence/reassess-20260928/combined-long-context-1/).

The baseline uses serial staged FP64 attention, vector16 FP8 decode, paired-FP64
A/B projections, and PRMT NVFP4 decode. The candidate adds prefix-parallel-v2
attention and reuse-input FP8 quantization. The reports' `stream_forward` fields
record these schedules; the top-level profile label alone does not distinguish them.

Each mode ran twice in forward/reverse order, with two repetitions per run,
512-row prefill chunks and 32 generated tokens. Median decode rates across the
four repetitions on Carrack GPU 0, RTX 5090:

| Input tokens | Baseline tok/s | Combined tok/s |
| --- | --- | --- |
| 2048, wiki-2k | 16.4595 | 29.8945 |
| 8192, pg19-8k | 5.6337 | 14.0600 |

All eight benchmark invocations completed. All eight generated sequences per
prompt contained 32 tokens and were identical across modes and rounds.
The original post-run comparison failed because it requested the absent
`fixed_output_tokens` field. The actual field is `output_tokens`. A separate
offline check of all saved reports passed completion, output count, repetition
count, partition status and cross-mode token equality. The original failure log
is retained; the entire script is not reported as having passed.

For prompts longer than 512, `partition_check.status` is `not_run`. Matching
greedy tokens does not establish full-logit or recurrent-state equivalence at
these lengths. There is no matched Ninfer long-context comparison, native MTP
measurement, or nonexact A16 qualification in this trial.

Both initially active services were restored after the run; Ninfer health was
HTTP 200. The GPU benchmark commands all exited zero before the offline
comparison error.
