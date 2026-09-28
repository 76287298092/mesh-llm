# Teacher-forced arithmetic reassessment

Status: measured September 28, 2026 on Carrack GPU0. Both existing arithmetic
candidates fail the predeclared agreement gates; neither is promoted. These
failures do not imply worse average predictive likelihood on this corpus.

## Protocol and identity

Source `ab33f730e1c33277c95765ca41a46bdb5ecdc805`, PTX SHA-256
`04f03b9b0e7124fce67d878c917c38693a0d9dd0b073cbbe7e04536dd1665f1d`.
Model SHA-256 `f2982d23cbc7f795f6607a63b89ed3b104009959e789b415db6a2dd36035444e`.
GPU UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`; manifests report CUDA
Driver API version 13040. Ninfer and ComfyUI were stopped and restored to their
initial active states. No other compute process appeared in the exclusive
snapshot; process samples remain in the raw trial directories.

Corpus: four independently windowed 12,288-token streams from
`ninfer-ppl-1m-v1`, with context 512 and stride 256. Each stream scores 12,287
targets, 49,148 total. Fresh state per window; no prefix reuse or greedy-input
divergence. The two ablations change only FP8 prefill arithmetic or attention
arithmetic, respectively. NVFP4 and all other profiles remain the default.
The scoring head stays exact for native-prefill by using chunks below its
native dispatch threshold. Scoring time is not inference throughput.

## Results against the exact control

| Profile | Mean NLL | Relative NLL change | Top-1 agreement | Mean partition KL | p99.9 partition KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact control | 1.62150638 | reference | reference | reference | reference |
| native FP8 prefill | 1.62035776 | −0.07084% | 93.9245% | 0.019601 | 0.839131 |
| online BF16 attention | 1.62155783 | +0.00317% | 94.0059% | 0.019285 | 0.816831 |

Both pass the overall and per-domain NLL budgets. Both fail top-1 agreement
(required 98%), mean KL for Chinese and English reference text (required
≤0.02), and p99.9 KL for Chinese text (required ≤1.0). For native FP8, Chinese
mean/p99.9 KL are 0.026971/1.287571; for online attention, 0.026916/1.514958.
The original gate thresholds are unchanged.

KL is measured on shared top-64 IDs plus one remaining-mass bucket. It is a
lower bound on full-vocabulary KL, not proof that the unseen tail agrees.
The union-support estimate is retained as a diagnostic only. Compact record
repeatability and full-logit determinism were not run in this campaign.
Generation-task quality and long-context quality are not certified by NLL.

## Scorer validation and Ninfer comparison

Each run's built-in scorer checks pass: four sampled rows against the independent
FP64 softmax/sort oracle, and two sampled batched-head rows against the one-row
head. Maximum sampled score error is ≤4.41e−7; sampled head logits match exactly.
This is sampled operator coverage, not a full independent reference for every
scored row. The separate bounded model/scorer memcheck passes; broader sanitizer
results are recorded with the stream trials.

The older Ninfer perplexity executable scored the same windows/counts with BF16
KV at mean NLL 1.620777 (PPL 5.0570). Our exact control is about 0.045% higher
in NLL. All four streams' window boundaries and counts match. Ninfer's artifact
hash matches the published unsloth-derived manifest, but canonical logical
weight equality and this scorer binary's source revision remain unverified.
A later extracted-asset check confirms identical token IDs for all four corpus
slices and identical rendered performance prompts; no server-internal token
buffer was captured. This comparison cannot isolate runtime arithmetic.

## Decision and next evidence

Keep exact as the default. The fast profiles materially change next-token
rankings, while these data do not show a substantial NLL regression. Do not
reinterpret a greedy change as semantic failure, or silently lower the
agreement gate. Any future promotion needs the remaining determinism/task gates
and an explicit decision about the intended quality contract if the fixed
agreement criteria remain unmet.

Evidence: `../evidence/reassess-20260928/stream-trial-2/` and
`../evidence/reassess-20260928/quality-ablation-1/`. Compact manifests and
comparisons are committed; `raw-sha256.txt` identifies the retained binary score
records and raw logs under `target/specialize/reassess-20260927/` on both hosts.
Updated comparator tests reject invalid distributions and mismatched protocols.
