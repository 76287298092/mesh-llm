# A16 head-only model comparison

Source 2e5f77d2f, RTX5090, driver615.71.09, retained head-pipeline PTX.
Exact decoder arithmetic, BF16 exact attention, MLP workspace enabled, GPU greedy
off, split-K off. Python and prose prompts, 32 generated tokens, three repetitions.
Fixed profile order, shared GPU with ComfyUI498MiB, not a matched Ninfer comparison.
All six profile reports pass strict within-profile output/state and whole/token
partition checks. Same teacher token gives exactly equal complete state across
all three profiles. This does not assert equal logits across arithmetic profiles.

| Prompt | Exact decode tokens/s | A16 GEMV head | A16 MMA head |
| --- | ---: | ---: | ---: |
| Python | 25.53780 | 25.61944 | 25.40962 |
| Prose | 25.50770 | 25.67297 | 25.49304 |

No useful ordinary-decode improvement is established. Both A16 schedules produce
identical short token sequences; Python matches exact, prose differs from exact.
Greedy winners agree at the two measured teacher-forced positions per prompt.
Largest exact-to-A16 KL is0.00210307 and TV0.0270414 on prose prefill. A16-GEMV
versus A16-MMA distribution differences are tiny at these positions. A tiny
negative KL in one comparison is floating-point evaluation noise near zero and
is retained in the raw metrics, not interpreted as negative divergence.
The 32-token outputs are incomplete answers, not functional or factual quality
qualification. No broad quality promotion follows from these four logit positions.

A single diagnostic Python decode event measures the head kernel at0.87216ms
exact,0.79155ms A16 GEMV,1.03398ms A16 MMA. These are instrumented single-launch
samples, separate from uninstrumented model medians; do not infer host overhead
by subtraction. Small-batch head throughput remains unmeasured. Keep the new
schedule experimental and move ordinary-decode effort elsewhere.

Full BF16 logit files remain in the identically named ignored trial directories
on local and Carrack hosts; retained manifests hash them. The comparison script
checks hashes, metadata, complete state and profile gates before computing metrics.
Copied build/test logs omit blank lines at EOF. Original logs remain ignored.
Ninfer was inactive before/after; ComfyUI was preserved. No GPU job remains from
this completed trial. Full-model sanitizer and sampled real-weight CPU head audit
are still pending; the separate 37-case operator suite passed all three sanitizers.
