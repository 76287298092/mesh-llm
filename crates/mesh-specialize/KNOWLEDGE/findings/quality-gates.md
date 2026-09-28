# Quality gates for non-exact execution profiles

Status: policy fixed on 2026-09-27, before any fast-profile scoring result
exists. Thresholds below must not be loosened after seeing results. A change
that fails keeps its evidence and stays opt-in; a new gate needs a new dated
entry that explains why the old one was wrong, without citing the failing run
as the reason.

## Why bit identity is not the only gate

The exact profile already quantizes activations: FP8 projections use per-row
FP8 codes and scales, and NVFP4 projections use packed E2M1 groups with FP8
scales. "Exact" means the integer-expanded dot products and reconstruction are
reproduced bit for bit by an independent CPU oracle. Native tensor-core
floating-point accumulation, online softmax and chunked recurrence change
summation order and intermediate rounding. They cannot reproduce those bits,
and neither does Ninfer. The exact profile stays as the control and keeps all
of its existing bitwise checks.

## Corpus and protocol

- Text: `ninfer-ppl-1m-v1-slice12k`, the first 12,288 tokens (artifact
  tokenizer, no special tokens) of the Ninfer perplexity corpus streams
  `wikitext-00`, `pg19-00`, `ninfer-00` and `zhwiki-00`. Built by
  `target/specialize/reassess-20260927/make_ninfer_plans.py`; token counts in
  `ppl-slice/token-counts.json`.
- Windows: context 512, stride 256 (Ninfer `plan_windows` semantics: the first
  window scores `[1, 512)`, later windows score only their last 256 targets).
  Once chunked prefill exists, the same gates repeat at context 4,096 and
  stride 2,048.
- Every scored position records the target log-probability, top-1 token and
  the top-64 log-probabilities of both profiles' full-vocabulary softmax.
- KL uses the exact profile as P. Clarified 2026-09-28, before any scoring run
  existed: records hold only each profile's own top 64, so a union id absent
  from one list has no known probability there. The gated KL is therefore
  computed on a common partition: every id present in both top-64 lists is its
  own cell, and one bucket holds all other ids, with each profile's bucket mass
  equal to one minus its mass on the shared ids. By the data-processing
  inequality this is a lower bound on the full-vocabulary KL. The comparator
  also reports an estimate that assigns absent union ids
  `min(p64, tail/(n+1))`; that estimate is informational, not gated.

## Internal gate: fast profile against the exact control, same weights

All must hold, overall and per domain unless stated.

| Metric | Threshold |
| --- | --- |
| Mean NLL increase, overall | at most 0.5% relative |
| Mean NLL increase, each domain | at most 1.0% relative |
| Top-1 agreement over all scored positions | at least 98.0% |
| Mean KL(exact ‖ fast) | at most 0.02 nats |
| 99.9th-percentile KL | at most 1.0 nat |
| Same-profile determinism | identical inputs give identical logits |

Rationale: per-row FP8 activation quantization is typically reported at a few
tenths of a percent of perplexity, and pure accumulation-order changes far
below that. These thresholds allow a numerical change of that size and reject
anything approaching NVFP4 activation-quantization damage (percent-level).
A lower NLL than the control passes the NLL rows but not the distribution rows.

## External comparison: Ninfer on the same text

Ninfer scores the same slice with `ninfer-perplexity` at matched context,
stride and KV type. Its weights come from a different NVFP4 checkpoint
(NVIDIA ModelOpt via a local converter, versus our unsloth-derived import), so
the comparison mixes weights and runtime. It is reported, not used as a
pass/fail gate for arithmetic, alongside the exact control's gap to Ninfer.
Scored token counts must match per stream, or the tokenization differs and the
comparison is invalid.

## Generation checks

Generated continuations of the matched prompts are retained for both
profiles. Greedy divergence alone is neither a failure nor a pass. Output that
degenerates (for example repeated n-gram loops absent from the control) fails
regardless of the metrics above.

## Coverage limits

Perplexity and teacher-forced agreement do not measure reasoning or long
answers. A small checkable task set remains a follow-up before any default
change is described as quality-equivalent for serving.
