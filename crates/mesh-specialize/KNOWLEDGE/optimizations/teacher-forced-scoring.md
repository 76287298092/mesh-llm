# Teacher-forced scoring

Status: first full-corpus measurement at `ab33f730e`, September 28. Four sampled
rows pass the independent FP64 scorer oracle and sampled batched heads agree
with the one-row head. The exact, native-prefill and online-attention runs each
scored 49,148 targets. Both candidates fail the fixed agreement gates; see
[arithmetic reassessment](../findings/arithmetic-quality-20260928.md). Bounded
model/scorer memcheck passes. Racecheck initially hit its memory cap; the smaller bounded retry and
synccheck pass (see the stream entry for the serialization caveat). No arithmetic default changed.

## What it does

`xtask specialize qwen-model-score` scores fixed token streams with
fixed-window, truncated-context causal perplexity semantics and records, for
every scored position, the target log-probability, the log-sum-exp and the
top-64 `(id, logprob)` pairs of the full-vocabulary softmax. Two runs under
different arithmetic profiles (same artifact) are compared by
`validation/scripts/compare_scores.py`; one run can also be compared with a
Ninfer `ninfer-perplexity` `report.json`.

```text
xtask specialize qwen-model-score --artifact P --streams STREAMS_JSON \
  --context C --stride S --ptx P --device N --output NEW_DIR
```

STREAMS_JSON is `{"corpus_id": ..., "streams": [{"id", "domain", "tokens": [...]}]}`,
written by `validation/scripts/tokenize_streams.py --manifest
target/specialize/reassess-20260927/ppl-slice/manifest.json --tokenizer
tokenizer.json --expect-counts .../token-counts.json --output streams.json`
(`add_special_tokens=False`). Stream ids must match `[A-Za-z0-9._-]`. The
output directory must not exist.

## Windows and state

`src/engine/teacher_scoring.rs::plan_windows` reimplements the Ninfer
semantics from `docs/perplexity.md` and `plan_windows` (read as a spec, no code
imported): the first window is `[0, min(C, N))` and scores targets
`[1, min(C, N))`; each later window ends `S` targets later, begins at
`max(0, end - C)`, and scores only `[previous_end, end)`. Unit tests reproduce
47 windows and 12,287 scored targets for 12,288 tokens at C=512/S=256, and
short-tail, single-window, exact-context and small-stride cases.

Every window allocates a new zeroed `ResidentState` and cursor
(`Session::new`) and drops it after the forward, so no KV, convolution or GDN
state carries between windows or streams. The forward is the ordinary
`forward_detailed(LogitsSelection::Last)` with unchanged arithmetic; the new
`Model::forward_hidden` returns its final hidden rows (before the final norm)
and discards the last-row logits. Target `t` is predicted by local row
`t - 1 - input_begin`.

## Head and statistics

`src/kernels/cuda/resident_score.rs` copies chunks of hidden rows and runs the
model's own `Head::run_all` (final RMSNorm plus FP8 head under the active
`MESH_SPECIALIZE_FP8_PROFILE` dispatch). Chunk rows are chosen so every chunk
uses the same head kernel family as the one-row generation head: 128 for
`exact` (BF16 plus FP32 head output about 191 MB), 1 for `a16-decode` and
`a16-head-gemv`, 8 for `a16-head`, 15 for native-prefill profiles (below the
16-row native FP8 threshold, so the head stays exact as in generation).

`kernels/nvptx/row_logprob_topk.rs::row_logprob_topk_bf16` (grid rows, block
256, 11,840 B static shared) computes per row: maximum, log-sum-exp, target
log-probability and top-64. `exp(x - max)` is an FP32 Rust polynomial
(`kernels/nvptx/logprob_math.rs`, emulated worst error 0.71 ulp on the tested
grid), accumulated in FP64 in fixed per-thread order plus a fixed shared tree;
`ln` is an FP64 series; log-probabilities are `(x - max) - ln(sum)` in FP64,
rounded once to FP32. Top-64 uses a two-level 8-bit radix histogram on the
ordered BF16 key, admits threshold ties in id order via a contiguous
per-thread prefix count, then sorts by (key descending, id ascending). Output
is deterministic run to run; atomics only change counts and pre-sort slots.
Any nonfinite logit, bad target or fewer than 64 finite logits sets a status
word and fails the run.

Independent evidence: `reference/row_logprob_topk.rs` (host FP64 `exp`/`ln`,
full sort) with hand fixtures for ties, zero signs, finite extremes and a
target outside the top 64. On the first window's first chunk the harness
compares the first four device rows against it (ids exact, values within
`1e-5 + 4e-7 |v|`) and compares chunked head logits with the one-row
generation head (`Head::run`) on the chunk's first and last rows. The manifest
`check` object records both; `all_passed` requires both.

## Record format

Per stream `<id>.scores.bin`, one 524-byte little-endian record per scored
position in stream order:

| Offset | Field | Type |
| --: | --- | --- |
| 0 | target id | u32 |
| 4 | target logprob | f32 |
| 8 | logsumexp | f32 |
| 12 | top-64 ids, logit descending, lower id first on ties | u32[64] |
| 268 | top-64 logprobs | f32[64] |

`manifest.json` holds corpus id, context, stride, per-stream and per-window
scored counts and total/mean NLL (Ninfer report field names), per-domain and
overall aggregates, FP8/NVFP4/attention profile names, every
`MESH_SPECIALIZE_*` variable, head chunk rows, the check object, device,
timing, artifact identity, artifact file sha256 and PTX sha256.

## Comparator

`compare_scores.py --control EXACT_DIR --candidate FAST_DIR [--repeat FAST_DIR2]
[--ninfer report.json] --output NEW.json` (stdlib only) evaluates every
internal gate row: NLL relative increase overall (0.5%) and per domain (1.0%),
top-1 agreement overall (98.0%), mean KL (0.02) and nearest-rank 99.9th
percentile KL (1.0) overall and per domain, and byte-identical `--repeat`
records for score repeatability (NOT RUN without it). Full-logit determinism
requires separate full-vocabulary hashes; compact-record equality is not enough. The
comparison against Ninfer checks context/stride, per-stream scored counts and
per-window bounds (INVALID on mismatch) and reports NLL differences; it is not
a gate.

The gated KL is the common top-64 intersection plus one remaining-mass bucket,
a lower bound on full-vocabulary KL. The union-with-imputed-missing-mass estimate
is informational only. Neither is a full-vocabulary distribution comparison.
The thresholds in the quality policy are unchanged. The comparator now rejects
mismatched artifact identity, protocols, domains and malformed probabilities;
12 stdlib tests cover these contracts and the lower-bound limitation.

Optional full-logit evidence: `MESH_SPECIALIZE_SCORE_LOGITS_HASH=on` downloads
and SHA-256 hashes every BF16 vocabulary value at each scored position, in
window order. One digest per stream is stored as `full_logits_sha256`, together
with an input-token digest covering even unscored context. It adds diagnostic
transfer cost and is off by default. The first campaign predates these fields;
it cannot retrospectively establish full-logit determinism.

## Limits

- Context at most 512 (one forward per window); 4,096/2,048 needs chunked
  prefill. Enforced in the xtask, package and harness.
- The last-row head and a 496,640 B logit download per window are wasted work
  kept to avoid touching the forward's execution path.
- Records alone carry derived statistics. Full-logit determinism is NOT RUN
  unless both repeated runs explicitly capture all-vocabulary hashes.
- Timing covers forward, head, statistics and host transfers; it is not a
  prefill benchmark.

## Assembly sites

`kernels/nvptx/row_logprob_topk.rs`: `%tid.x`/`%ctaid.x` reads; one function
scope `.shared` declaration with `cvta.shared.u64`; `bar.sync 0`; generic
`atom.add.u32` into the CTA scratch. `logprob_math.rs` has none. Inventory rows
are pending qualification.
