# Resident MTP continuation

Status: implementation in progress after dedicated decode and larger prefill
qualification. Retaining MTP tensors does not count as implemented speculation.

The target path needs a captured raw final hidden row, all-row verification logits,
and independent device-state copies. Cursor copies must reject poisoned sessions;
verification writes only a fork until its accepted prefix is committed. A rejection
must replay the accepted input prefix from the untouched base session because GDN
recurrent state cannot be repaired by shortening a KV cursor.

The existing BF16 projection, normalization, attention, activation and residual
kernels can execute the 15 retained MTP tensors. Projection precision moves to its
own module because both the attention and MLP paths now select BF16/FP8/NVFP4
implementations. Existing text-layer precision and arithmetic order stay unchanged.
Pinned upstream inspection establishes the input contract: post-target-final-norm
hidden, separately normalized shared embeddings and hidden, embedding first in
the concatenation, BF16 fusion, one full-attention block, then MTP output norm.
Prefill pairs hidden rows with shifted IDs `[x1, ..., target_next]` at unchanged
positions. Later draft steps consume the preceding post-MTP-norm hidden.
Teacher-forced extension after verification uses target hidden rows and accepted
next-token predictions. There is no extra RoPE position offset.

Sources: [vLLM MTP](https://github.com/vllm-project/vllm/blob/e7900156e130c9880eb03b7c1f2df32820e7a2be/vllm/model_executor/models/qwen3_5_mtp.py),
[SGLang MTP](https://github.com/sgl-project/sglang/blob/b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce/python/sglang/srt/models/qwen3_5_mtp.py),
[SGLang draft extension](https://github.com/sgl-project/sglang/blob/b252aceffecd1e313cb5a03d3cbf56c99fc8c9ce/python/sglang/srt/speculative/eagle_worker_v2.py).

The first implementation forks full target state for verification. On rejection,
it replays the accepted input prefix from the untouched base. This is deliberately
measurable overhead, with replay/accepted/drafted/verification counters; it must
not be mistaken for an optimized rollback implementation. Exact greedy token and
whole target-state comparison against ordinary decode is required.

Three template-rendered prompts were run on retained source `6a6d7222d47c322a2e960c17be49ee7a534967e2`
before MTP integration. Sixteen output tokens give sensible counting, Python and
explanation starts at 24.39–24.65 decode tokens/s. This is a smoke test, not a
quality evaluation. The fixture records the pinned tokenizer/template hashes and
uses external development-only tokenizers 0.22.2 and Jinja 3.1.6; neither is an
engine dependency.

Required closure: independent real-weight MTP-head comparison; draft and target
state isolation; forced rejection and rollback; emitted tokens and target state
matching target-only greedy execution; acceptance counters and repeated measured
end-to-end timing. First scope is greedy, single sequence, bounded draft depth.
Real-text fixtures use the pinned tokenizer and chat template with thinking off,
not synthetic ID repetition alone. No serving or stochastic-sampling claim follows.

Initial implementation uses `qwen-mtp-reference` to compose a CPU MTP fixture from
an identity-matched target reference, and `qwen-mtp-check` to validate the resident
head, whole/token MTP partitioning, forced rejection and target-only equivalence.
Qualification requires 3–128 outputs so at least one draft can be rejected.
The runtime loop itself permits two outputs. Host checks: 213 macOS library tests,
Clippy and no-console check pass; Linux/GPU qualification still pending.
No new PTX instructions are introduced in this implementation step.

First GPU run at `8b8659da985732492a206d8ffd51f2a26cb6f368` passes the
independent head's unchanged 1% L2 / 0.9999 cosine budgets and greedy token check,
exact MTP whole/token partitioning, target-only token/state equality and forced
rejection replay. The independent CPU/GPU head is not bit exact: 434 hidden values
and 94,159 logits differ, with maximum absolute differences 0.125 / 0.0625.
Five of eight drafts were accepted on the synthetic eight-output fixture;
23.36 tokens/s is slower than its 25.45 target-only control. Full text/sanitizer
qualification is in progress. Added phase wall attribution and separate head /
verification event captures to select the next change from measured costs.
