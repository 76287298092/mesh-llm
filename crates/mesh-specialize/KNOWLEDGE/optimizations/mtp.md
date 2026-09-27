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

The initial eight-output/depth-four memcheck completed with zero errors. Racecheck
exceeded its 300-second bound and has no terminal result; preserve that incomplete
run under `evidence/iterate-20260927/mtp-qualified/` despite the original directory
name. Ninfer was restored at 14:49:53 EDT, PID 3329921, HTTP 200; ComfyUI unchanged.
Subsequent sanitizer runs use three outputs/depth one to exercise target verification,
forced rejection and replay within the bound. Full-depth text comparisons remain
separate. Instrumented memory snapshots retained 2 MiB after release; uninstrumented
runs returned to their initial free-memory value. No transient peak is measured.

The next isolated candidate selects the exact 16x8 FP8 tensor-core tile for all
multirow work, including two-to-five-row verification, while single-row decode
keeps its dedicated path. Source `65ab2242c` measured 204 ms verification out of
300 ms total synthetic MTP decode; the separate five-row capture spent 90.38 ms
on kernels, dominated by `fp8_linear_exact4`. MTP-head kernels took only 5.21 ms
for the two-row oracle. Existing FP8 fixtures already exercise the larger tile's
partial rows; the candidate still needs full-model and MTP regression evidence.

Uninstrumented real-text baseline at `65ab2242c` (32 outputs, three samples):

| Prompt | Target-only decode | MTP depth 1 | MTP depth 4 | Depth-4 accepted drafts |
| --- | ---: | ---: | ---: | ---: |
| Counting, 24 inputs | 24.53 | 30.56 | 31.34 | 92.3% |
| Python, 32 inputs | 24.36 | 30.42 | 38.21 | 100% |
| Explanation, 31 inputs | 24.35 | 25.33 | 12.95 | 32.1% |

Rates are median decode tokens/s for the speculative runs. Target controls are
measured in the same process. Every result preserves all greedy output tokens and
the entire final target state, including forced rejection. At depth one the prose
case is still slightly slower end to end because of MTP prefill. Fixed depth four
is not a safe default: low acceptance makes target replay expensive. Preserve this
negative result, and distinguish a bounded working MTP engine from a serving policy.

Rejected candidate `c51fe9e66`: selecting the existing 16-row tile for all multirow
work passes the independent model and MTP checks but reduces synthetic MTP decode
from 23.37 to 20.71 tokens/s. Two-row prefill/replay rises from 55.7 to 94.3 ms;
five-row verification remains about 204 ms across two rounds. Revert that dispatch.
The next candidate transposes the exact integer MMA operands: 16 output channels
by eight input rows, with transposed stores and unchanged final scale order. This
halves the padding of small verification batches without changing arithmetic.

The transposed verification candidate initially selects four-to-fifteen rows only;
one-to-three-row dispatch stays at the qualified baseline. Every MTP trial now
runs the complete independent FP8 probe across all four variants, so the reduced
sanitizer workload still exercises the new tile, including every finite code pair
and M/N/K tails, even when its target verification batch contains only two rows.

Rejected transposed tile `97383f9ee`: every FP8 fixture, independent model, MTP
partition and exact target output/state check passes, but synthetic MTP drops to
21.39 tokens/s. Verification rises from 204.07 to 231.49 ms; replay stays 56.54 ms.
The ordinary target control remains 25.27 tokens/s. Restore the existing four-row
FP8 kernel for four-to-fifteen rows. The new kernel remains an explicitly
experimental probe, with no resident-model dispatch. Offline compilation uses
64 registers with no spills; zero spills alone does not establish performance.
Ninfer was restored at 15:08:42 EDT, PID 3400966, HTTP 200; ComfyUI unchanged.

Follow-up candidate: select the transposed tile only for small batches with at
least 16,384 output channels. The profile shows the 17,408-channel MLP projections
fall from 6.52 to 4.44 ms and the vocabulary head from 5.57 to 2.41 ms, while
narrow projections regress. The wider matrices launch enough channel tiles to
make this shape useful; keep the old four-row kernel for narrower matrices.
This is a measured dispatch hypothesis, pending end-to-end qualification.
