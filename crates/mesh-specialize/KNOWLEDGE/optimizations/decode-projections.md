# Decode projection performance iteration

The user requested direct inspection of Ninfer and performance iteration on
September 27. The parent archived tracked Ninfer source at Carrack revision
`9e163eee4b8acec21ab0ac765107b6a3f287b217` into ignored
`target/specialize/perf-20260927/ninfer-source/`. This is a source archive, not a
Git checkout. The working-tree serving configuration modification is excluded.
The installed binary's source provenance remains unverified. No Ninfer source is
copied into the runtime or independent reference.

The short-prefix profile attributes 83.39% of kernel-event time to FP8 projections
and 11.26% to BF16 gate projections. The first experiment replaces single-row FP8
execution with four warp-owned exact dot products per block. Finite E4M3 values
are signed integers divided by 512. Product sums at width at most 32768 are exact
in i64 and fit below 2^51. This removes positive-product MMA, FP64 tile accumulation
and divergent serial recomputation. Conversion to FP32 followed by the exact
power-of-two scale and existing row/channel products preserves the independent
FP64-dot contract. Prefill keeps the prior kernel.

Independent fixtures cover every finite code pair, K/output tails, maximum-width
same-sign totals and cancellation. Full-model one/two-token checks, partition
state equivalence and sanitizers must pass before retaining the optimization.
The existing raw-token model timing and kernel profile are the before control.
Status: implementation under qualification, no speedup claim yet.


A second bounded candidate uses a warp-parallel FP64 dot for the small BF16 A/B
projections. It avoids computing padded 16-row tiles and ambiguous-rounding serial
fallback. It retains FP64 arithmetic, but the summation order differs from the
sequential CPU reference. Component and full-model exact-output gates decide
acceptance; no universal BF16-input equality is asserted. It applies to the small
BF16 projections for all row counts; unrelated BF16 MMA controls remain available.

First trial source `451f960ec5cac0e82db8af118fdbc364dea0f0b4`, PTX SHA256
`19bc7879f2d6497813f7f461566c3648e5fa8f1381083d7b0c5012b381cbd190`.
All 64,544 standalone FP8 outputs match the independent FP64 dot at FP32 and BF16.
Two-token full-model hidden/logits and whole/token state agree exactly. Three
samples give short-prefix decode 8.029, 8.029, 8.031 tokens/s and 128-prefix decode
7.244, 7.242, 7.243 tokens/s, with unchanged output IDs. Prefill remains about
19.29 tokens/s. Profile total falls from 633.56 to 115.38 event milliseconds;
BF16 gates now account for 71.57 ms. This motivates the second candidate.
Raw evidence: `target/specialize/perf-20260927/fp8-exact-round/` on both hosts.
Ninfer restored 09:15:47 EDT, PID3218546, HTTP200; ComfyUI448118 unchanged.
Sanitizers for the final combined candidate remain pending.


The next candidate also evaluates the exact FP8 kernel for prefill rows, retaining
its existing multi-row launch contract. This is an experiment against the same
128-token prompt, not a general large-batch GEMM claim. The old wide kernel stays
available as an arithmetic/performance control. No activation-quantization policy
or output conversion changes accompany this dispatch experiment.


NVFP4 now reads aligned full four-byte data/scale words rather than reconstructing
each from four separate byte reads. All MMA instructions, operand bits, reduction
order and launch geometry remain unchanged. Bounds/alignment-checked tails keep
the byte path. The model trial also runs the existing independent signed/tail
NVFP4 fixtures at K16 and K80 to exercise both paths and unaligned scale rows.
Resident arena objects and CUDA buffers provide the required base alignment.

Second trial source `600c30cc0857abcca199835565356735f0f04fbd`, PTX SHA256
`f8cec86413ab74197e06cee2a5a9e82613472c687ed58deccef6e503ad54edfc`.
BF16 cancellation/tail probes and all full-model exact-output/state checks pass.
Three short-prefix samples give 17.935, 17.928, 17.931 decode tokens/s; 128-prefix
samples give 14.395, 14.391, 14.390. Prefill remains 19.66 tokens/s. The profile
measures BF16 gates at 2.677 ms, down from 71.568 ms, and NVFP4 at 24.328 ms.
Raw evidence: `target/specialize/perf-20260927/bf16-round/` on both hosts.
Ninfer restored 09:20:18 EDT, PID3221946, HTTP200; ComfyUI448118 unchanged.


The next NVFP4 candidate groups four independent warp MMA tiles into each CTA.
Logical N tiles are CTA.x*4+warp; K order and arithmetic are unchanged. The original
one-warp entrypoint remains compiled as a control. This tests scheduling geometry,
not a numerical or quantization change. Compare the same two/128-token cases
before retaining it; higher thread count alone does not establish a speedup.

Third trial source `b8214880c102d90d0a9851953912d73bda50c52e`, PTX SHA256
`28ee5779ee2d37d4875ab9dca3a9c230550b4531b28a98588aebd9f183de8f73`.
The exact FP8 path for prefill plus aligned NVFP4 loads passes signed/tail fixtures
and all exact full-model gates. The 128-input prefill rises to 112.517-112.667
tokens/s across three samples. Decode reaches 19.982-19.992 at the short prefix
and 15.680-15.688 at the 128-token prefix. Output IDs are unchanged. NVFP4 summed
event time drops to 18.633 ms. Ninfer restored 09:25:27 EDT, PID3224976, HTTP200;
ComfyUI448118 unchanged. Raw directory: `word-round/` under the experiment root.


The attention candidate preserves the original FP64 reduction addition tree while
replacing its final five CTA-wide barriers with warp shuffles. Only thread zero
now computes the identical score/online-softmax scalars, which it publishes in
shared slots1..3 before a CTA barrier. Shared slot0 remains the reduction result
until every thread has consumed it. The end-of-token barrier protects reuse.
The old implementation performed the same exponentials on all256 threads.
Qualification must preserve full-model logits/state and benchmark output IDs;
all three sanitizers cover the changed synchronization.
