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
