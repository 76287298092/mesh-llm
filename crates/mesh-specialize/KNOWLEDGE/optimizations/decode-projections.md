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
