# Larger prefill continuation

The dedicated decode change is retained. Before changing prefill kernels, expand
the bounded benchmark/profile harness from 128 to 512 input tokens. The profile
now compares its ordinary batched prefix with an independent token-at-a-time
execution through the same resident model, then checks prefill logits, one decode,
and the complete recurrent/KV state. This supplements the independent arithmetic
oracle; it is not an independent 512-token whole-model reference.

Next candidate: exact FP8 16x8 output tiles using signed INT8 tensor-core products.
E4M3 integer units decompose into three signed base-128 digits; nine K32 products
reconstruct each exact dot in i64 before the existing scale/rounding sequence.
No quantization profile change is intended. Status: retained after full qualification; runtime dispatch selects the new
tile for 16 or more rows. Smaller batches retain the existing exact kernels.
The component oracle exercises all finite code pairs, row/column/K tails, signed
cancellation and maximum width across all three tile variants. The profiler now
collects separate prefill and decode kernel events; timings remain diagnostic.
Numerical and throughput results are recorded below.

Baseline source `05fb3f8a4` and retained decode PTX
`64cf82c90ba3d25299e61693af46b8523aba122cd989810274a7b2786687cda9`
pass the new whole/token partition check for both two and 512 prefix tokens.
Three-sample median prefill is 136.910 tokens/s at 128 inputs and 142.130 at 512;
512-prefix decode is 16.355 tokens/s. Its attention decode event time is 22.065 ms,
which identifies another longer-prefix bottleneck to revisit separately. The
[raw baseline](../evidence/iterate-20260927/prefill-baseline/) includes exact
logit/state comparisons, memory checks and service restoration at 14:04:38 EDT,
PID 3297121, HTTP 200. ComfyUI PID 448118 remained unchanged.

## First candidate result

Source `6a6d7222d47c322a2e960c17be49ee7a534967e2` and PTX
`eec97149de8153eee85a9bb6d5422091aeeddacdc6896c07fa49d4e0ea36a0d7`
pass all independent finite-code, signed, tail, cancellation and maximum-width
cases in the three FP8 tile variants. Full two-token hidden/logit/state evidence
stays exact; whole/token prefix and post-decode state also agree exactly at 512.
The JIT reports 60 registers and zero local/shared bytes; offline PTXAS reports
64 registers without spills. Linux tests and Clippy pass.

Initial medians: 259.333 input tokens/s at 128 and 296.347 at 512, compared with
136.910 / 142.130 before. Decode is unchanged at about 25.36 short, 22.38 after
128 inputs and 16.34 after 512. The 512-input prefill profile totals 1,563.845 ms
of GPU events: attention 243.595 ms, GDN recurrence 237.888 ms, and FP8 exact
tile groups remain substantial. These are raw-token single-sequence results.
The [initial trial evidence](../evidence/iterate-20260927/prefill-round/) includes
memory release and Ninfer restored at 14:09:15 EDT, PID 3300523, HTTP 200.
ComfyUI PID 448118 remained unchanged. The [final qualification round](../evidence/iterate-20260927/prefill-qualified/)
passes memcheck, racecheck and synccheck without errors/hazards, as well as exact
full-model and 512-token partition/state checks. Repeated prefill medians are
259.383 tokens/s at 128 and 296.581 at 512. The service restored at 14:14:38 EDT,
PID 3311383, HTTP 200; ComfyUI remained unchanged. This closes the first larger
prefill improvement; resident MTP is the next area.
