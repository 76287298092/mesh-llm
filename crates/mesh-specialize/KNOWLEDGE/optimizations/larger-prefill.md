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
No quantization profile change is intended. Candidate status: implemented and compiling; runtime dispatch selects the new
tile for 16 or more rows. Smaller batches retain the existing exact kernels.
The component oracle exercises all finite code pairs, row/column/K tails, signed
cancellation and maximum width across all three tile variants. The profiler now
collects separate prefill and decode kernel events; timings remain diagnostic.
Candidate numerical and throughput results are pending.

Baseline source `05fb3f8a4` and retained decode PTX
`64cf82c90ba3d25299e61693af46b8523aba122cd989810274a7b2786687cda9`
pass the new whole/token partition check for both two and 512 prefix tokens.
Three-sample median prefill is 136.910 tokens/s at 128 inputs and 142.130 at 512;
512-prefix decode is 16.355 tokens/s. Its attention decode event time is 22.065 ms,
which identifies another longer-prefix bottleneck to revisit separately. The
[raw baseline](../evidence/iterate-20260927/prefill-baseline/) includes exact
logit/state comparisons, memory checks and service restoration at 14:04:38 EDT,
PID 3297121, HTTP 200. ComfyUI PID 448118 remained unchanged.
