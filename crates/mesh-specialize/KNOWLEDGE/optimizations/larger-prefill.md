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
No quantization profile change is intended. Candidate status: design review; no
new device kernel is selected yet. Larger-context and throughput results pending.
