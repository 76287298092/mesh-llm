# GDN recurrent matrix update

Status: implementation under local validation. Real GPU execution and sanitizer
qualification are pending. This is not a complete attention layer or model.

The pinned [Transformers recurrent fallback](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L441)
decays the FP32 state matrix, predicts the current value from the key, applies a
beta-weighted correction and projects the updated state with the query. Layer
zero has 48 value heads and a 128 by 128 state per head. Each value head maps to
one of 16 unrepeated key heads by integer division by three.

The Rust kernel consumes resident normalized Q/K, beta and decay from the
qualified preparation kernels, and reads BF16 V from the retained convolution
buffer. No host reference output replaces those device buffers. State is FP32
and time-major outputs are BF16 with a separate FP32 diagnostic buffer. One block
owns each value head, and one thread owns a complete value column of its state.
It processes tokens sequentially, with explicit rounded FP32 multiply/add and
no FMA. The prediction and query reductions run in increasing key-coordinate
order. Disjoint column ownership needs no inter-thread state synchronization.
This first implementation prioritizes a checkable arithmetic contract; no speed
claim or efficient parallel prefill is implied.

The independent scalar oracle indexes logical row/head/key/value coordinates.
Its ordered FP32 mode supplies the exact arithmetic acceptance contract. Every
FP32 output and state element must match bit for bit; BF16 outputs must match
exactly. Hand-calculated scalar fixtures guard the equation and head mapping.
A second mode accumulates the two dot products in f64, then rounds to FP32.
The report preserves output/state error and BF16 differences from this wider
reduction. Those diagnostics are not an acceptance tolerance or evidence of
end-to-end model quality. A full-layer/logit comparison remains required.

Whole-sequence execution is compared with `[1, remaining]`, `[2, 1, remaining]`
and single-token partitions. The host reads state at each boundary for checking,
but never uploads a replacement between chunks. All output and final state bits
must agree between partitions. Dedicated signed fixtures use asymmetric nonzero
state, two key heads, six value heads, zero/one beta and decay, and a separate
zero-state reset. Host validation rejects invalid dimensions/extents, nonfinite
values, gate-domain violations and arithmetic overflow before launch.

Two bounded Luna-max workers own the device kernel and scalar reference. The
parent specifies arithmetic, ownership and ABI, then integrates retained inputs,
chunk lifecycle, diagnostics and deployment. Reproduction extends the existing
`qwen-projection-check` command to schema 5, with fresh evidence under
`target/specialize/qwen-gdn-recurrent-20260927/`. Prior preparation, convolution
and projection evidence remains intact. Model performance and context fields
stay null and `model_executable` stays false.

Local validation passes 107 library tests, focused all-target/all-feature Clippy
with warnings denied, and Rust PTX compilation. The initial test run exposed two
fixture errors: a query selects a state row across every value column, and a
rejection fixture accidentally used valid dimensions. Both expectations were
corrected without changing the recurrence. The initial failed log is retained.
The f64 diagnostic also rejects any product that would overflow FP32, keeping
its input domain aligned with the ordered contract. GPU fixtures additionally
cover widths one and 256, alongside the model's width 128.
