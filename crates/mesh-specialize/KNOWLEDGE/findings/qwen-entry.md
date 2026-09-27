# Qwen embedding and input normalization

Status: implementation in progress. This is the first real-weight operation;
full model execution and serving performance remain unimplemented.

The compiled inventory checks exact artifact identity and all 1,635 tensor names,
physical shapes, dtypes and byte lengths. Its independent evidence is the pinned
upstream Safetensors headers. The 64-layer schedule has 48 Gated DeltaNet layers
and 16 full-attention layers. MLP layers 0–55 use packed NVFP4, and 56–63 use FP8.
MTP tensors remain preserved but are not executed by this trial.

The first operation loads the verified BF16 embedding table and layer-zero input
norm weights. GPU embedding lookup preserves each BF16 word into the residual.
The same kernel applies zero-centered RMSNorm with epsilon 1e-6, multiplies in
FP32 by one plus the stored weight, and rounds once to BF16. It also writes the
unrounded FP32 result for qualification. Future optimization may remove that
validation-only output once the real execution path is established.

These semantics are checked against
[Transformers Qwen3_5RMSNorm at 96331a9](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L841).
The checkpoint config names the Qwen3.5 architecture despite its Qwen3.8 product
name. The reference calculation independently accumulates squares in f64, then
uses FP32 normalization and BF16 round-to-nearest-even. The GPU uses a 256-thread
FP32 reduction. Acceptance requires exact embedding words, finite FP32 results
within `2e-6 + 2e-6 * abs(reference)`, exact BF16 rounding of the GPU FP32 output,
and no more than one BF16 ULP from the scalar reference at rounding boundaries.

The trial uses one, seven and 128 token IDs, including vocabulary endpoints,
repeated IDs and model special IDs. These are lookup fixtures, not tokenized
requests or context-capacity evidence. It loads the complete 2.54 GB embedding
table; it does not represent full-model GPU residency. No prefill/decode metric
is emitted. Use `xtask specialize qwen-entry-check --artifact PATH --ptx PATH
--device ORDINAL --output NEW_FILE` after building through `just`.

Parent owns model semantics, module wiring, scalar reference, launch/check code,
CLI and deployment. Bounded Luna-max workers own the compiled inventory and one
kernel source file. Tests and Carrack execution evidence will be recorded here.

Local validation: 89 macOS library tests, focused all-target/all-feature Clippy
with warnings denied, changed-file rustfmt and repository no-console checks pass.
The captured-header test verifies the complete compiled tensor contract. These
are host checks; Linux compilation, device execution and sanitizers are pending.
The pinned upstream implementation file has SHA-256
`f567c3e4b7db57b5d04be57a8fe9788e7c19aaa01aad877ee0daf94b2dab1925`.
