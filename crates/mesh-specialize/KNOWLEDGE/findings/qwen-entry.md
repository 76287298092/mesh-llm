# Qwen embedding and input normalization

Status: real-weight GPU execution, independent numerical comparison and all three
CUDA sanitizers pass on Carrack. Full model execution and serving performance
remain unimplemented.

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
kernel source file.

Local validation: 89 macOS library tests, focused all-target/all-feature Clippy
with warnings denied, changed-file rustfmt and repository no-console checks pass.
The captured-header test verifies the complete compiled tensor contract. These
are host checks; device execution is recorded below.
The pinned upstream implementation file has SHA-256
`f567c3e4b7db57b5d04be57a8fe9788e7c19aaa01aad877ee0daf94b2dab1925`.

## Carrack execution

Source revision: `06ba51c74fc3c85a735ce2cbefe833abf27329f1`.
Device: RTX 5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120,
driver API 13040. Host release xtask was built with `just specialize-tools-build`;
SHA-256 `5d0b0793ef746e2132db8705d796a178ea0d732995d8d05ae77a3c00de31d82a`.
Device PTX was built on macOS with `just specialize-ptx`, pinned
nightly-2026-09-25 and rebuilt NVPTX core, then transferred to Carrack. Both copies
have SHA-256 `edf86ed54f26e52a9bd26a8f1201bcad1a78d32e55090457653bd15d03e7e90c`.
The initial attempt to compile PTX on Carrack failed because rustup is absent;
its log is preserved. No compiler was installed and no global toolchain changed.

The [ordinary report](../evidence/qwen-entry-20260927/normal.json) passed all three
cases, with 696,320 values in each output array. Residual BF16 words and normalized
BF16 outputs match the scalar reference exactly. FP32 normalization has maximum
absolute error `9.5367431640625e-7`, within the declared tolerance. Every GPU BF16
output also equals the independently rounded GPU FP32 output. The allowed
one-ULP boundary difference was not needed for these fixtures.

The kernel uses 22 registers, 1,024 bytes static shared memory and zero local
memory. Resident embedding/norm payload is 2,542,807,040 bytes. Driver-reported free
memory falls from 32,221,822,976 to 29,675,880,448 bytes with these weights resident.
This observation excludes full-model weights, KV/recurrent state and serving
buffers; it is not a full-model memory measurement. The 12.973-second command
duration includes artifact hashing, copies and reference computation, and is not
a kernel throughput measurement. No GPU timing or context-capacity claim is made.

[Memcheck](../evidence/qwen-entry-20260927/memcheck.log),
[racecheck](../evidence/qwen-entry-20260927/racecheck.log) and
[synccheck](../evidence/qwen-entry-20260927/synccheck.log) each pass with zero
errors or hazards and successful numerical reports. Each run has an 8 GiB host
memory limit, swap disabled and a 180-second timeout. The changed PTX also passes
all 14 existing representative GEMM/RMSNorm cases in check-only mode. Linux passes
93 library tests, 17 validator tests and all-target/all-feature Clippy with
warnings denied. The release link reports the existing mold-to-lld fallback.
GitHub returns no Actions runs for this branch; no CI-green claim is made.

Ninfer was stopped only during the four entry runs and workload regression.
ComfyUI PID 448118 remained at 498 MiB. Ninfer restarted at 01:18:50 EDT on
September 27, reached engine-ready at 01:18:57, and returned HTTP 200 from
`/health`. New PID 2837344 uses the original 30,046 MiB GPU allocation.
The before/during/after process records and readiness evidence are committed in
the [evidence directory](../evidence/qwen-entry-20260927/). Raw logs and PTX remain
under `target/specialize/qwen-entry-20260927/` on both hosts.

This closes the compiled tensor metadata contract and the first real-weight
operation only. Q01 is partial: the model still needs layer execution, recurrent
and KV state, output projection, tokenizer/sampling and host integration. The
next bounded operation is layer-zero projections feeding Gated DeltaNet, using
the same imported FP8 weights and independent numerical checks. Full prefill,
decode, context and memory comparison remains open.
