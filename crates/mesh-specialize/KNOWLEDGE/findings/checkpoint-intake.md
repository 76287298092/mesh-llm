# Pinned Qwen checkpoint intake

Status: source files located and independently hashed; Rust importer implementation
and its real-file trial are in progress. This is not model execution or a kernel
layout qualification.

The input is `unsloth/Qwen3.8-27B-NVFP4` at revision
`f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Carrack already holds the source files
under `/data/ai/models/src/Qwen3.8-27B-NVFP4-unsloth/`. The superficially related
Hugging Face hub directory contains only a revision reference; it is not the
location of the resident weight data. No extra source-model download is needed.

The [pinned upstream metadata](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/revision/f0b7c9e722f5565102fff8481c99e4d86ae099c7?blobs=true)
reports the following LFS identities, all matched by a fresh full-file SHA-256
on Carrack on September 27, 2026:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 22,568,192,096 | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` |
| `model_mtp.safetensors` | 849,400,392 | `1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |

Config and index files match fresh downloads at that same immutable revision.
The importer also pins the tokenizer config, generation config and chat template;
all eight source identities are recorded in its recipe. It does not parse the
installed `.ninfer` artifact or import NInfer code.

## Physical tensor contract

The main Safetensors header is 251,128 bytes and contains 1,953 tensors. Of these,
1,620 are text-model tensors (21,646,480,768 data bytes), and 333 are vision tensors
(921,460,192 data bytes). The MTP file has a 1,600-byte header and 15 BF16 tensors
(849,398,784 data bytes). These are file-layout counts and sizes, not GPU residency.

The first import keeps all text and MTP tensors, omits vision, and preserves each
kept tensor's dtype, physical shape and bytes. In particular, packed NVFP4 weight
codes remain U8 physical tensors, their block scales remain FP8, and global
scales remain FP32. Interpreting these storage tensors and choosing CUDA operand
layouts belongs to the compiled model implementation. This step does not quantize,
dequantize, tile, dispatch a graph or execute MTP. The six auxiliary assets and
recipe are included in the artifact's content-derived identity.

The [Safetensors format](https://github.com/safetensors/safetensors/blob/main/README.md#format)
uses an eight-byte little-endian header length followed by JSON and a contiguous
tensor buffer. The importer uses the workspace's existing Rust `safetensors` 0.8
format validation, with a bounded header and explicit duplicate-name rejection.
It rejects malformed lengths and ranges before reading large payloads.

## Pinning copied bytes

One sequential pass hashes the exact file prefix, header and all payload bytes
against the pinned full-file digest. During that same pass it hashes each tensor.
Only successful whole-file verification exposes those per-tensor digests. This
avoids accepting hashes calculated from a different read after source validation.

The artifact writer accepts an absolute source-file range and expected digest for
each tensor. It reads only that range, compares its digest, then rechecks it during
assembly. Replacing or editing an input cannot silently substitute a different
tensor after its pinned verification. The writer still opens one source at a time,
uses bounded buffers and refuses to replace an existing output. No intermediate
per-tensor files or second extracted checkpoint are required.

The final artifact is read back through `VerifiedArtifact::open_for_identity`.
The report labels artifact integrity separately from model execution, with
`model_executable: false`. Real source intake cannot qualify an absent execution
engine or establish that NInfer's additional conversion choices were identical.

Raw source evidence is under `target/specialize/checkpoint-intake-20260927/`.
The committed [evidence directory](../evidence/checkpoint-intake-20260927/)
contains upstream identities, full-file hashes, config and compact JSON copies of
the tensor headers. Those JSON copies omit padding; reported `header_sha256`
values refer to the original padded source headers, not the compact copies.
Tests, exact code revision, output identity and measured conversion resources will
be appended after the importer runs. Ninfer remains active during this CPU work.

## Local implementation validation

The Rust importer and range-copy writer pass 82 library tests on macOS, including
bounded metadata parsing, malformed/duplicate entries, physical dtype and range
mapping, checksum rejection, source mutation and no-clobber behavior. Clippy for
`mesh-specialize` and `xtask` passes across all targets/features with warnings
denied; changed Rust files pass rustfmt and the repository no-console check passes.
A missing-input CLI trial exits 1, saves `all_passed: false` and creates no artifact.
These checks qualify the importer implementation, not model execution.
