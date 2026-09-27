# Pinned Qwen checkpoint intake

Status: pinned Rust import and full artifact readback passed on Carrack. This is
not model execution or a kernel layout qualification.

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
Ninfer remained active during this CPU work.

## Local implementation validation

The Rust importer and range-copy writer pass 82 library tests on macOS, including
bounded metadata parsing, malformed/duplicate entries, physical dtype and range
mapping, checksum rejection, source mutation and no-clobber behavior. Clippy for
`mesh-specialize` and `xtask` passes across all targets/features with warnings
denied; changed Rust files pass rustfmt and the repository no-console check passes.
A missing-input CLI trial exits 1, saves `all_passed: false` and creates no artifact.
These checks qualify the importer implementation, not model execution.

## Carrack real-file trial

Code revision `963d5a33ef4f32a40c3819a7e30614bf7e83472a` passed 85 Linux library
tests, 17 validator tests and Clippy for both crates across all targets/features
with warnings denied. The `just with-lld cargo build -p xtask` build succeeded;
the linker reported its existing mold-to-lld fallback. GitHub returned no Actions
runs for this branch; this is local and Carrack validation, not a CI-green claim.

The real import completed in 58.635 seconds and passed full `.mspec` readback.
It produced `/data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec`:

- 22,516,627,200 bytes; 1,635 text/MTP tensors, six assets and one recipe.
- Model ID: `qwen3.8-27b:text:nvfp4-fp8:upstream-raw-v1`.
- Weights ID: `sha256:f49713878a072f8c9043060dc0e2f3b28421301e49471bee0c13c7570e59e81e`.
- `model_artifact_verified: true`; `model_executable: false`.

The command ran in a systemd user scope with `MemoryMax=8G`, `MemorySwapMax=0`
and a 600-second timeout. Cgroup memory peaked at its 8 GiB limit, including file
cache; this is not process RSS or model GPU memory. A single observation near
readback showed 11,208 KiB process RSS; process peak RSS was not measured. The
shell reported 44.594 user CPU seconds and 10.482 system CPU seconds. These are
debug-build conversion resources, not model throughput or optimized import speed.

The first launch failed before import because `/usr/bin/time` was absent. Its
`import.log` is retained; the successful retry uses shell timing and systemd
accounting in `import-retry.log` and `import-resources.txt`. Reproduce the core
operation after building through `just`, using fresh output/report paths:

```sh
target/debug/xtask specialize checkpoint-import \
  --input-directory /data/ai/models/src/Qwen3.8-27B-NVFP4-unsloth \
  --output /data/ai/models/mesh-specialize/FRESH-NAME.mspec \
  --report target/specialize/FRESH-REPORT.json
```

Ninfer user service `ninfer-qwen38.service` stayed active with PID 2697705 and
30,046 MiB GPU allocation. ComfyUI PID 448118 remained at 498 MiB. Before/after
service and GPU-process evidence is saved beside the import report.

Read-only review found the source ranges, physical dtypes, pins and identity
mapping consistent. One intentional failure behavior needs care: if final
readback fails after publication, the failed report is authoritative and the
artifact stays at its final path for diagnosis. Do not treat existence as
success; inspect `all_passed`. A retry requires a fresh output path. The importer
does not automatically delete failed evidence.

The next model gate is a compiled tensor inventory and executable schedule,
followed by independent layer/state/logit validation. Import success closes only
pinned source intake; H03/H04, full inference and competitive serving remain open.
