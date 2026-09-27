# Ninfer model format and resident layout investigation

Status: source investigation, September 27, 2026. No performance experiment, implementation, build, service action, or GPU job ran for this investigation. Gains are not measured.

Ninfer's packed format enables useful execution choices, especially fused projection parents and NVFP4 scale staging. Its FP8 weight matrix is still row-major. There is no hidden persistent FP8 transpose that explains the current F02 gap. Prioritize kernels and fusion while isolating the smaller, concrete layout differences.

## Revisions and coverage

Ninfer source references below are pinned to `9e163eee4b8acec21ab0ac765107b6a3f287b217`. `N:` denotes a file at [that revision](https://github.com/Neroued/ninfer/tree/9e163eee4b8acec21ab0ac765107b6a3f287b217), not the installed executable. Local source was inspected under `target/specialize/perf-20260927/ninfer-source/`; missing converter and model-card files were read from the corresponding immutable `raw.githubusercontent.com` URLs. `M:` denotes this repository at `5298b42708088fa04e507ddcfd44f0e1f62402a4`, paths relative to `crates/mesh-specialize/`.

The investigation traced the Qwen3.8 NVFP4 recipe, container writer, FP8/NVFP4 codecs, device materializer, and representative FP8 decode/prefill, NVFP4 decode/prefill, and fused SwiGLU consumers. It did not exhaustively audit every groupwise integer, MoE, vision, DFlash, or TMA variant. This worker did not inspect the current installed artifact's directory or refresh its hash, establish installed binary-to-source provenance, or measure whether the running profile selects any particular candidate dispatch. The recorded September 26 baseline artifact hash does match the published manifest, as detailed below. These limits matter because the current pinned source uses container v3, while an older installed artifact may differ.

Hardware target from the existing trial record is Carrack GPU0 RTX5090, SM120, driver 615.71.09; this worker did not refresh hardware, clocks, or process state. Rust toolchain, contention, and timing are not applicable to this read-only investigation.

## Provenance is a comparison gate

The pinned published [NVFP4 artifact manifest](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/model-cards/Qwen3.8-27B-nvfp4-NInfer/artifact-manifest.json#L18-L43) names base `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, quantized `unsloth/Qwen3.8-27B-NVFP4` revision `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`, and a separate DFlash2 source. Its artifact is 23,719,715,844 bytes, SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`, with text, vision, MTP, and DFlash2 components. The recorded September 26 deployed baseline identifies the same artifact bytes by SHA-256; this is a historical identity match, not a fresh check of the currently installed file.

The [September 26 baseline report](ninfer-baseline-20260926.md), lines 36-39, records weight SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` for `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`. It exactly matches the pinned published artifact manifest. The baseline hashed file bytes without parsing the container. The same report records binary SHA-256 `140ec5e660fb613ec2f57d2a99c4a274a0363f2cdca7e2652ca7a0337bff49ad`, but does not connect that executable to a source revision. Its companion [metrics JSON](../evidence/ninfer-baseline-20260926.json) contains nine request records, including warmup and warm-cache repeat, with token counts, timing, and draft acceptance counters; it contains no artifact or binary hashes. Thus the hash evidence comes from the saved report, while the JSON supports its workload/timing observations.

This historical artifact match connects the baseline file to the published artifact's declared conversion provenance and component inventory. It does not establish that the file remains installed unchanged, that the recorded executable implements every path in the source pin, or that the artifact's logical quantized weights equal our raw Safetensors intake. Missing historical HF metadata still prevents that last comparison. Source claims about the general converter remain separate from the manifest's claims about this artifact.

Our intake pins quantized revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`, source-file digests, 1,620 text tensors, and 15 BF16 MTP tensors. It copies original tensor bytes and retains a BF16 embedding. See `M:src/checkpoint/qwen3_8_recipe.rs:9-40,79-104` and [checkpoint intake](checkpoint-intake.md). A different revision string does not prove different tensor bytes; neither does a shared model name prove equality. Parent should compare canonical logical tensor digests and conversion records before attributing output differences to arithmetic or claiming matched weights.

Ninfer's official recipe generates FP8 embedding weights from base BF16, imports encoded FP8/NVFP4 for text projections, and quantizes optional MTP projections to Q8. Our BF16 MTP and embedding are therefore additional potential workload/quality differences. The baseline artifact hash matches the published manifest that names this recipe; a fresh installed-state check and direct conversion-input comparison remain outstanding.

## Hugging Face revision metadata follow-up

A read-only metadata lookup on September 27, 2026 resolved our pinned revision through the [Hugging Face model API](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/revision/f0b7c9e722f5565102fff8481c99e4d86ae099c7?blobs=true). The response `sha` exactly matched the requested revision. It reported these identities; no weight payload was downloaded or rehashed in this follow-up.

| File | Bytes | Reported identity at `f0b7c9e7` |
| --- | ---: | --- |
| `model.safetensors` | 22,568,192,096 | LFS SHA-256 `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` |
| `model_mtp.safetensors` | 849,400,392 | LFS SHA-256 `1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe` |
| `tokenizer.json` | 19,989,325 | LFS SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `model.safetensors.index.json` | 164,371 | Git blob ID `7608ff001dbfc8936318df32aaaaef7c8c9f340d` |
| `config.json` | 22,564 | Git blob ID `b6f6347774036d406eabed6cfffb0fec424ba075` |
| `tokenizer_config.json` | 1,047 | Git blob ID `088fbebf189b39e2dabcdb12a83a31617fe98c2e` |
| `generation_config.json` | 214 | Git blob ID `0bc3addd19dc59c5c8899fc1fb887d50b592e7c3` |
| `chat_template.jinja` | 9,993 | Git blob ID `a087700658910c336c9ca9f5780a75a3cdd4fcdd` |
| `vocab.json` | 6,722,759 | Git blob ID `0aa0ce0658d60ac4a5d609f4eadb0e8e43514176` |
| `preprocessor_config.json` | 390 | Git blob ID `2ea84a437d448ff71b08df68fdd949d5cc4ebb64` |
| `video_preprocessor_config.json` | 385 | Git blob ID `3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe` |

The other two listed files were `.gitattributes`, 1,570 bytes, blob `52373fe24473b1aa44333d318f578ae6bf04b49b`, and `README.md`, 6,746 bytes, blob `3ddbfc9c1f10c6c82edbcb04a0edfde3f830b6f4`. Git blob IDs are not raw-file SHA-256 digests; their equality would be a metadata identity comparison of the corresponding Git objects.

The published Ninfer manifest's quantized revision `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` returned HTTP 404 from all four attempted primary endpoints:

- [Model revision with blob metadata](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/revision/60e813d4dbbdc5d64cf3f5a8caf2897bedf03679?blobs=true).
- [Recursive expanded tree](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/tree/60e813d4dbbdc5d64cf3f5a8caf2897bedf03679?recursive=true&expand=true).
- [Commit listing at the revision](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/commits/60e813d4dbbdc5d64cf3f5a8caf2897bedf03679).
- [Raw config at the revision](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/raw/60e813d4dbbdc5d64cf3f5a8caf2897bedf03679/config.json).

The [current main commit listing](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/commits/main) returned five commits: `f0b7c9e7`, `57926bac`, `9e3d73c7`, `7d6f8d4d`, and `16b6615a`. The last is titled `Super-squash branch 'main' using huggingface_hub`, dated August 15, 2026. This suggests a possible reason historical metadata is unavailable, but does not establish that the missing revision was an ancestor or that its payload matched. The listing is a current observation, not an immutable historical proof.

Result: our existing source pins are confirmed by current primary metadata. Comparison against the manifest revision is unresolved. There is no evidence here to classify its differences as weight changes, metadata-only changes, or repository additions. Closing this gap needs an independently preserved tree/LFS manifest for that exact revision, or canonical logical tensor hashes from the actual Ninfer conversion inputs. The September 26 Ninfer artifact identity is connected to the published manifest by its recorded SHA-256, but that does not supply the missing HF file identities. Current installed state and binary-to-source correspondence remain unverified.

## Conversion and numeric choices

The actual CLI builds the architecture model and recipe, records source provenance, and calls `convert`; the pipeline prepares jobs and writes their objects through `ArtifactWriter`, then writes a `.conversion.json` report. See `N:tools/convert/__main__.py:84-184` and `N:tools/convert/pipeline.py:27-139`.

The [official Qwen3.8 recipe](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/tools/convert/official_recipes.py#L151-L174) selects NVFP4 MLP projections for layers 0 through 55; other text projections, including the output head and last eight MLPs, use row-scaled FP8. GDN A/B and nonprojection parameters remain direct. Activation permissions are `AllowA4` for NVFP4 and `AllowA8` for FP8; permission does not establish which dispatch executes. Optional-component projection quantization is Q8 at lines 19-46, with listed exceptions.

Encoded import preserves compatible FP8 code bytes and BF16 multipliers, or NVFP4 nibble bytes, E4M3 block scales, and FP32 global divisor. NVFP4 reconstruction is `E2M1(code) * E4M3(scale) / weight_divisor`; the activation divisor is separate metadata. See `N:tools/convert/sources/compressed_tensors.py:31-132`. The NVFP4 method is an import, not an independently defined fresh NVFP4 quantization algorithm.

Embedding FP8 conversion takes BF16 input, computes row max-absolute divided by 448, rounds the multiplier to BF16 ties-to-even, protects nonzero underflow with the minimum BF16 subnormal, normalizes using the rounded multiplier, clamps to 448, and rounds E4M3FN ties-to-even. See `N:tools/convert/quantization/fp8_row.py:28-121`. This changes weight values relative to our retained BF16 embedding and must be assessed as quantization, not just layout.

## Container and byte layout

| Representation | Persistent bytes and alignment | Execution consequence |
| --- | --- | --- |
| Ninfer v3 framing | 32-byte little-endian `<8sQ16s>` header: `NINFER\0\3`, directory length, 16-byte artifact ID. Continuations use `NINPRT\0\3` and part index. JSON directory includes components, objects, bindings, uses, and files. Payload starts on 4096-byte boundaries; tensor layout alignment is 256 bytes. | Directory bindings describe logical subranges of physical parents; neither JSON nor file alignment accelerates an already resident dot product. |
| FP8 `row_scale_v1` | `N*K` row-major E4M3FN bytes; BF16 row multipliers begin at `align_up(N*K,256)` and occupy `2*N` bytes. No padded N/K, persistent transpose, or code swizzle. | Decode and prefill load contiguous K segments from each output row. |
| NVFP4 `block_scale_k16_m128x4_v1` | `N*K/2` low-nibble-first code bytes in row-major order; scale plane starts at the next 256-byte boundary and occupies `N*K/16` bytes; FP32 weight divisor follows. Requires N divisible by 128 and K by 64. | Scale plane is already arranged for grouped contiguous reads used by MMA staging. Codes remain row-major. |
| Direct values | BF16, FP32, INT32 words in contiguous little-endian storage. | No conversion at device upload. |
| Groupwise integer alternatives | `row_split_k128_v1` pads K to 128 and separates code/high-bit/scale planes. | Relevant to optional MTP and other recipes; not the primary text FP8/NVFP4 code format. No complete integer-kernel audit here. |
| Our raw `.mspec` | 64-byte header, content identity and JSON directory digest, 256-byte object alignment. Each Safetensors code/scale/global-scale tensor remains its own raw object. | Current resident loader binds raw row-major tensors. NVFP4 scales stay natural `[N,K/16]`. |

Source: `N:tools/artifact/framing.py:5-8`; `N:tools/artifact/writer.py:43-84,128-177`; `N:tools/artifact/layouts.py:26-27,80-94,159-247`; `N:src/ops/linear/fp8/fp8_format.cpp:40-75`; `N:src/ops/linear/nvfp4/nvfp4_format.cpp:41-78`; `M:src/artifact/header.rs:5-103`, `M:src/artifact/schema.rs:6`, and `M:src/checkpoint/qwen3_8_recipe.rs:24`.

For natural NVFP4 scale coordinates row `r` and K-group `g`, Ninfer stores at:

```text
((r / 128) * (K / 64) + g / 4) * 512
    + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4
```

The converter implements this as reshape `[N/128,4,32,K/64,4]`, permute `[0,3,2,1,4]`, then contiguous flatten. See [codec](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/tools/artifact/codecs/nvfp4.py#L21-L37). The exact same address appears in `N:src/ops/linear/nvfp4/nvfp4_gemv.cuh:87-95`. Prefill copies scale slices with `cp_async` from the same persistent tile address, with a contiguous-row fast path and indexed-row fallback at `nvfp4_w4a4_mma.cuh:138-201`. Its MMA fragment loading consumes those shared scales at lines 258-314. This is a proven source-level layout/consumer connection, not a measured speedup.

Our exact decode reads scale `sw[output_channel * (K/16) + group]` and packed codes at `(row * groups_per_row + group) * 8`, in `M:kernels/nvptx/nvfp4_decode_exact.rs:104-118,178-206`. Merely applying Ninfer's scale permutation would break this consumer. A separate resident layout and matched kernel interpretation are required.

## GPU transformations and fused parents

FP8 prefill loads `weight_codes[weight_row*K + k_begin + logical_byte]`, then writes a schedule-specific shared-memory swizzle. It performs the activation swizzle similarly. Thus shared-memory banking is a kernel transformation, not an on-disk FP8 transform. See `N:src/ops/linear/fp8/fp8_a8_mma.cuh:77-82,145-185,209-229`. A16 MMA loads the same row-major bytes and widens them into swizzled shared BF16 at `fp8_a16_gemm_mma.cuh:130-150`. Decode reads contiguous code packs from the same row-major address at `fp8_gemv.cuh:120-129`, then applies the selected parent's BF16 row multiplier at lines 145-168. Our F02 also stages row-major FP8 into shared tiles and applies row/channel multipliers, `M:kernels/nvptx/fp8_native_prefill.rs:116-170,374-405`. Persistent FP8 packing is therefore not the first missing mechanism to chase.

The converter exposes mathematical projections, then concatenates compatible groups. Attention source Q/gate rows arrive interleaved by head; the adapter extracts their logical row ranges before forming parents in Q, K, gate, V order. GDN groups Q, K, V, Z and separately A/B. Dense MLP groups gate then up. See `N:tools/convert/qwen3_5.py:496-540,609-632`. Grouping is conditional on compatible representation and method, not a promise every arbitrary recipe fuses.

The fused FP8 SwiGLU consumer maps local rows to `row_begin + local_row % RowsPerBranch + (up ? IntermediateRows : 0)`. It executes one parent GEMV with gate/up output handling. See `N:src/ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh:12-19` and `fp8_linear_swiglu_decode.cu:17-41`. Our resident MLP currently invokes separate gate and up projections, then an activation launch, `M:src/kernels/cuda/resident_mlp.rs:53-64`. Storage grouping makes one-parent interfaces convenient; a Rust fused kernel can also take two pointers. A format rewrite is not a prerequisite for F03.

The optional proposal head is also a representation choice: default 131,072 selected vocabulary rows with an INT32 token mapping and Q4 format, while retaining the ordinary output head. See `N:tools/convert/proposal.py:44-105`. Selection comes from token ranking and special tokens. This can change draft cost and acceptance, and is separate from exact target verification. The September 26 baseline report records MTP four with a draft LM head. That does not independently identify the exact shortlist rows or verify which source implementation the binary used.

## Loading and startup

The pinned runtime uses positional reads, not mmap, for this artifact path. `InputFile` opens read-only and uses bounded `pread`; device staging uses aligned `O_DIRECT` reads. See `N:src/artifact/file_io.cpp:33-91`. Materialization allocates one device arena, assigns object ranges, sorts and coalesces source ranges, then cycles up to four pinned 64 MiB staging slots through `cudaMemcpyAsync` on the transfer stream. Events protect slot reuse. It uploads stored payload bytes without weight repacking. See `N:src/artifact/materializer.cpp:21-51,149-200,205-295`.

Our loader also creates one resident arena and streams raw objects into it with digest verification, `M:src/kernels/cuda/resident_weights.rs:25-49,169-182,245-265`. Binding enforces `safetensors-row-major-v1` at lines 80-95. `.mspec` reader uses seek/read and bounded verification, `M:src/artifact/reader.rs:96-107,160-191`. Improving pinned asynchronous upload or coalescing can reduce startup. It cannot explain steady-state prefill/decode measurements after full residency. No startup before/after measurement exists in this investigation.

## Ranked opportunities and isolating gates

Ranking reflects expected relevance from source and the existing FP8-dominated profile, not measured per-feature percentage gains.

1. Continue F02 native FP8 scheduling and numerical analysis on unchanged resident bytes. Compare the retained fast candidate and shorter accumulation chains against identical quantized real inputs. Preserve exact arithmetic separately. Measure raw error, BF16 boundary crossings, register/shared-memory resources, and projection latency before model teacher-forced logits and text quality. This most directly tests the current bottleneck; a new FP8 file layout adds no demonstrated benefit.
2. Integrate F03 paired projection/fused epilogue with existing separate pointers first. Compare separate versus fused launches on identical real gate/up inputs, retaining BF16 boundaries and independent references. Measure launches, intermediate traffic, operator latency, and full-model prefill/decode. Only then test an optional contiguous parent to isolate storage from fusion.
3. Test NVFP4 scale swizzling as a lossless resident-layout ablation. Reorder scale words once into a separate experimental buffer and add a separately identified consumer. Prove inverse permutation restores every source byte and exact logical scale coordinate; run independent irregular/signed fixtures and sanitizers. Compare natural versus tiled scale reads using the same arithmetic and tile schedule on gate/up and down shapes. Record load/repack time, extra resident bytes, and decode versus prefill separately. Do not credit changes in A4/A16 arithmetic or MMA scheduling to layout.
4. Qualify dedicated FP8 A16 decode as its own arithmetic profile. Row-major weights already support contiguous vector reads. Measure quantization cost saved and dot-product throughput, but compare teacher-forced outputs and state against exact A8 and an independent A16 reference. This is F01 integration, not a format conversion win.
5. Investigate optional draft-head/MTP quantization only after baseline artifact identity and runtime selection are known. Q8 MTP or shortlist Q4 draft changes can reduce draft work but alter acceptance. Require target-only agreement, rejection at every position, all-accepted state commits, and prose/code acceptance-aware end-to-end timing. Retain BF16 MTP control. F09 compact replay can proceed independently.
6. Optimize loader staging only when startup becomes a measured target. Preserve object hashes, identity rejection, and bounded memory; measure cold/warm load wall time, bytes read/transferred, staging memory, and ready-to-first-token separately. No steady-state gain should be claimed from a load-only change.

The strongest format-specific next experiment is item 3, lossless NVFP4 scale-layout ablation. The strongest next overall performance experiment remains item 1 because current measured cost and real-weight numerical evidence concern FP8. These are different priorities.

## Durable rule and reproduction

Treat container framing, persistent numeric values, GPU-resident layout, shared-memory layout, and arithmetic profile as separate variables. A packed extension is useful only when an identified consumer exploits it. Keep exact raw `.mspec` as the source/control; do not add a `.ninfer` parser or import Ninfer implementation.

Reproduce source lookup read-only with the pinned GitHub tree API and raw URLs, then line-number the named files. Compare local Rust references against HEAD `5298b42708088fa04e507ddcfd44f0e1f62402a4`. Expected finding is ordinary row-major FP8, offline NVFP4 scale swizzling, conditional fused parents, and direct upload. Those findings were observed in source. The recorded September 26 artifact SHA-256 matches the published artifact; current installed state, binary/source correspondence, logical-weight equivalence to our intake, selected dispatch, quality, and performance remain unverified by this investigation.
