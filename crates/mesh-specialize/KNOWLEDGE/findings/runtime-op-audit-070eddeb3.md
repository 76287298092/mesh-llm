# Runtime operation audit at 070eddeb3

Status: read-only source and evidence audit, 2026-09-27. No build, GPU run, SSH or
sanitizer was performed for this entry. Source pin
`070eddeb337a0e533758875069f9face5927c531`. The working tree also had an
uncommitted `KNOWLEDGE/optimizations/nvfp4-prefill-pipeline.md` edit and an
untracked `kernels/nvptx/nvfp4_prefill_large.rs`, which is not referenced from
`src/`. Paths are relative to `crates/mesh-specialize` unless they start with
`tools/`. **FACT** means read from source or evidence JSON. **INFERENCE** means
derived arithmetic, extrapolation or CUDA-documentation behavior not measured here.
All timings are per-launch synchronized CUDA event sums: "event ms", not wall time.

Evidence used:
- Decode, short context (33 positions) plus 32-row prefill:
  `KNOWLEDGE/evidence/iterate-20260927/a16-model-2/profile-python-exact.json`.
  It has 1,476 launches, 31.63 event ms and 40.67 ms unprofiled decode wall.
- 512-input prefill and 513-position decode:
  `.../nvfp4-wide-model-1/profile-512-baseline.json`. Prefill has 1,562.6 event ms.
  Decode has 52.40 event ms and 59.48 ms wall. Also used: `profile-512-wide.json`,
  `attention-prefill-1/profile-128-{exact,online}.json` and
  `native-prefill-ablation-2/profile-native-prefill.json`.

## 1. Execution path map

| Step | Source | Notes |
| --- | --- | --- |
| CLI dispatch | `tools/xtask/src/specialize.rs:18-25` | Subcommands: `qwen-model-bench`, `qwen-model-profile`, `qwen-model-check` (vs CPU reference), `qwen-model-reference` (CPU, 1..17 tokens), `qwen-mtp-check`, `mlp-workspace-check` |
| Bench | `tools/xtask/src/specialize/model_bench.rs:12-59` | Fixed flag order: `--artifact P --tokens COMMA_IDS --output-tokens 2..512 --repetitions 1..3 --ptx P --device N --output NEW_FILE` |
| Profile | `tools/xtask/src/specialize/model_profile.rs:12-47` | Same inputs without output count or repetitions, plus optional trailing `--teacher-token ID` |
| Token ingress | `model_bench.rs:51-54`, `resident_embedding.rs:69-70` | **FACT:** Tokens arrive as raw comma-separated IDs. There is no tokenizer in `src/` or xtask. `tokenizer.json`/`chat_template.jinja` are only packed into the artifact (`src/checkpoint/qwen3_8_recipe.rs:58-63,243`). Evidence prompts were rendered externally with Python `tokenizers` 0.22.2 and Jinja; see `a16-model-2/prompts.json` |
| Config | `src/packages/qwen3_8_27b/model_benchmark.rs:18-26`, `decoder.rs:8-52`, `schedule.rs:46-103` | Capacity is `prompt + outputs - 1`. Layer `i % 4 == 3` is full attention, others GDN. `i < 56` uses NVFP4 MLP, otherwise FP8. That gives 42 GDN+NVFP4, 6 GDN+FP8, 14 Attn+NVFP4 and 2 Attn+FP8 layers |
| Timed step | `src/kernels/cuda/resident_model_bench.rs:398-421` | `cuCtxSynchronize`, `Instant`, `forward_selected` and `cuCtxSynchronize` |
| Forward | `src/kernels/cuda/resident_model.rs:263-396` | Cursor transaction (287), embedding (288), 64 blocks in order (291-358), head (359-366), selection (372-387) and commit (388) |
| GDN block | `resident_gdn.rs:160-232` | norm, QKV FP8, Z FP8, A BF16, B BF16, conv, core, out FP8, residual+norm, MLP, residual add |
| GDN core | `resident_gdn_core.rs:79-119` | `gdn_qk_norm` (256-288), `gdn_gates` (290-333), `gdn_recurrent` (335-386) and `gdn_gated_rms_norm` (388-430), each launched then synchronized (439-471) |
| Attention block | `resident_attention.rs:149-244` | Host RoPE tables plus two uploads (166-168, 258-266), norm, Q (12,288 channels including gate half), K, V, Q/K prepare, KV append and attention, sigmoid gate, O, residual+norm, MLP, residual add |
| MLP | `resident_mlp.rs:101-119`, workspace variant `63-99` | gate, up, `mlp_silu_product` and down as separate projections |
| Head | `resident_head.rs:38-58,81-100` | Copies the last hidden row, norm, FP8 248,320x5,120 projection. The A16 head variants are profile-selected |
| Selection | `resident_model.rs:386,399-420` | Default: DtoH of 496,640 B of BF16 logits, then CPU `sampling::greedy`. Opt-in GPU: `resident_greedy.rs:39-112` (2 launches, 1 sync, 16 B read) |

Environment variables (**FACT**). Each is read once per process through `OnceLock`
unless marked per call.

| Variable | Default | Values and effect | Source |
| --- | --- | --- | --- |
| `MESH_SPECIALIZE_FP8_PROFILE` | `exact` | `a16-decode` uses the one-row BF16-activation FP8 GEMV for decoder and head. `a16-head` and `a16-head-gemv` affect only the head. `native-prefill[-short][-audit]` use a native E4M3 MMA for FP8 rows >=16. Bench rejects `*-audit` | `src/kernels/fp8_profile.rs:49-78`; dispatch `resident_fp8.rs:92-123`; head `resident_head.rs:88-99` |
| `MESH_SPECIALIZE_NVFP4_PROFILE` | `baseline` | `tiled-prefill` (32x32) or `wide-prefill` (32x128) for rows 16..=512. M=1 always uses `nvfp4_decode_exact` | `src/kernels/nvfp4_profile.rs:28-73` |
| `MESH_SPECIALIZE_ATTENTION_PROFILE` | `exact` | `online` uses `attention_online_bf16` (FP32 online softmax). `online-audit` is rejected by bench | `src/kernels/attention_profile.rs:29-49` |
| `MESH_SPECIALIZE_GPU_GREEDY` | `off` | `on` uses device argmax | `resident_model.rs:137-144`, `model_greedy.rs:5-16` |
| `MESH_SPECIALIZE_MLP_WORKSPACE` | `off` | `on` shares one model arena across MLP chains. It requires exact decoder arithmetic and split-K off | `model_workspace.rs:8-34`; `resident_gdn.rs:211-216` |
| `MESH_SPECIALIZE_FP8_SPLIT_K` | `off` | `2`/`4`/`8`/`16` only for rows 4..15 with channels <16,384 (MTP verify) | `resident_fp8_splitk.rs:9-32` |
| `MESH_SPECIALIZE_MTP_RECOVERY` | `full-forward` | `compact` (per call) | `resident_speculation.rs:487-493` |
| `MESH_SPECIALIZE_LOGIT_DUMP_DIR` | unset | Profile writes 4 BF16 logit vectors plus a manifest | `resident_logit_dump.rs:11-45` |
| `MESH_SPECIALIZE_PARTITION_AUDIT` | unset | `1` downloads every layer hidden in profile | `resident_model_profile.rs:441-449` |
| `MESH_SPECIALIZE_PARTITION_STAGE_LAYER` | unset | Layer index for stage capture in profile | `partition_stage_audit.rs:16-25` |
| `MESH_SPECIALIZE_NVFP4_AUDIT` | unset | `1` runs the layer-22 down-projection audit (profile only; bench rejects) | `nvfp4_projection_audit.rs:22-28`; `resident_model_bench.rs:52-63` |

## 2. Per-operation table, default exact profile

**FACT.** `Function::launch` always uses the null (default) stream
(`driver.rs:766-815`, stream argument `ptr::null_mut()` at 806). Every `Buffer::new`
is a `cuMemAlloc_v2` (`driver.rs:448-469`). Every drop is a `cuMemFree_v2`
(`driver.rs:565-577`). Uploads and downloads are synchronous `cuMemcpyHtoD_v2`/`DtoH_v2`
(`driver.rs:512-562`). DtoD is `cuMemcpyDtoD_v2` (`485-510`). Every
`module.function(name)` call runs `cuModuleGetFunction` with no cache
(`driver.rs:691-709`), so the number of lookups equals the number of launches. "Sync" below means
`cuCtxSynchronize` (`driver.rs:419-426`).

Allocations are freed when the Rust owner drops, within the op or at the end of
the layer. The alloc count therefore equals the free count. Grids are from the
evidence JSON. Decode means M=1; prefill means M=512.

| Op | Host function | Kernel source (`kernels/nvptx/`) | Decode grid x block | Prefill grid | Arithmetic | Alloc | Copies | Sync | Launch |
| --- | --- | --- | --- | --- | --- | --: | --- | --: | --: |
| Input/first norm | `resident_embedding.rs:52-108` | `embedding_norm.rs` | 1x256 | 512 | BF16 gather, FP32 RMS | 4 | 1 HtoD (IDs) | 1 | 1 |
| RMSNorm (block input, head) | `resident_norm.rs:55-116` | `embedding_norm.rs` (row-ID gather) | 1x256 | 512 | BF16 in, FP32 norm, BF16 out, plus residual copy and FP32 raw | 4 | 1 HtoD (row IDs) | 1 | 1 |
| FP8 projection | `resident_fp8.rs:74-197` | `fp8_quantize.rs` and `fp8_linear_exact.rs` (M<4) or `fp8_prefill_exact.rs` (M>=16) | quant 1x256; linear ceil(N/4)x128 | quant 512; linear [N/8, 32]x32 | E4M3 W (per-channel BF16 scale) times E4M3 A (per-row FP32 scale, software RNE search). Decode: warp per column, exact integer dot, i64. Prefill: 3 signed base-128 digits per operand, **9 INT8 MMA per K32**, exact | 4 | none | 1 | 2 |
| BF16 A/B (48 ch) | `resident_bf16.rs:42-96` | `bf16_linear_decode.rs` | 12x128 | [12, 512]x128 | BF16 by BF16, **FP64** warp dot | 2 | none | 1 | 1 |
| Causal conv4 | `resident_conv.rs:40-98` | `causal_conv4.rs` | 40x256 | 20,480 | BF16, FP32 conv and SiLU | 4 | 1 DtoD (history to state) | 1 | 1 |
| GDN Q/K norm | `resident_gdn_core.rs:256-288` | `gdn_prepare.rs` | 16x256 | 8,192 | FP32 | 2 | none | 1 | 1 |
| GDN gates | `resident_gdn_core.rs:290-333` | `gdn_prepare.rs` | 1x256 | 96 | FP32 | 3 | none | 1 | 1 |
| GDN recurrence | `resident_gdn_core.rs:335-386` | `gdn_recurrent.rs:110-199` | **48x128** | **48** (serial over rows) | FP32 state `[48,128,128]` in global memory, 2 read and 2 write passes per row | 2 | none | 1 | 1 |
| GDN gated norm | `resident_gdn_core.rs:388-430` | `gated_rms_norm.rs` | 48x256 | 24,576 | FP32 | 5 | none | 1 | 1 |
| Residual+norm | `resident_norm.rs:119-180` | `residual_norm.rs` | 1x256 | 512 | BF16 sum, FP32 norm | 3 | none | 1 | 1 |
| NVFP4 projection | `resident_nvfp4.rs:83-155,210-258` | `nvfp4_quantize.rs` and `nvfp4_decode_exact.rs` (M=1) or `nvfp4_linear.rs` (M>=2) | quant K/16x32; linear ceil(N/4)x128 | quant M*K/16 (163,840 or 557,056)x32; linear [N/8, 32]x32 | E2M1 plus UE4M3/16 plus global, both operands. Decode: exact integer i64 per warp. Prefill: `mma.m16n8k64 mxf4nvf4 block_scale` FP32 accumulate | 5 | none | 1 | 2 |
| SiLU product | `resident_activation.rs:12-59` | `mlp_activation.rs` | 68x256 | 34,816 | BF16 and FP32, writes 4 outputs | 4 | none | 1 | 1 |
| Residual add | `resident_norm.rs:182-216` | `residual_add.rs` | 20x256 | 10,240 | BF16 | 1 | none | 1 | 1 |
| RoPE tables | `resident_attention.rs:166-168` | none (host `engine/rope.rs:43`) | none | none | host FP64 tables | 2 | 2 HtoD | 0 | 0 |
| Q/K prepare (x2) | `resident_attention_prepare.rs:67-151` | `attention_prepare.rs` | Q 24x256, K 4x256 | 12,288 / 2,048 | norm, RoPE, gate split | 4 each | none | 1 each | 1 each |
| KV append and attention | `resident_attention_core.rs:46-112` | `causal_attention.rs` | append 4x256; attention **24x256** | 2,048; 12,288 (row x head) | BF16 KV, **FP64** scores and softmax, one CTA per (row, q-head), serial over keys | 2 | none | 1 | 2 |
| Attention gate | `resident_attention_gate.rs:9-100` | `attention_gate.rs` | 24x256 | 12,288 | sigmoid in FP32 | 4 | none | 1 | 1 |
| Head | `resident_head.rs:38-58` plus norm plus FP8 | as above | FP8 62,080x128 | same (last row only) | exact FP8 | 1+4+4 | 1 DtoD, 1 HtoD | 2 | 3 |
| CPU select | `resident_model.rs:399-420` | none | none | none | host argmax | 0 | **1 DtoH, 496,640 B** | implicit | 0 |

Per-layer and per-forward totals (**FACT**, counted from the table). Decode and
prefill counts are identical; only extents differ.

| Unit | Launches | Allocs=frees | ctx syncs | HtoD | DtoD | Event ms decode (short ctx) |
| --- | --: | --: | --: | --: | --: | --: |
| GDN+NVFP4 layer (x42) | 23 | 59 | 17 | 1 | 1 | ~0.472 |
| GDN+FP8 layer (x6) | 23 | 56 | 17 | 1 | 1 | ~0.525 |
| Attn+NVFP4 layer (x14) | 23 | 59 | 15 | 3 | 0 | ~0.478 |
| Attn+FP8 layer (x2) | 23 | 56 | 15 | 3 | 0 | ~0.530 |
| Whole forward | **1,476** | **3,765** | **1,059** (+2 in `timed_forward`) | 98 | 49 | 31.63 |

The event column is reconstructed from group averages and sums to 31.64 ms
(INFERENCE). Measured launch count 1,476 matches the source count exactly. With
`MLP_WORKSPACE=on`, each MLP goes from 19 or 16 allocs and 4 syncs to 1 alloc
(`copy_region`, `resident_workspace.rs:87-101`) and 1 sync. That gives about 2,637
allocs and 867 syncs per forward. The arena is reallocated whenever rows change
between prefill and decode (`model_workspace.rs:41-55`).

## 3. Measured attribution and efficiency

Decode weight bytes per token (**INFERENCE** from shapes): FP8 is N·K bytes plus
scales. NVFP4 is N·K/2 + N·K/16. GDN QKV is 10,240x5,120, Z is 6,144x5,120, out is
5,120x6,144. Attention Q is 12,288x5,120, K and V are 1,024x5,120 each, O is
5,120x6,144. MLP is 17,408x5,120 (gate, up) and 5,120x17,408 (down). The BF16
embedding (2.54 GB) is not streamed. Total streamed is 19.10 GB per token, so the
1.79 TB/s floor is **10.67 ms**. That makes **93.7 tok/s the one-token-per-forward
ceiling**. Ninfer's recorded 164-201 tok/s used MTP4 and FP8 KV
(`optimizations/decode-projections.md:115`).

| Decode class (short ctx) | Launches | Event ms | Weight MB | GB/s | % of 1.79 TB/s |
| --- | --: | --: | --: | --: | --: |
| NVFP4 MLP (`nvfp4_decode_exact`) | 168 | 8.243 | 8,423 | 1,022 | 57% |
| GDN QKV+Z FP8 | 96 | 3.857 | 4,030 | 1,045 | 58% |
| FP8 N=5,120 out (GDN out, attn O, FP8 down) | 72 | 2.604 | 2,728 | 1,048 | 59% |
| Attention Q/K/V FP8 | 48 | 1.448 | 1,175 | 812 | 45% |
| FP8 MLP gate/up | 16 | 1.130 | 1,427 | 1,263 | 71% |
| FP8 head | 1 | 0.868 | 1,272 | 1,466 | 82% |
| GDN A/B BF16 | 96 | 2.698 | 47 | 17 | **1%** |
| **All weight kernels** | 497 | **20.85** | 19,102 | 916 | **51%** |
| `fp8_quantize_bf16` (1 CTA each) | 233 | 2.607 | n/a | n/a | n/a |
| `gdn_recurrent` | 48 | 2.483 | state 302 MB minimum r+w | ~122 minimum-traffic | 7% |
| `causal_attention_bf16` (33 keys) | 16 | 1.526 | n/a | n/a | n/a |
| norms (residual+first) | 130 | 1.925 | n/a | n/a | n/a |
| other glue (NVFP4 quant, SiLU, conv, gated norm, add, prep, gates) | 552 | 2.25 | n/a | n/a | n/a |
| **Total events / wall** | 1,476 | 31.63 / **40.67** | n/a | n/a | n/a |

- **Correction, 2026-09-28:** the 9.0 ms numerical difference between unprofiled
  wall time and separately instrumented event sums is not host/driver attribution.
  Profiling changes execution. No per-call host costs were measured here.
- **Decode at 513 positions** (`profile-512-baseline.json`): attention rises to
  **22.06 ms of 52.40 event ms**, wall 59.48 ms. KV read is 512 × 4 KiB × 16 layers
  = 33.5 MB, about 1.5 GB/s, roughly 0.1% of bandwidth. All other classes are
  unchanged within 0.2 ms.
- Online attention at 128 context: 1.52 vs 5.63 ms exact. It still uses 24 CTAs.
- **INFERENCE** (linear in context): exact decode attention is about 350 ms per token at 8K and about 5.6 s at 128K.

| Prefill M=512 class | Event ms | % of 1,562.6 | Work | Achieved |
| --- | --: | --: | --: | --: |
| `fp8_prefill_exact` (all FP8 projections) | 724.7 | 46.4% | 9.58 TFLOP | **13.2 TFLOPS** (9x INT8 expansion) |
| `causal_attention_bf16` (FP64) | 243.9 | 15.6% | 0.052 TFLOP (QK+PV, causal) | 0.21 TFLOPS |
| `gdn_recurrent` (48 CTAs, serial) | 237.6 | 15.2% | 0.116 TFLOP (recurrence form) | 0.49 TFLOPS |
| `nvfp4_linear` gate/up | 98.9 | 6.3% | 10.22 TFLOP | 103 TFLOPS |
| `nvfp4_linear` down (K=17,408) | 96.9 | 6.2% | 5.11 TFLOP | 53 TFLOPS |
| `bf16_linear_decode` A/B | 59.6 | 3.8% | 0.024 TFLOP | 0.4 TFLOPS |
| `mlp_silu_product` | 39.0 | 2.5% | ~9.1 GB r/w incl. diagnostics | ~234 GB/s |
| NVFP4 quantize, conv, gated norm, FP8 quantize | 56.2 | 3.6% | n/a | n/a |
| Everything else | 5.8 | 0.4% | n/a | n/a |

- Wall: 512/305.6 tok/s is about 1.675 s in the cited benchmark. Its difference
  from instrumented event sums cannot quantify host overhead.
- Comparators (FACT, measured):
  - Native FP8 at 128 rows: 43.6 ms vs exact 212.7 ms (about 55 TFLOPS).
  - Wide NVFP4 at 512 rows: 101.4 ms vs 195.8 ms.
  - Online attention at 128 rows: 3.94 ms vs 16.89 ms.
- **INFERENCE:** Spec-sheet dense FP8 and FP4 tensor peaks for the RTX 5090 are
  in the hundreds of TFLOPS and above. They were not verified here. No prefill
  class is near a compute limit. `fp8_prefill_exact` uses 32-thread CTAs with no
  shared-memory staging, so weights are refetched per 16-row tile (32 times at M=512).

## 4. Structural blockers (FACT unless marked)

1. **Per-op device allocation and free.** There are 3,765 `cuMemAlloc`/`cuMemFree`
   pairs per decode forward (section 2). Every output owns a fresh `Buffer`,
   including FP32 "unrounded" and diagnostic regions that are never consumed:
   norm residual copies, SiLU and sigmoid FP32 values, gated-norm intermediates,
   and the unused K-prepare gate (`resident_attention_prepare.rs:101-104`).
   `cuMemFree` may implicitly synchronize (INFERENCE from CUDA semantics).
2. **Host round trips.** There are 1,059 `cuCtxSynchronize` calls per forward:
   one after every op (for example `resident_fp8.rs:176`, `resident_nvfp4.rs:155`,
   `resident_gdn_core.rs:469`). There are also 98 synchronous HtoD copies (norm
   row IDs every call `resident_norm.rs:76-78`, RoPE cos/sin every attention
   layer `resident_attention.rs:166-168`) and a DtoH of all logits
   (`resident_model.rs:404-405`).
3. **Default stream only.** `driver.rs:766-815`. No overlap, and events are only
   default-stream (`driver.rs:839-875`).
4. **Per-launch `cuModuleGetFunction`** (`driver.rs:691-709`) and a host `Vec` of
   kernel arguments built on every launch.
5. **Host-value positions.** `past`/`capacity` are passed by value into
   append/attention kernels (`resident_attention_core.rs:46-112`), and RoPE tables
   are built on the host per step. A captured graph cannot advance position
   without device-position kernel variants
   (`ordinary-decode-graph-plan.md`, table row for `resident_attention_core.rs`).
6. **State copy.** Conv history writes to a scratch buffer, then a separate DtoD
   copies it into state (`resident_conv.rs:66,96`).
7. **CPU selection.** CPU selection is the default. GPU greedy exists behind an
   env var and still downloads 16 B and syncs (`resident_greedy.rs:92-94`).
8. **Full-state forks.** `ResidentState::fork` allocates the whole arena and copies
   it DtoD (`resident_state.rs:36-51`). At capacity C that is 156 MB + 64 KiB·C.
   MTP calls fork per round (`resident_speculation.rs:296,343`; `resident_mtp.rs:61-62`).
9. **Capture guards reject all of the above.** `driver_graph.rs:120-142` blocks
   allocation, sync, lookup and default launch during capture, but `Buffer::drop`
   frees without a guard (`driver.rs:565-577`). The current forward therefore
   cannot be captured (`ordinary-decode-graph-plan.md` "What the current call
   chain prevents").

State of the prepared explicit-stream path:
- `device_view.rs` provides checked borrowed `DeviceRead`/`DeviceWrite`. Per its
  header (`device_view.rs:1-6`) these are not leases.
- `mlp_prepared_projection.rs:47-221` pre-resolves quantize/linear functions and
  enqueues with `launch_on_stream` (`driver_graph.rs:487-538`) and stack
  arguments. `mlp_prepared_chain.rs:19-110` chains gate, up, activation and down.
- `resident_workspace.rs` supplies the arena, stream step, `complete`/drain
  (`175-196`) and `copy_region` (allocates, 87-101).
- `driver_graph.rs` has Stream, capture, instantiate and launch. It was only
  replayed on a 257-element residual add (`feature_graph_trial.rs`).
- The prepared chain is called **only** from
  `resident_mlp_workspace.rs:139-171` (`prepared_trial`), reached through
  `mlp_workspace_trial.rs:94,139,147` and xtask `mlp-workspace-check`.
- The model's opt-in workspace path (`resident_mlp.rs:63-99` to
  `resident_mlp_workspace.rs:191-254`) still uses default-stream `launch`, one ctx
  sync per chain, and an allocating `copy_region`. It is also capped at 512 rows (209).
- No norm, GDN, attention, head or embedding op has a prepared or stream variant.
- **No part of the model forward is captured today.**

## 5. Harness and measurement capability

| Limit | Value | Enforced at |
| --- | --- | --- |
| Bench prompt | 1..=512 IDs | `packages/qwen3_8_27b/model_benchmark.rs:12-17`; `resident_model_bench.rs:189-192` |
| Bench outputs | 2..=512 (EOS ignored) | same files, `:14` and `:193-196` |
| Bench capacity | <=2,048, and set to prompt+outputs-1. The 512 caps bind first | `resident_model_bench.rs:185-188,205-212` |
| Repetitions | 1..=3 plus one 1-token warmup session | `resident_model_bench.rs:197-200,290-312` |
| Profile prefix | <=512, capacity exactly prefix+1 (<=513) | `resident_model_profile.rs:18-19,318-342`; `packages/.../model_profile.rs:13-20` |
| Rows per forward | <=2,048 (cursor and every op) | `engine/session.rs:6`; `resident_fp8.rs:11`; `resident_nvfp4.rs:14`; `resident_bf16.rs:12`; `resident_gdn_core.rs:12`; `resident_attention_core.rs:9`; `resident_conv.rs:50`; `engine/rope.rs:44` |
| MLP workspace rows | <=512 | `resident_mlp_workspace.rs:209` |
| Tiled/wide NVFP4 | rows 16..=512, else falls back to `nvfp4_linear` | `nvfp4_profile.rs:31-45` |
| Context capacity | <=262,144 | `engine/session.rs:5`; `schedule.rs:47-50`; `resident_attention.rs:158-165` |
| All-row logits | <=17 rows | `resident_model.rs:431-436` |
| Admission reserve | weights + state + 1 GiB | `resident_model_bench.rs:18,216-227` |

- **Prefill is one-shot.** The whole prompt goes into one forward
  (`resident_model_bench.rs:332-333`). There is no chunk loop. The model supports
  chunked calls structurally: the cursor transaction commits `past`, attention
  appends at `past`, and GDN state persists. Whole/chunk/token equivalence has
  prior evidence (`findings/resident-decoder.md`).
- **8K-128K needs all of the following (INFERENCE):**
  1. Lift the 512/513 harness caps.
  2. Add a <=2,048-row chunk loop, or raise every `MAX_ROWS`.
  3. KV is BF16, 64 KiB per token across 16 layers (`schedule.rs:13,236`). That is
     0.54 GB at 8K and 8.59 GB at 128K. Weights (21.65 GB) + state (0.154 GB) + 128K KV
     + 1 GiB reserve = 31.47 GB against 32.22 GB free observed. The margin is too
     small for per-op transients at 2,048 rows (one FP32 MLP unrounded buffer is
     143 MB). FP8 KV (`kernels/nvptx/kv_fp8.rs`) is unwired.
  4. Attention must be replaced. It is O(n²) prefill with one CTA per (row, head)
     and FP64, and O(n) decode on 24 CTAs. Extrapolated exact prefill attention
     is about 62 s at 8K.
  5. GDN must be chunked. Recurrence is serial over rows at about 4.95 ms per layer
     per 512 rows, so about 3.8 s per 8K prefill. `gdn_chunked.rs` is a candidate
     that is not connected (`optimizations/feature-f04-gdn.md:3-5`).
- **Chat templates:** not supported in-repo. Tokenization is external Python.
- **Teacher-forced logits:** `qwen-model-profile --teacher-token` gives one forced
  continuation. `LOGIT_DUMP_DIR` exports 4 full BF16 logit vectors (whole/token
  prefill and decode). `engine/logit_quality.rs` computes KL/TV per vector.
  **There is no per-position logprob capture over a fixed text beyond 17 rows**
  (`forward_detailed(All)` cap), and no harness loop for it. Options: raise the
  cap and chunk head rows, or drive token-by-token decode and keep the logits.
- **Timing reported:**
  - `prefill_seconds` and input tok/s: one forward, including head, the
    logit DtoH and CPU argmax of the first token. This is TTFT-like, excluding
    tokenization and transport.
  - Per-step `decode_intervals`, and `decode_tokens_per_second` = (outputs-1)/sum.
  - Memory checkpoints.
  - No median or percentile in-tool; scripts compute medians.
  - The warmup does not exercise the 512-row shapes (**FACT**,
    `resident_model_bench.rs:298-301`).

## 6. Arithmetic profiles

| Op / phase | Kernel | Status | Evidence (one line) |
| --- | --- | --- | --- |
| FP8 decode M=1 | `fp8_linear_exact` | **default** | Exact vs CPU FP64 oracle. Full model bit-exact hidden, logits and state (`findings/resident-model.md`) |
| FP8 decode M=1 | `fp8_a16_decode` | opt-in `a16-decode` | Events 31.63 to 28.60 ms. Medians 24.2 to 26.4-26.7 tok/s. KL <=0.0021 at 2 positions. Prose diverges at token 8. Not promoted (`feature-f01-decode.md:131-156`) |
| FP8 head | `fp8_a16_head` / GEMV | opt-in | No decode gain (25.54/25.62/25.41 tok/s), not promoted (`a16-head-schedule.md:174-183`) |
| FP8 M=4..15 | `fp8_linear_exact4`, `fp8_verify_exact`, split-K | MTP verify only | Exact. Split-K off by default |
| FP8 prefill M>=16 | `fp8_prefill_exact` | **default** | Exact 9xINT8 MMA. 256.6/295.2 tok/s at 128/512 |
| FP8 prefill M>=16 | `fp8_prefill_native[_short]` | opt-in | 393.0/444.8 tok/s. Layer-0 differences 108-251 BF16 codes per projection. Partition check fails. Greedy token can change. Not promoted (`feature-f02-prefill.md:125-182`) |
| NVFP4 decode | `nvfp4_decode_exact` | **default, all profiles** | Exact integer |
| NVFP4 prefill | `nvfp4_linear` | **default** | Hardware mxf4nvf4 FP32 accumulate. Strict 128-row partition fails at the NVFP4 rounding boundary under online attention (`README.md`) |
| NVFP4 prefill | `nvfp4_prefill_wide` / `_tiled` | opt-in | Wide: 290.5/324.2 vs 274.6/305.6 tok/s. **Logits and state exact across schedules** at 128/512 (`nvfp4-prefill-pipeline.md:304-321`) |
| Attention | `causal_attention_bf16` (FP64) | **default** | Exact reference-grade |
| Attention | `attention_online_bf16` (FP32) | opt-in | 4.3x prefill and 3.7x decode kernel speed at 128. Strict 128-row partition fails. Two-answer smoke check only (`feature-f05-attention.md`) |
| GDN | `gdn_recurrent` | only path | Exact vs scalar. Chunked and replay variants not in forward |
| Selection | CPU argmax / `greedy_bf16_*` | CPU default | GPU exact-match, <1% speed (`gpu-greedy.md:147`) |
| MLP scratch | per-op alloc / arena | alloc default | Arena +5.8% decode, +8.4%/+3.8% prefill 128/512, exact (`feature-f07-workspace.md:93-116`) |

## 7. Top rebuild candidates

Historical candidate ranking from source and event profiles. These event costs
are prioritization clues, not removable wall-time bounds. New model measurements
in `../optimizations/stream-forward.md` supersede the execution-path speculation.

| Rank | Candidate | Removable time (derivation) | Cost / risk | Files |
| --: | --- | --- | --- | --- |
| 1 | Promote wide NVFP4 prefill | 94 ms of prefill at 512 (195.8 to 101.4), about 5.6% of wall. Outputs are already bit-exact vs baseline | Trivial. Needs an M>512 fallback check | `src/kernels/nvfp4_profile.rs:57` default |
| 2 | Replace BF16 A/B `bf16_linear_decode` (FP64, 12 CTAs) | Decode 2.70 of 40.7 ms (6.6%): 47 MB should take 0.03 ms. Prefill 59.6 ms (3.6%). Fold the 96 channels into the QKV/Z pass or use a tiled BF16 kernel | Low. Arithmetic changes from FP64 to FP32, so it needs a tolerance decision | `resident_bf16.rs`, `kernels/nvptx/bf16_linear_decode.rs`, `resident_gdn.rs:179-182` |
| 3 | Tensor-core flash attention: prefill tiles plus split-sequence decode | Prefill 243.9 ms (14.6%). Decode is small at short context but 22.06 of 59.5 ms (37%) at 513 positions, and it dominates at 8K+ (about 350 ms per token). FLOP floor below 1 ms | Medium. Replaces FP64 exactness, as online already does | `kernels/nvptx/causal_attention.rs`, `attention_online.rs`, `resident_attention_core.rs` |
| 4 | Real tiled FP8 prefill GEMM (shared-memory pipeline, large tiles) | 724.7 ms (43% of wall). Native already cuts the class about 4.9x (212.7 to 43.6 ms at 128). At 55 TFLOPS the floor is 174 ms; at 200 TFLOPS it is 48 ms | **Policy gate:** exact 9xINT8 cannot be fast. Native arithmetic needs a quality basis first | `resident_fp8.rs:111-123`, `kernels/nvptx/fp8_native_prefill.rs` |
| 5 | Prepared single-stream decode plus graph replay | Host share unmeasured. Stable storage, fewer waits and prepared submission require a whole-model ablation; event sums cannot bound the gain. Arena alone measured +5.8% | High: every `resident_*.rs` op, `resident_model.rs`, device-position attention, `driver.rs` function cache. Design in `ordinary-decode-graph-plan.md` | as listed in section 4 |
| 6 | Faster weight-streaming GEMV (wide loads, more bytes in flight, fused quantize) | 20.85 ms at 51%. At 85% of peak, 12.6 ms, saving about 8.3 ms (20% of decode). Worst classes: attention QKV 45%, NVFP4 57% | Medium. Keep exact integer NVFP4. FP8 A16 is a numerics change | `nvfp4_decode_exact.rs`, `fp8_linear_exact.rs`, `resident_fp8.rs`, `resident_nvfp4.rs` |
| 7 | Chunked GDN prefill plus a wider decode recurrence | Prefill 237.6 ms (14.2%). Decode 2.48 ms vs 0.17 ms state-traffic floor (6% of decode). Mandatory for long prompts | Medium-high. Candidate kernels exist but are unwired | `resident_gdn_core.rs:335-386`, `kernels/nvptx/gdn_recurrent.rs`, `gdn_chunked.rs` |
| 8 | Fuse glue: quantize into norm, drop diagnostic writes, conv writes state in place | Decode about 5.1 ms (FP8 quantize 2.61, norms 1.93, NVFP4 quantize 0.62). Prefill about 95 ms (SiLU 39.0, quantizes 28.4, conv 17.3, gated norm 10.5) | Medium. Diagnostic outputs are part of current audit tooling | `resident_norm.rs`, `resident_activation.rs`, `resident_conv.rs`, quantize kernels |

No combined speed forecast is established. The former 65 tok/s decode and
1.7–3.4K tok/s prefill estimates added hypothetical component improvements and
unsupported host-gap estimates; they are withdrawn. Use isolated whole-model
measurements instead. The weight-traffic calculation above is an idealized
bandwidth model, not a measured performance ceiling.

## Could not determine

- The exact spec-sheet FP8 and FP4 tensor peaks for this SKU were not measured or verified.
- Whether `cuMemFree` or `cuMemAlloc` serialize in this driver (13.04): no trace exists.
- Host gap split across alloc, free, sync, lookup and HtoD: no host profile exists.
- The 128K memory headroom with per-op transients was not tested.
- No uncontended (ComfyUI-free) timing exists; all evidence is shared-GPU.
