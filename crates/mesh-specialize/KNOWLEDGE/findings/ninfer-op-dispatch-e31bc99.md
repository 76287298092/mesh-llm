# Ninfer per-operation dispatch for Qwen3.8-27B NVFP4 at e31bc99

Status: read-only source trace, September 27, 2026. Ninfer commit `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`
(the source of the running binary). Configuration traced: `--kv-dtype fp8 --max-concurrency 2
--prefill-chunk 2048 --spec mtp --draft-tokens 4 --lm-head-draft`, CUDA graphs on (default,
`src/serve/serve_options.h:50`). No build, GPU run, or measurement was made. Byte and launch counts are
source arithmetic.

**Legend.** FACT = read in the executed path, with the dispatch condition traced. INFERENCE = derived,
or depends on artifact contents that could not be read. Paths are relative to
`target/specialize/ninfer-deep-dive/`. Abbreviations: `TX` = `src/models/qwen3_5/execution/text.cpp`,
`LF`/`LN` = `src/ops/linear/{fp8,nvfp4}`, `LA` = `src/ops/linear_add`, `LS` = `src/ops/linear_swiglu`,
`AI` = `src/ops/attn_input_proj/fp8`, `GI` = `src/ops/gdn_input_proj`, `GDN` =
`src/ops/linear_attention/gated_delta_net`, `ATT` = `src/ops/softmax_attention/dense/causal_cache`,
`LC:` = the untracked local converter under `target/specialize/reassess-20260927/ninfer-local-convert/`.

**Workload A does not run in this service.** With `--spec mtp`, every decode round is an MTP round
(`src/models/qwen3_5/program/decode.cpp:778-784`). Its width is always `draft_window+1 = 5`, even when the
proposal extent is 0 (`decode.cpp:505-506`, `program/speculative/mtp.cpp:81`). Workload A below describes
`--spec none` (FACT). The layer code in A is also the M=1 draft and serves as a reference.

## 1. Weight format inventory and bytes read per token

**Provenance.** `LC:convert_nvfp4_nvidia.py:1-7` and `LC:modelopt_source.py:3-17` state the format
assignment: NVFP4 for the MLP of layers 0-55 and row-scaled FP8 for everything else. NVIDIA's NVFP4 MLP 56-63
and output head are re-encoded from BF16 with the row-scaled FP8 encoder (`modelopt_source.py:96-106,188-202`).
Every other NVIDIA FP8 matrix has a **per-tensor** scalar scale, broadcast to a per-row BF16 column
(`modelopt_source.py:107-111,136-147`). Its FP32 scalar is rounded to BF16 at line 144, so the claimed "exact
re-expression" (line 14) does not hold exactly for that scale. The recipe modules (`recipe_nvfp4`,
`convert_nvfp4`, `fp8_embedding`) are absent. Therefore the rows below also rely on the in-tree official recipe
(`tools/convert/official_recipes.py:151-174`) and on whether the loader would accept the format.

| Weight class (per layer) | Logical shape | Stored format | Evidence |
|---|---|---|---|
| Attn Q,K,gate,V (16 layers) | joined [14336,5120] (6144+1024+6144+1024) | FP8 E4M3 row-major + BF16 row scale; one contiguous parent; `AllowA8` | FACT: `src/ops/weight_input.cpp:115-141` accepts only a contiguous NVFP4/FP8/BF16 parent. A non-contiguous parent must be Q4+Q5 (`:142-147`). Recipe `official_recipes.py:167-173` |
| Attn O | [5120,6144] | FP8 row | same |
| Attn q_norm/k_norm | [256] | BF16 direct | `src/models/qwen3_5/load/text.cpp:18-19` |
| GDN Q,K,V,Z (48 layers) | joined [16384,5120] (2048+2048+6144+6144) | FP8 row, single parent, `AllowA8` | FACT: `weight_input.cpp:122-124,183-187` |
| GDN A,B | 2×[48,5120] | **BF16** (kept high-precision) | FACT: `weight_input.cpp:189-206` requires BF16 |
| GDN conv | [4,10240] | BF16 direct | `load/text.cpp:50-51`; read as bf16 in `GI/fp8/fp8_gdn_conv_fused.cu` |
| GDN A_log, dt_bias | [48] | FP32 | `load/text.cpp:48-49` |
| GDN out | [5120,6144] | FP8 row | recipe |
| MLP gate+up, layers 0-55 | joined [34816,5120] | NVFP4: E2M1 codes, E4M3 scale per K16 in the m128x4 tiled plane, FP32 weight divisor, separate static activation divisor; `AllowA4` | `weight_input.cpp:80-112,208-213`; `LC:modelopt_source.py:70-95` |
| MLP down, layers 0-55 | [5120,17408] | NVFP4 | same |
| MLP gate+up / down, layers 56-63 | [34816,5120] / [5120,17408] | FP8 row, re-encoded from BF16 with genuine per-row scales | `LC:modelopt_source.py:98-106` |
| Norms (input/post/final/GDN norm) | [5120] / [128] | BF16 | `src/ops/wrapper/gdn_gating_proj.cpp:148` requires BF16 |
| Token embedding | [248320,5120] | FP8 row (INFERENCE: recipe `official_recipes.py:156` and module name `fp8_embedding`) | gather `src/ops/wrapper/embedding.cpp:223-228` |
| lm_head (`text/output_head`) | [248320,5120] | FP8 row | FACT: the only FP8 `ops::linear` shape user, `LF/shapes/n248320_k5120.cu` |
| MTP input_proj, QKGV, O, gate/up, down | [5120,10240], [14336,5120], [5120,6144], [34816,5120], [5120,17408] | Q8_G32_FP16 (int8 codes + FP16 scale / 32) | INFERENCE by elimination: only the Q8 shape table has all five shapes (`src/ops/linear/q8/shapes/n5120_k10240.cu`, etc.). The FP8, NVFP4, BF16 and Q4 tables lack `n5120_k10240`. Recipe `_optional` gives Q8 (`official_recipes.py:35-46`) |
| MTP embedding, output head | shared with target | FP8 | `load/mtp.cpp:13-14` |
| Proposal head (`--lm-head-draft`) | [131072,5120] + INT32 id map | Q4_G64_FP16 | INFERENCE: `tools/convert/proposal.py:45-95` defaults to 131072 rows in Q4. Q4 has `n131072_k5120.cu`. Loader: `load/text.cpp:119-144` |

**Bytes read per decode token (weights, B=1).** FP8 bytes = N·K + 2N. NVFP4 bytes = N·K·(1/2 + 1/16).

| Class | Arithmetic | Bytes |
|---|---|---:|
| NVFP4 MLP ×56 | (34816·5120 + 5120·17408)·0.5625 = 150,405,120 ×56 | 8,422,686,720 |
| GDN ×48 | QKVZ 83,918,848 + out 31,467,520 + A/B 983,040 + conv 81,920 = 116,451,328 ×48 | 5,589,663,744 |
| FP8 MLP ×8 | 178,327,552 + 89,139,200 = 267,466,752 ×8 | 2,139,734,016 |
| Attention ×16 | QKGV 73,428,992 + O 31,467,520 = 104,896,512 ×16 | 1,678,344,192 |
| lm_head | 248320·5120 + 2·248320 | 1,271,895,040 |
| **Target weights / token** | (norms and one embedding row are negligible) | **≈19.10 GB** (NVFP4 44%, GDN 29%, FP8 MLP 11%, attn 9%, head 7%) |
| GDN recurrent state | 48 layers × 48·128·128·4 B = 151.0 MB. Read + write in ordinary decode | 0.302 GB |
| FP8 KV, 16 layers | (4·256·2 codes + 4·2·2 FP16 scales) ·16 = 33,024 B per cached key | 0.27 GB at 8K, 4.33 GB at 128K |
| MTP layer (Q8) | 424,673,280 elements × 1.0625 | 0.451 GB per pass |
| Proposal head (Q4) | 671,088,640 × 0.53125 | 0.357 GB per proposal |

INFERENCE: one MTP4 round (B=1) reads about 19.10 GB of target weights once. All M=5 routes use a single token
tile. The draft adds 4 × 0.451 GB of MTP passes and 4 × 0.357 GB of proposals, or about 3.2 GB. GDN state adds a
verify read plus a fold read and write, 0.45 GB. The round total is about 22.8 GB plus KV. The resident total is
19.10 + 1.27 (embedding) + 0.45 + 0.36 = 21.2 GB. The remaining ~2.5 GB of the 23.72 GB file is plausibly vision
and DFlash2.

## 2. Per-operation dispatch tables

Common kernel families:
- **A16 GEMV:** row-major packed weights decoded in registers × BF16 activation, FP32 FMA over 4 chains, then a warp reduction and scaling. `LF/fp8_a16_gemv.cuh:3-5,95-97`; `LN/nvfp4_a16_gemv.cuh:139,192-199`.
- **A16 sliced-K MMA:** codes widened exactly to BF16 fragments, `mma m16n8k16 bf16` with FP32 accumulation. Each warp owns a K slice; the reduction happens in shared memory inside the CTA. There is no global split-K. `LF/fp8_a16_sliced_k_mma.cuh:3-6,142-156,171-199`.
- **A8 MMA:** `mma kind::f8f6f4 m16n8k32 e4m3·e4m3→f32` (`src/ops/common/mma.cuh:59-62`). Loads use `cp.async` 16 B with an XOR-swizzled shared layout, `ldmatrix`, and ping-pong fragments. The epilogue multiplies by activation scale × row scale (`LF/fp8_a8_mma.cuh:55-276`).
- **A4 MMA:** `mma kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64 e2m1·e2m1→f32, ue4m3` (`mma.cuh:89-90`). This is warp-level `mma.sync` on SM120, not tcgen05.

Schedule notation: FP8 A8 = `Fp8A8MmaSchedule<Tok,Rows,K,WarpsTok,WarpsRows,Stages,MinBlk>`
(`LF/fp8_schedule.cuh:126-160`). FP8/NVFP4 A16 MMA = `<Rows,Tok,K,WarpRows,WarpTok,Stages,MinBlk>` (`:86-110`).
Sliced = `<Tok,KWarps,Stages>` (`LF/fp8_instances.cuh:28-31`); its CTA covers 16 rows × KWarps·64 K.

### A. Ordinary decode (`--spec none`), M=1, B=1. One graph per round (§6).

Layer order: `TX:1078-1117`. Every row is FACT unless it is marked otherwise.

| Op | Kernel (file:line) | W × A, accumulation | Tile / grid | Fusion | Dispatch condition |
|---|---|---|---|---|---|
| Embedding | `embed_gather_fp8` (`src/ops/launcher/embed_gather.cu:152`) | FP8 row → BF16 | per id | — | qtype switch, `wrapper/embedding.cpp:223` |
| GDN input RMSNorm + A/B + gating | `gdn_norm_gating_27_simt` (`src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu:15,101`) | BF16 W × BF16 x, FP32 SIMT | grid (48 heads, ⌈T/T_tile⌉) | norm, A/B dots, softplus/exp gating and h all in one kernel | `cols<=42` → FusedSimt27, `bf16_gdn_gating_proj_plan.cpp:376-377` |
| GDN QKVZ + causal conv + SiLU + conv-state snapshot | FP8 A16 GEMV `<8w,2rows/w,8vals,4chains>` with the `GdnConvOutput` epilogue (`GI/fp8/fp8_gdn_conv_fused.cu:84-97`; conv/SiLU `GI/gdn_conv.cuh:48-104`) | FP8 × A16, FP32 | 8 warps × 2 rows per CTA | projection, conv1d(k=4), SiLU and history publish in one kernel; Z stored plainly | b1 width 1 → FusedA16 (`GI/fp8/fp8_gdn_conv_plan.cpp:45-47,88-91`) |
| GDN recurrence | `recurrent_batch_update_kernel` (`GDN/recurrent.cuh:666-672`, launch `GDN/recurrent.cu:39-65`) | FP32 state in registers, SIMT; q/k L2-normalized in-kernel | grid (48, B, 8) × 128 threads; a warp owns 4 v-rows × 128 k | reads the source slot, writes the destination slot | `TX:1035-1040` (Verify phase, W=1) |
| GDN gated RMSNorm | `gated_rmsnorm` | BF16 | — | norm·silu(z) | `TX:1058` |
| GDN out + residual | FP8 A16 GEMV `<8,2,8,4,..,2,2>` (`LA/fp8/fp8_linear_add_decode.cu:18-36`) | FP8 × A16 | — | residual added in the FP32 epilogue (`src/ops/linear/common/epilogue.cuh:13-19`) | T=1 → decode (`LA/fp8/fp8_linear_add_plan.cpp:33`); A8 only when T≥22 (K=6144) |
| Attn input RMSNorm | `rmsnorm` | BF16 | — | — | `TX:850` |
| Attn QKGV | FP8 A16 GEMV `<8,2,8,4,Default,2,6>` (`AI/fp8_attn_input_decode.cu:16-25`) | FP8 × A16 | — | four outputs from one parent | T<5 → A16 (`AI/fp8_attn_input_plan.cpp:24`), T=1 → decode (`:29-30`) |
| q_norm, k_norm | 2× `rmsnorm` | BF16 | — | none; the fused `rmsnorm_rope` op is used only by DFlash | `TX:872-873` |
| RoPE (partial, `rotary_dim=head_dim·factor`) | `rope_launch` q+k (`wrapper/rope.cpp:123`) | FP32 math | — | q and k in one kernel | `TX:879`; `config.cpp:72` |
| KV append (FP8) + attention | `causal_attention_small_t_fp8_tiled_kernel` + `..._reduce_output_kernel` (`ATT/small_t_fp8.cu:14-88`) | Q: Hadamard-D256 + per-row E4M3 in-kernel; K: E4M3·E4M3 MMA, FP32; PV: `mma_f16` FP32 (`ATT/small_t_fp8.cuh:247-270,381,531`) | TokenTile 1: 8 warps, Bc=32, 2 blocks/SM; grid (4 KV heads, splits, B) | new K/V row Hadamard-rotated, FP16-scaled and E4M3-coded inside the partial kernel (`small_t_fp8.cuh:186-215`) | q_heads 24, W≤16, FP8 with W≤4 → prompt_limit 0 → SmallT (`ATT/causal_softmax_attention.cpp:344-366`) |
| Output gate | `sigmoid_gate_mul` | BF16 | — | — | `TX:922` |
| Attn O + residual | FP8 A16 GEMV, as for GDN out | FP8 × A16 | — | residual | `TX:924` |
| Post-attn RMSNorm | `rmsnorm` | BF16 | — | — | `TX:1073` |
| MLP gate/up, layers 0-55 | NVFP4 A16 GEMV `<8,2,16,4,Direct,..,2>` (`LS/nvfp4/nvfp4_linear_swiglu_decode.cu:8-9`) | NVFP4 × A16, FP32 | — | gate+up rows and SiLU·up in the epilogue; writes [17408] BF16 | T=1 → DecodeFusedA16, even with AllowA4 (`LS/nvfp4/nvfp4_linear_swiglu_plan.cpp:33`) |
| MLP down + residual, 0-55 | NVFP4 A16 GEMV `<8,2,16,4,StagedRaw,..,2>` (`LA/nvfp4/nvfp4_linear_add_decode.cu:14-16`) | NVFP4 × A16 | — | residual | T<8 → A16 (`LA/nvfp4/nvfp4_linear_add_plan.cpp:27-28`), T=1 → decode (`LA/nvfp4/nvfp4_linear_add_a16.cu:118-120`) |
| MLP gate/up, layers 56-63 | **A8**: `fp8_a8_quantize_kernel` then A8 MMA `T16R64K128` (16 tok × 64 rows × K128, 1×2 warps, 2 stages) (`LS/fp8/fp8_linear_swiglu_a8.cu:9-19`) | FP8 × **A8** | 15 of 16 token rows are padding | SwiGLU token-major epilogue | AllowA8 & (T==1 or T≥3) → A8 (`LS/fp8/fp8_linear_swiglu_plan.cpp:26`) |
| MLP down, 56-63 | FP8 A16 GEMV (`LA/fp8/fp8_linear_add_decode.cu`) | FP8 × A16 | — | residual | K=17408: A8 from T=25 (`fp8_linear_add_plan.cpp:28`) |
| Final norm | `rmsnorm` | BF16 | — | — | `TX:706` |
| lm_head | FP8 A16 sliced-K `<KWarps 16, Tok 8, 1>` (`LF/shapes/n248320_k5120.cu:11-31`) | FP8 × A16, BF16 MMA, FP32 | 16 rows × K1024 per CTA, 512 threads, 15,520 CTAs | — | T<42 → `launch_ksplit`, T≤8 → tile<8>. `uses_a8` is always false (`:100`) |
| Hidden scatter; sampling | `scatter`; `sample_row_kernel` or partial-topk + finalize (`src/ops/launcher/sampling.cu:22-50`) | BF16 logits on device | — | the host receives only an egress record | `program/decode.cpp:52-62` |

### B. MTP4 round, B=1: target verify M=5 plus draft. One graph (§6).

Target verify: `mtp.cpp:125-151` → `target_verification.cpp:14-44` → `TX:713-772` with `GdnStateAction::RecordForReplay`.

| Op (M=5) | Kernel | W × A | Tile | Dispatch |
|---|---|---|---|---|
| GDN norm+A/B+gating | FusedSimt27, as in A | BF16 × A16 | grid (48, ⌈5/T⌉) | cols≤42 |
| GDN QKVZ | FP8 A16 sliced-K, 16 KWarps, capacity 8, 1 stage (`GI/fp8/fp8_gdn_input_matrix.cu:30-39,63`) | FP8 × A16 | 16 rows × K1024 | record→snapshot plan, b1 width 5 → **MaterializedA16**. Width 4-6 is not fused; A8 only when width≥10 (`GI/fp8/fp8_gdn_conv_plan.cpp:45-47,88-105`) |
| GDN conv (record) | `gdn_projected_conv_record_launch` (`GI/gdn_projected_conv.cu:147`), a separate kernel | BF16 | — | `fp8_gdn_conv_plan.cpp:185-189` |
| GDN recurrence | `recurrent_record_kernel`: reads state, runs 5 steps sequentially, records k/v/gate, **does not write state** (`GDN/recurrent.cuh:676-684`) | FP32 SIMT | grid (48,1,8)×128 | `TX:1026-1033` |
| GDN out / Attn O | FP8 A16 sliced `<8 tok, 8 KWarps, 2 stages>` + residual (`LA/fp8/fp8_linear_add_a16.cu:69-79`) | FP8 × A16 | 16 rows × K512 | T=5 > 4 → matrix. A8 only at T≥22 |
| Attn QKGV | **A8**: quantize, then `Small32` = `Fp8A8<32,64,128,1,2,3,2>` (`AI/fp8_attn_input_a8.cu:31-49`) | FP8 × A8 | 32 tok × 64 rows × K128, 2 warps, 3 stages | T≥5 → A8 (`AI/fp8_attn_input_plan.cpp:24`) |
| Attention | visible ≤128 keys → Prompt (separate FP8 append + prompt kernel). Otherwise SmallT TokenTile 5: 8 warps, Bc=64, 1 block/SM, with fused append and reduce | as in A | grid (4, splits, 1) | FP8 prompt_limit 128 for W≤8 (`causal_softmax_attention.cpp:358-360`) |
| MLP gate/up, 0-55 | **A4**: NVFP4 quantize (row-major scales), then `M32N128 = Nvfp4A4<32,128,256,2,4,2,1>` (`LS/nvfp4/nvfp4_linear_swiglu_a4.cu:23,39-54`) | NVFP4 × **A4** | 32 tok × 128 rows × K256, 8 warps, 2 stages | AllowA4, 5≤T<256 → FusedA4 (`nvfp4_linear_swiglu_plan.cpp:33-38`) |
| MLP down, 0-55 | NVFP4 A16 SIMT, exact T=5 (4 warps) (`LA/nvfp4/nvfp4_linear_add_small_t.cu:20-30`) | NVFP4 × A16 | — | T<8 → A16 and T≤5 → small_t (`nvfp4_linear_add_a16.cu:122`) |
| MLP gate/up, 56-63 | A8 `T16R64K128` + quantize | FP8 × A8 | — | T≥3 |
| MLP down, 56-63 | FP8 A16 sliced `<8,8,2>` | FP8 × A16 | — | T<25 |
| lm_head (5 cols) | sliced-K tile<8>, as in A | FP8 × A16 | 15,520 CTAs | T≤8 |
| Argmax; accept | `cudaMemsetAsync` + `argmax_tiled_atomic_kernel` (`src/ops/launcher/argmax.cu:56-78`); `speculative_accept_greedy_drafts`, hidden select and scatter | on device | — | `TX:768`; `target_verification.cpp:34-44` |

B=2 (two active requests, M=10) variant (FACT): GDN QKVZ → MaterializedA8 when width·batch≥9 (`fp8_gdn_conv_plan.cpp:92-94`),
using `Fp8A8T64R128K128` (`GI/fp8/fp8_gdn_input_a8.cu:9-13`). NVFP4 down → A4 (T≥8). FP8 K=6144 → still A16.
The KV split target grows to 160 or 320 CTAs (`ATT/small_t.cu:228-245`).

Draft (`mtp.cpp:153-199`): `mtp_prepare_next_round`, then one MTP layer at **width 5** over all target
hidden columns (`TX:798-828`). It selects the accepted hidden, then runs the proposal: Q4 [131072,5120] A16 GEMV,
argmax, and id remap (`TX:564-591`). Three more autoregressive MTP layers follow at width 1, each with a proposal
and a D2D hidden copy (`mtp.cpp:173-197`). The MTP layer (`TX:287-414`) has two stem RMSNorms plus a pack, then
Q8 input_proj [5120,10240], RMSNorm, Q8 packed QKGV, q/k norm, RoPE, attention, sigmoid·, and **Q8 O-proj followed
by a separate `residual_add`**. It then runs RMSNorm and **unfused FFN**: Q8 gate_up `ops::linear`, `silu_mul`, Q8 down,
and `residual_add` (`execution/ffn.cpp:68-80`), then the final norm. Q8 paths are A16 (BF16 MMA over widened int8,
`src/ops/linear/q8/q8_a16_mma.cuh:5,219`; shape selection e.g. `q8/shapes/n34816_k5120.cu:25-35`). After the
host commit, one eager `recurrent_fold_kernel` covers all 48 layers (grid z = layers·8). It re-runs the committed
columns from the records and writes the final state (`GDN/recurrent.cu:104-127`, `recurrent.cuh:687-697`, call
`program/prefill.cpp:803`).

### C. Prefill chunk, M=2048 (eager). Thresholds for M=256/512/1024 are in the last column.

`TX:1178-1395`: one chunk per call (`break` at `TX:1385`).

| Op | Kernel @2048 | W × A, accumulation | Tile | Changes at 256 / 512 / 1024 |
|---|---|---|---|---|
| ids H2D; positions; embedding | `copy_i32`, `fill_i32_positions`, FP8 gather (`TX:1236-1249`) | — | — | — |
| GDN RMSNorm + A/B | **Composed**: `rmsnorm`, then cooperative split-K BF16 MMA gating (`bf16_gdn_gating_proj_plan.cpp:451-455`, coop launch `bf16_gdn_gating_proj_kernels.cu:301-311`) | BF16 × BF16 MMA, FP32 | SplitK=4 for cols 1025-2048 | SplitK=8 for 9-1024 cols (`plan.cpp:30-39`) |
| GDN QKVZ | quantize, then `Fp8A8T64R128K128 = <64,128,128,2,4,2,2>` (`GI/fp8/fp8_gdn_input_a8.cu:7-13`) | FP8 × A8, FP32 | 64 tok × 128 rows × K128, 8 warps, 2 stages. 4,096 CTAs | same for any T≥8 (`GI/fp8/fp8_gdn_input_plan.cpp:23`) |
| Causal conv + SiLU | `causal_conv1d_prefill_split_launch` (`wrapper/causal_conv1d_silu.cpp:320`) | BF16 | — | — |
| Chunked GDN | `prepare_kernel` + `chunk_recurrence_kernel<64>` (§4) | BF16/TF32 MMA, FP32 state | prepare 48·128 CTAs; recurrence 96 CTAs × 256 thr | chunked when T≥16 (`GDN/gated_delta_net.cpp:215-220`) |
| Gated RMSNorm | 1 kernel | BF16 | — | — |
| GDN out / Attn O + residual | quantize, then `Fp8A8T64R128K128` + residual (`LA/fp8/fp8_linear_add_a8.cu:21-47`) | FP8 × A8 | 64×128×128 | T≤64 → T32R32, ≤128 → T64R64K128 (`:30-32`). A8 from T=22 |
| Attn QKGV | quantize, then `Prefill = <64,128,128,2,4,2,2>` (`AI/fp8_attn_input_a8.cu:46,59`) | FP8 × A8 | 64×128×128 | 97-144 → Wide128/Tail144; >144 → Prefill (`:48-59`) |
| q/k norm, RoPE | 3 kernels | BF16 | — | — |
| KV append + attention | `kv_cache_append_batch_launch`, then `causal_attention_prompt_fp8_kernel` (`ATT/prompt_fp8.cu:63-85`) | §5 | 32 q-tiles × 24 heads = 768 CTAs × 512 thr, 92,416 B smem | route Prompt for W>16 (`causal_softmax_attention.cpp:367-376`) |
| MLP gate/up, 0-55 | NVFP4 quantize (Tiled256 scales), then **TMA** `Nvfp4A4TmaMmaSchedule<256,3,1>` (`nvfp4_linear_swiglu_plan.cpp:82-89`; `LS/nvfp4/nvfp4_linear_swiglu_a4_tma.cu:8`) | NVFP4 × A4, FP32 | 256 tok × (64 gate + 64 up) rows × K128/stage, 3 stages. 1 producer thread (of 128, `setmaxnreg 40`) + 8 consumer warps (`setmaxnreg 232`), full/empty mbarriers (`LN/nvfp4_a4_tma.cuh:130-233`). 272×8 = 2,176 CTAs | TMA from T≥256 (`plan.cpp:37`); below that `M32/64/96/128N128` A4 (`nvfp4_linear_swiglu_a4.cu:23-60`) |
| MLP down, 0-55 | NVFP4 quantize (Tiled256), then TMA `<256,3,1>` + residual (`LA/nvfp4/nvfp4_linear_add_a4_tma.cu:8`) | NVFP4 × A4 | 256 tok × 128 rows. 40×8 = 320 CTAs, K=17408 → 136 K-tiles | TMA only when T≥1024 (`LA/nvfp4/nvfp4_linear_add_a4.cu:23`). 256-384 → `M128N128Resident` (1 stage, 2 blk/SM); 385-512 → `Pipelined` (2 stages); `:15-49` |
| MLP gate/up, 56-63 | quantize, then `Fp8A8T64R128K128` + SwiGLU epilogue (`LS/fp8/fp8_linear_swiglu_a8.cu:19-23`) | FP8 × A8 | 64×128×128 | same for T>96 |
| MLP down, 56-63 | quantize, then `Fp8A8T64R128K128` + residual | FP8 × A8 | — | same for T>128 |
| Final norm (all rows) | `rmsnorm` | — | — | `TX:1266` |
| MTP prefill (every chunk) | Stem (embedding, 2 norms, pack, Q8 input_proj at M=2048), then K/V-only projection, k-norm, RoPE and `kv_cache_append` (`TX:473-491`). The full MTP layer runs only on the last row of the final chunk (`TX:506-561`) | Q8 × A16 | — | — |
| lm_head (final chunk only) | last row only, tile<8> A16 (`TX:1268-1284`) | — | — | — |

## 3. Activation quantization

| Consumer | Decode M=1 | Verify M=5 | Prefill M≥256 |
|---|---|---|---|
| Attn QKGV (FP8) | A16 | **A8** (T≥5) | A8 |
| GDN QKVZ (FP8) | A16 (fused GEMV) | A16 (materialized) | A8 (T≥8) |
| Attn O / GDN out (FP8, K=6144) | A16 | A16 | A8 (T≥22) |
| MLP gate/up NVFP4 0-55 | A16 | **A4** (T≥5) | A4 (TMA ≥256) |
| MLP down NVFP4 0-55 | A16 | A16 | A4 (T≥8; TMA ≥1024) |
| MLP gate/up FP8 56-63 | **A8** | A8 | A8 |
| MLP down FP8 56-63 | A16 | A16 | A8 (T≥25) |
| lm_head FP8 | A16 | A16 | A16 (one row) |
| GDN A/B BF16; MTP Q8; proposal Q4 | A16 | A16 | A16 |

- **FP8 A8** (FACT): one **separate** kernel per consumer call, `fp8_a8_quantize_kernel` with 256 threads/token (`LF/fp8_a8.cu:20-78`). It uses a per-token scale `amax/448` in FP32 and E4M3 SATFINITE codes. Every A8 op launches it first: `AI/fp8_attn_input_a8.cu:33`, `GI/fp8/fp8_gdn_input_a8.cu:9`, `LA/fp8/fp8_linear_add_a8.cu:41`, `LS/fp8/fp8_linear_swiglu_a8.cu:11`. It is not fused into the preceding RMSNorm or epilogue. The joined parents amortize one quantization across Q/K/gate/V and across Q/K/V/Z.
- **NVFP4 A4** (FACT): separate `nvfp4_a4_quantize_kernel`, one thread per 16-element group (`LN/nvfp4_a4_quantize.cuh:28-69`). The scale is `E4M3(divisor·amax/6)` per K16, where `divisor` is the static per-tensor `input_global_scale`: ModelOpt `input_scale`, reciprocated in `LC:modelopt_source.py:89-95,124-133`. Codes are `E2M1(x·divisor/scale)` (`LN/nvfp4_codec.cuh:85-115`). The scale plane is row-major for the non-TMA path and Tiled256 for TMA (`nvfp4_linear_swiglu_plan.cpp:85`, `LA/nvfp4/nvfp4_linear_add_a4.cu:57-58`). Gate and up share one divisor (`weight_input.cpp:100-105`).
- **Attention Q and KV**: quantized inside the attention kernels, not by a separate op (§5).

## 4. GDN algorithms

**Prefill (T≥16), two kernels** (`GDN/gated_delta_net.cpp:206-240`, chunk constants `GDN/chunked/launch.h:14-15`):
1. `prepare_kernel`, grid = 48 v-heads × ⌈T/16⌉ chunks, 256 threads (`chunked/prepare.cu:17-159`). It L2-normalizes q/k in FP32 (`:41-56`) and stores them BF16 in a swizzled layout. It computes the inclusive cumulative g with `__shfl_up`, prefix `exp(G_i)` and suffix `exp(G_last−G_i)` (`:63-76`). One warp forms K·Kᵀ and another Q·Kᵀ with **BF16 `mma m16n8k16`** (`:80-114`). It masks the upper triangle before exponentiation and builds `L_ij = β_i e^{G_i−G_j} k_i·k_j`. `(I+L)⁻¹` is found by **scalar FP32 forward substitution in one warp** (`:116-128`, not tensor-core), then scaled by diag(β) (`:130`). This is the WY/UT-transform `solve` matrix. The kernel materializes `solve[16×16]`, `mqk[16×16]`, prefix and suffix per (head, chunk) in a 2,304-B `ControlChunk`. Q/K for each QK group go to an 8,192-B `QkChunk` (`launch.h:27-45`). This workspace is about 22.4 MB at T=2048 (`launch.h:63-73`).
2. `chunk_recurrence_kernel<DV>`, grid = 48·(128/DV). On 170 SMs, `value_tile` picks DV=64 → **96 CTAs**, 256 threads, 2 blocks/SM (`chunked/recurrence.cu:142,353-403`; cost table `:367-389`). The FP32 state tile Sᵀ[128×DV] lives in registers. **Chunks run serially inside each CTA** (`:181-337`). Stages are double-buffered with `cp.async` (`:63-95,176-179,333-336`). Per chunk: prediction K·S by **TF32 MMA** and history Q·S by **BF16 MMA** (`:205-229`). Then residual R = V − prefix·prediction (`:97-109`) and Δ = solve·R by TF32 MMA (`:266-275`). Output = prefix·history + mqk·Δ by TF32, stored BF16 ×scale (`:278-297`). The state update is Sᵀ = decay·Sᵀ + Kᵀ·diag(suffix)·Δ by **BF16 MMA**, with Δ rounded to BF16 (`:299-331`). The master state stays FP32. There is no inter-chunk parallel state passing and no separate output kernel.

**Decode/verify** (FACT): `recurrent_batch_update_kernel` / `recurrent_record_kernel`, grid (48 v-heads, B, 8), 4 warps. Each warp holds 4 value rows × 128 keys of FP32 state in registers (`GDN/recurrent.cuh:13-20`, `recurrent.cu:39-100`). It is SIMT, one token at a time. Q/K normalization and the gating use g/β from the fused control kernel. The state is ~3.1 MB per layer. Ordinary decode reads it and writes it. MTP verify only reads it and records k/v/gate. The commit-time fold re-reads it and writes it.

## 5. Attention (16 layers, 24 q heads / 4 KV heads, D=256, FP8 KV `Fp8E4M3Row256`)

- **KV format** (FACT): K and V are both rotated by a normalized D256 Hadamard (`src/ops/kv_cache/hadamard_d256.cuh`; used at `ATT/small_t_fp8.cuh:205`). The scale is one FP16 per (key, KV head, K or V) row, `amax/448` clamped, with the represented FP16 scale fed back before coding (`src/ops/kv_cache/fp8_e4m3_row_codec.cuh:18-67`).
- **Prefill prompt kernel** (`ATT/prompt_fp8.cuh:18-47,64-414`; launch `prompt_fp8.cu:14-37`): Br=Bc=64, 16 warps (8 producer and 4 D-consumer groups). One CTA per (64 query rows, q-head), so the 6 q-heads of a GQA group re-read the same KV tile (INFERENCE: each CTA is per q-head). Q is Hadamard-rotated and quantized per row to E4M3 in-kernel (`:137-149`). QK uses **native E4M3 m16n8k32 MMA** with FP32 accumulation, then × q_scale × k_scale (`:257-272`). The online softmax uses `exp2` with running max and sum rescale (`:313-333`). V is widened to FP16 with its row scale; PV uses **FP16 MMA with FP32 accumulation** (`:409`). K/V pages come from the page table with `cp.async` 16 B (`:192-212`). Keys above `max_query_abs` are zero-filled, which gives the causal bound. At a 2,048-token chunk there are 768 CTAs. The FP8 append is a separate kernel before attention (`prompt_fp8.cu:68`).
- **Decode/verify SmallT** (`ATT/small_t_fp8.cu:14-200`): split-KV partial kernel (grid 4 KV-heads × splits × B) plus a reduce kernel (grid 24 heads × 1 × W·B, 256 thr, `:62-88`). Each CTA handles the whole GQA group (rows = T·6). The number of splits is `causal_small_t_split_upper_bound` (`ATT/small_t.cu:23-44`): ⌈min(L,4096)/64⌉ with a minimum of 4, then ⌈min(L,8198)/128⌉, then /256 to 16,390, then /480, capped at 85. For FP8 at T=1 and L>8198 it goes straight to 85 (`:49-52`). **8K context → 64 splits → 256 partial CTAs (128 keys each); 128K → 85 splits → 340 CTAs (~1,542 keys each)**. Verify T=5 gives the same 256/340. Graph replays size the grid to the profile-interval maximum. The device computes the active split count from the actual window, and extra CTAs exit (`small_t_fp8.cuh:153-157`; profile ends `program/planning/graph_profiles.cpp:56-60`).

## 6. Launch structure and synchronization (INFERENCE counts from the FACT call sequences above)

Assumption: full attention is at layers 3, 7, …, 63 (the standard Qwen3.5 interval). That puts 2 attention and 6 GDN layers inside the FP8-MLP range 56-63.

| Workload | Per-layer kernels | Total per round/chunk | Graph | Host syncs |
|---|---|---|---|---|
| A: ordinary decode | GDN 8 (9 with FP8 MLP); attn 12 (13) | 42·8+6·9+14·12+2·13 = **584** + embedding, final norm, lm_head, scatter and 1-2 sampling kernels ≈ **590 kernels + 2 memcpy nodes** | one `cudaGraphLaunch`; H2D ingress and D2H egress are captured (`decode.cpp:32-34,60-62`; `graph_execution.h:11-22`; capture `src/core/decode_graph.cpp:59-68`) | 1 (`decode.cpp:348-352`) |
| B: MTP4 round | verify: GDN 10, attn 14 | target ≈ 48·10+16·14 = 704, plus ~8 (embedding, norm, head, argmax memset+kernel, accept, select, scatter). Draft ≈ 4 MTP passes × ~23 + 4 proposals × 4 + copies ≈ 110. **≈820 nodes** | whole round captured in one graph, profile chosen by (batch, frontier interval) (`decode.cpp:447-452`; `graph_profiles.cpp:62-88`) | 2: round wait (`decode.cpp:508-512`), then eager fold + commit wait (`program/prefill.cpp:803,875-876`) |
| C: 2,048-token chunk | GDN 15, attn 16 | 48·15+16·16 = **976** + ~25 (embedding, positions, final norm, MTP stem/KV) ≈ **1,000 launches** | **eager, no graph** (only decode families are captured; `program/graphs.cpp:315-420`) | 1 per chunk (`TX:1389-1393`) |

PDL (`src/core/pdl.cuh`) is used only by Q4, Q5 and Q8 SIMT/GEMV and MoE kernels (grep). It is not used by the FP8/NVFP4 target kernels. That means the proposal head and some MTP Q8 kernels may overlap their prologues, but the main path does not. `chunk_recurrence` calls `cudaFuncSetAttribute` on every launch (`recurrence.cu:357-358`). This is a host API call, not a sync. All scratch comes from a bump-pointer `WorkspaceArena` that is reset per round (`TX:689,709`), so no allocation happens per operation.

## 7. Corrections to the existing findings

- `ninfer-runtime-deep-dive.md`: "Ninfer's installed binary is still not connected to either source pin". Per this task's premise, the running binary is e31bc99.
- deep-dive §5: "gated_delta_net.cpp:244-275" does not exist; the file has 242 lines. Dispatch is at `:206-240`. "Parallel chunk algorithms" overstates it. The chunks are processed **serially** in one 96-CTA kernel. Tensor cores apply only within a chunk, and the triangular solve is scalar FP32 forward substitution.
- deep-dive §4: the GDN projection+conv plan under AllowA8 is fused for widths 1-3 and 7-**9**. Width 10 is MaterializedA8, because the `width>=10` A8 check runs first (`fp8_gdn_conv_plan.cpp:88-91`). The fused Qwen27 norm/control kernel is used only for ≤42 columns. Prefill uses a composed RMSNorm plus a cooperative split-K MMA.
- deep-dive §2 and `ninfer-performance-comparison.md` cite the generic `ops::linear` shape tables (`LF/shapes/n5120_k17408.cu`, `n5120_k6144.cu`, `n14336`, `n16384`; `LN/shapes/n5120_k17408.cu`) as the routes that execute. In this model **only the lm_head goes through `ops::linear`**. The executed plans are the fused-op plans, with different thresholds. Attention input reaches A8 at **T≥5**, not 12. GDN input in prefill reaches A8 at T≥8, and in verify at width≥10 (or B·W≥9). FP8 O/GDN-out with K=6144 reaches A8 at **T≥22**, not 25.
- performance-comparison: the "A16 T=1 GEMV pairs gate/up … SiLU epilogue" route is not taken. The artifact policy is AllowA8, so FP8 MLP 56-63 at decode runs **A8** (quantize + 16×64 MMA tile).
- `ninfer-feature-parity.md` F04 describes "chunk preparation, state passing and output kernels". At e31bc99 there are 2 kernels, prepare and a fused recurrence/output kernel. F05's "separate packed-prefill" is wrong for this model: prefill uses the `causal_cache` prompt kernel, not `dense/packed`. F06 omits the D256 Hadamard rotation of K, V and Q, which is essential to the FP8 KV scheme.
- `ninfer-model-format.md` provenance (unsloth revision, manifest SHA) is stale for the running artifact. The current file comes from NVIDIA ModelOpt through the local converter, with NVIDIA-sourced FP8 matrices carrying **per-tensor** scales broadcast to rows. Its byte size equals the old manifest's, which follows from having the same format assignment and does not prove identical bytes. The adapter's claim of an exact re-expression is contradicted by the BF16 rounding of the scale (`modelopt_source.py:144`).
- The earlier findings do not mention that `--spec mtp` makes **every** decode round a 5-column MTP round, and that the draft side (Q8 MTP ×4 + Q4 shortlist head ×4) reads ~3.2 GB per round.

## 8. Implications for a competitor (ranked by likely time share; reasoning, not measurement)

**Decode / MTP round (bandwidth-bound; ≈22.8 GB per round plus KV):**
1. **Small-M weight streaming at near-peak bandwidth**: NVFP4 MLP (8.4 GB) and FP8 GDN/attention/head (10.7 GB). Ninfer reads packed weights directly with A16 GEMV at M=1 and sliced-K BF16 MMA at M=5–8, with no global split-K and no quantize launch on most ops. The 19.1 GB of target weights is the floor. At ~1.8 TB/s that is ~10.6 ms, so any kernel below ~85% of peak bandwidth dominates.
2. **Cheap draft**: Q8 MTP (0.45 GB) plus a Q4 131,072-row shortlist head (0.36 GB) instead of the 1.27 GB FP8 full head per proposal. That saves ~3.6 GB per round versus using the full head.
3. **One graph per round with on-device accept/argmax**: ~820 nodes and 2 host syncs. Eager submission of that many launches would cost milliseconds per round (INFERENCE).
4. **Long-context attention**: FP8 KV halves traffic (4.3 GB/token at 128K ≈ 23% of weights). Split-KV grids of 256–340 CTAs, Hadamard, and fused append matter beyond ~16K.
5. **GDN state traffic** of 0.3–0.45 GB per round, plus SIMT recurrence and fold.
6. **Epilogue fusions** remove small launches and intermediate round trips: residual in out/down, conv+SiLU in the QKVZ GEMV, norm+A/B+gating, SwiGLU.

**Prefill (compute-bound; ~24.3 G MAC per token ≈ 99.7 TFLOP per 2,048 chunk before attention):**
1. **NVFP4 A4 GEMM** carries 61.5% of MACs (MLP 0-55): a warp-specialized TMA/mbarrier kernel with 3 stages and a 256×128×128 tile. The down projection uses TMA only from T≥1024. Its 320-CTA grid at T=2048 leaves tail waves on 170 SMs (INFERENCE).
2. **FP8 A8 GEMM**, 38.5% of MACs (GDN QKVZ is the largest share): 64×128×128 tiles, 2-stage `cp.async`, ping-pong `ldmatrix`, plus a separate quantize pass per op.
3. **Prompt attention** grows with context: ~0.8% of MACs at a 2K prefix, ~13% at 32K, ~50% at 128K. Ninfer's per-q-head CTAs re-read KV 6×, which leaves room for a GQA-shared tile.
4. **Chunked GDN**: few FLOPs, but 128 serial chunk steps × 48 layers on only 96 CTAs. It is likely latency-bound and a candidate for chunk-parallel state passing (INFERENCE).
5. ~1,000 eager launches per chunk are ≲1–2% of a multi-hundred-ms chunk (INFERENCE). This is low priority for prefill.

## Unverified

- The recipe modules for the local artifact are absent. The MTP=Q8 and proposal=Q4-131072 assignments come from elimination and defaults. The embedding=FP8 and conv=BF16 assignments come from the recipe.
- `layer_types` placement and the 170-SM count (taken from code comments `bf16_gdn_gating_proj_plan.cpp:33`, `small_t.cu:229`) are assumed.
- The MTP KV cache storage dtype is assumed to be FP8.
- The Q8/Q4 draft tile choices were only sampled.
- Exact launch counts for the draft and sampling are approximate. No measured timings exist.
- Per-q-head KV re-read in the prompt kernel is inferred from its grid, not traced line-by-line.
