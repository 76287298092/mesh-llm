# Direct Ninfer weight contract

Status: source and metadata contract established September 28, 2026. The user
explicitly selected direct `.ninfer` reading, superseding the earlier no-parser
and offline-conversion-only constraints. The paused offline export/assembler is
retained but is not a runtime prerequisite. No Ninfer implementation is imported.

Source references below are relative to the read-only Ninfer checkout at
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`. The actual source artifact is pinned
by SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`,
23,719,715,844 bytes. Its published artifact manifest declares an unsloth-derived
quantized checkpoint. Untracked NVIDIA conversion scripts are not provenance.

## Reader and runtime boundary

The independent Rust reader accepts the observed single-file v3 format. It
retains the opened file, limits JSON to 4 MiB, rejects duplicate keys, checks
object extents, alignment, encoded lengths and binding references, and rejects
multi-file artifacts explicitly. Structural validation alone does not establish
content identity: model execution additionally requires the pinned whole-file
hash. The opaque 16-byte artifact ID is not a checksum.

The model-facing source adapter presents checked logical tensor views to the
existing resident loader. Virtual offsets are not offsets in the native file.
Every view has an actual canonical-byte hash; copies recheck it before uploaded
weights may execute. Old `.mspec` framing, content checks and identity admission
remain unchanged. Source inspection and hashing are startup costs, not decode
speedups. The native profile is `qwen3.8-27b:text:ninfer-v3-control-v1`.

## Canonical target inventory

All 963 source text bindings map to 1,589 canonical arrays totaling
20,375,588,160 bytes before alignment. Existing tensor names are retained:
`tensors/model.language_model.*` and `tensors/lm_head.*`.

- Embedding is FP8 `[248320,5120]` plus BF16 row scales `[248320,1]`.
- All target FP8 projections preserve their code bytes and BF16 row multipliers.
- First 56 MLPs preserve NVFP4 codes, E4M3 block scales, and FP32 weight/input
  divisors. Last eight MLPs remain FP8.
- GDN `A_log` and `dt_bias` remain FP32 `[48]`, not narrowed to BF16.
- Norm words remain unchanged BF16. Input/post/final and Q/K norms apply `1+w`
  at execution; GDN gated norm uses direct gamma. No offline norm adjustment.
- The source has no `self_attn.k_scale` or `v_scale` bindings. Do not fabricate
  these 32 old-inventory tensors. This profile initially uses BF16 KV.
- Packed Q8 MTP, Q4 shortlist and INT32 token mapping remain in the native file.
  Their execution is rejected until typed consumers are qualified; they are not
  reinterpreted as the old BF16 MTP tensors.

Stored norms and decay semantics: `tools/convert/qwen3_5.py:521-568,601,702-712,968`,
`tools/convert/methods.py:169-191`, `src/ops/kernel/rmsnorm.cuh:14-25`,
`src/ops/kernel/gdn_gating.cuh:15-26`, `src/ops/wrapper/rmsnorm.cpp:91-98`.

## Checked addresses and reversible transformations

Binding ranges count logical elements, not encoded bytes. Matrix children must
cover complete rows of a validated parent. See `src/artifact/binder.cpp:44-63,109-145`
and `src/core/weight_view.cpp:201-223`. The actual source payload starts at byte
393216, but code derives that boundary from validated framing.

Let parent shape be `[N,K]`, child begin row `R`, and `A(x)=ceil(x/256)*256`.

| Plane | Byte address relative to physical parent |
| --- | --- |
| FP8 code `(r,k)` | `(R+r)*K+k` |
| FP8 BF16 row scale | `A(N*K)+2*(R+r)` |
| NVFP4 code byte | `(R+r)*(K/2)+floor(k/2)`, low nibble first |
| NVFP4 scale base | `A(N*K/2)` |
| NVFP4 weight divisor | `A(N*K/2)+N*K/16`, four unchanged FP32 bytes |
| Direct BF16/F32 element e | `2*e` / `4*e` |

The NVFP4 source scale index for parent row p and K-group g is
`((p/128)*(K/64)+g/4)*512+(p%32)*16+((p%128)/32)*4+g%4`.
The canonical view copies to natural index `r*(K/16)+g`, preserving bytes.
The independent inverse indexes each stored byte back to its canonical row/group.
Both N and child rows are multiples of 128 and K is a multiple of 64.

Other transforms:
- Convolution: BF16 `[4,10240]` to `[10240,1,4]`,
  `dst[channel*4+tap]=src[tap*10240+channel]`, no tap reversal.
- Attention source parent `[14336,5120]` has Q/K/gate/V rows
  `[0,6144)`, `[6144,7168)`, `[7168,13312)`, `[13312,14336)`.
  Canonical q_proj interleaves 256 Q rows then 256 gate rows per head, applying
  the same permutation to code rows and BF16 scale rows. K/V stay independent.
- GDN Q/K/V/Z parent `[16384,5120]` splits at rows 2048,4096,10240,16384.
  QKV is contiguous; Z uses the remaining rows.
- MLP gate/up parent `[34816,5120]` splits at row17408. Both children retain the
  parent weight divisor; activation divisors come from their individual uses.

Sources: `tools/artifact/layouts.py:159-245`, codecs `fp8_row.py:25-96`,
`nvfp4.py:21-60,79-144`, and `tools/convert/qwen3_5.py:496-632`.
The runtime planner and inverse checks are independently implemented in
`src/packages/qwen3_8_27b/native_views/`.

FP8 gather must round `E4M3(code)*BF16(row_scale)` to BF16 before normalization;
see `src/ops/kernel/embed_gather.cuh:25-53`. No full embedding dequantization
buffer is needed. FP32 decay parameters need a separate consumer variant;
changing their precision to fit an old loader would change represented weights.

## Weight parity is not arithmetic parity

The initial native-file execution retains our arithmetic control, explicitly
identified by the new model profile. These known differences remain:

- Ninfer NVFP4 activation scale is `RN(RN(d_x*amax)/6)` rather than
  `RN(RN(amax/6)*d_x)`. Code normalization multiplies by d_x then divides by
  decoded scale; our existing route divides by `RN(scale/d_x)`.
- A zero encoded NVFP4 activation scale produces a zero group in Ninfer; the
  existing Rust route substitutes scale0.125.
- Ninfer FP8 normalization multiplies by `RN(1/scale)` rather than dividing by
  scale, and its all-zero row uses scale0 rather than scale1.
- GDN beta is FP32 in the cited Ninfer path; our control rounds it to BF16.
  Our gated norm also rounds normalized and weighted intermediates to BF16.
- Accumulation, A16/A8/A4 selection, exp/rsqrt approximations and fusion differ.

Evidence: `src/ops/linear/nvfp4/nvfp4_codec.cuh:85-114`,
`src/ops/linear/nvfp4/nvfp4_operands.h:38-48`,
`src/ops/linear/fp8/fp8_a8.cu:56-68`, `src/ops/kernel/gdn_gating.cuh:15-26`.
Do not compensate for runtime arithmetic by rewriting imported divisors/weights.
Any alternative arithmetic belongs in a separately qualified execution profile.

## Qualification scope

Twenty independent-reader synthetic Rust tests and six canonical-view tests pass.
The view tests cover actual metadata cardinality, missing/invalid bindings,
partial rows, parent extents, scale permutation, Q/gate order and signed-zero
convolution bytes. These are not real-file payload or GPU execution evidence.

Next: real-file whole/object hash comparison against official-reader evidence;
encoded embedding and FP32 gate oracles; direct-file legacy/stream model checks;
short and chunked teacher-forced quality, sanitizers, and matched model timings.
MTP remains explicitly unsupported for direct loading until its packed consumers,
shortlist mapping and recovery behavior are independently qualified.
