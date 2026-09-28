# StreamForward: whole-model single-stream execution

Status: implemented, unqualified (2026-09-28, issue 1393). Source written by a
worker without running Cargo, a GPU, sanitizers or `qwen-stream-check`. Nothing
here is measured. The default execution path is unchanged (`legacy`).

## What it is

A second forward executor over the same resident weights, state and cursor as
`resident_model.rs`. It launches the same kernels with the same pointer order,
scalars, grids and blocks as the legacy default exact profile, but:

- One nonblocking `driver::graph::Stream`. The context is pushed once per
  forward (`Stream::enter` returns an `ActiveStream` guard), not per launch.
- All 24 kernel handles are resolved once at construction
  (`stream_forward/functions.rs`).
- One arena allocation holds every intermediate. Offsets come from a pure-host
  planner (`plan.rs`) over a forward template (`program.rs`); a separate
  persistent buffer holds the RoPE tables for every position up to capacity.
- The forward enqueues no allocation, free or synchronization. Kernel arguments
  are packed into a fixed stack array (`ops.rs::Args`), not a `Vec`.
- Selection runs `greedy_bf16_tiles`/`greedy_bf16_finish` on the same stream.
  The forward ends in one `cuStreamSynchronize` and one 16-byte readback.
  `forward(.., return_logits = true)` also downloads the BF16 logits for
  equivalence checks.
- State layout and transaction semantics are unchanged. The cursor
  transaction begins before enqueue and commits only after a valid selection.

## Construction preconditions

`StreamForward::new` fails with a named error unless every profile is default:
`MESH_SPECIALIZE_FP8_PROFILE=exact`, `NVFP4_PROFILE=baseline`,
`ATTENTION_PROFILE=exact`, `MLP_WORKSPACE=off`, `FP8_SPLIT_K=off`, and no
`NVFP4_AUDIT`. `MESH_SPECIALIZE_GPU_GREEDY` only affects the legacy path.
`max_rows` is 1..=512. Callers must ensure no legacy-stream work on the session
is pending (`Session::new` and legacy forwards end with a context synchronize).

## Kernel selection (mirrors legacy default profile)

| Rows | FP8 linear | NVFP4 linear |
| --- | --- | --- |
| 1 | `fp8_linear_exact` (1x4 tile, 128 threads) | `nvfp4_decode_exact` (1x4, 128) |
| 2..=3 | `fp8_linear_exact` | `nvfp4_linear` (16x8, 32) |
| 4..=15, channels >= 16,384 | `fp8_verify_exact` (8x16, 32) | `nvfp4_linear` |
| 4..=15, channels < 16,384 | `fp8_linear_exact4` (4x4, 128) | `nvfp4_linear` |
| >= 16 | `fp8_prefill_exact` (16x8, 32) | `nvfp4_linear` |

Two deliberate storage-only differences from legacy, both argued bit-identical:

1. The head norm gathers the last hidden row through the persistent row-ID
   table (pointer `row_ids + 4*(rows-1)`) instead of a DtoD copy into a one-row
   buffer followed by row ID 0. `embedding_norm_bf16` reads
   `table[ids[row] * width + column]`, so the element values are the same.
2. RoPE tables are built once for positions `0..capacity` with the same
   `TextRope::tables` in 2,048-row chunks and addressed at `past`. Tables are
   position-local (`engine/rope.rs` partition test).

The GDN convolution still writes next history to scratch and copies it into
state, now with a stream-ordered `cuMemcpyDtoDAsync_v2`.

## Arena planner

Every buffer gets an inclusive step interval from first write to last read.
Inputs and outputs of the same operation share a step, so they never alias.
Placement is largest-first, first-fit at 256-byte alignment, followed by an
O(n^2) validation that no two lifetime-overlapping buffers share bytes. The
template runs one GDN block, one attention block, one FP8 MLP, one NVFP4 MLP,
the head and greedy. Only `hidden`, `post.sum`/`post.x`, tokens, row IDs,
logits and the greedy result cross sections; their template lifetimes cover
every real layer ordering. Dead diagnostic outputs (FP32 raw, residual copies,
SiLU/sigmoid intermediates) still get storage because the kernels write them.

Size (Python port of the planner, same algorithm; the Rust tests assert the
lower bound and placement validity, not these numbers):

| max_rows | 121 buffers placed, arena bytes | peak-live lower bound |
| --: | --: | --: |
| 1 | 1,506,048 | 1,505,292 |
| 16 | 4,948,480 | 4,948,096 |
| 128 | 39,584,768 | 39,584,768 |
| 512 | 158,339,072 (151.0 MiB) | 158,339,072 |

For rows >= 16 the peak is the MLP SiLU-product step:
`arena = M * (16*I + 6*H + 8)` with `I = 17,408`, `H = 5,120`: gate, up, product,
FP32 SiLU, BF16 activated, FP32 raw (16 bytes per element), plus `hidden`,
`post.sum`, `post.x` and the token/row-ID words. RoPE adds `capacity * 128`
bytes (for example 65,664 B at capacity 513, 32 MiB at 262,144).

## Remaining host transfers and syncs per forward

- Token IDs: one `cuMemcpyHtoDAsync_v2` from pageable memory, `4 * rows` bytes.
- One `cuStreamSynchronize`.
- Selection: one synchronous 16-byte `cuMemcpyDtoH` after the synchronize.
- Optional diagnostic logits: `2 * vocabulary` bytes (496,640 B).
- Host-only work: per-layer state-region lookups by name and argument packing.
  Two context push/pop pairs (the stream guard and the readback).

## Capture blockers for the M=1 forward

- Token upload from pageable memory is not capturable; it needs a pinned staging
  buffer or a device-side copy from the previous greedy result into the token slot.
- `past` is a by-value argument to `attention_kv_append` and
  `causal_attention_bf16`, and the RoPE pointer is `base + past*64`. Both change
  every step and need device-resident position variants or per-replay
  `cuGraphExecKernelNodeSetParams`.
- State addresses depend on the session's state arena. A graph binds one session.
- The readback, result validation and cursor commit stay outside the graph.
- Kernel choice depends on rows only, so a fixed-M graph is structurally stable.

## Hooks

- `MESH_SPECIALIZE_EXECUTION=legacy|stream` (default legacy) selects the path in
  `qwen-model-bench`; the report gains an `execution` object with the arena size.
  The stream arena is sized for the prompt length.
- `xtask specialize qwen-stream-check --artifact P --tokens IDS --ptx P --device N
  --output NEW_FILE --decode-steps N` runs legacy and stream from identical zeroed
  sessions (prefill, then N greedy steps fed with the legacy token), recording per
  step token equality, logit SHA-256, every state region's SHA-256 (with
  KV/conv/recurrent equality summaries) and wall times. Pass means bit-identical.
  Legacy uses CPU greedy unless `MESH_SPECIALIZE_GPU_GREEDY=on`; the report also
  records CPU greedy over the stream logits.

## Files

- `src/kernels/cuda/stream_forward.rs` (executor, RoPE table, profile gate)
- `src/kernels/cuda/stream_forward/{plan,program,functions,weights,ops,layers,bench,check}.rs`
- `src/kernels/cuda/driver_graph.rs` (`ActiveStream`: `enter`, `launch`,
  `copy_device`, `copy_from_host`, `synchronize`; two async memcpy symbols)
- `src/kernels/cuda/mod.rs` (module declaration)
- `src/kernels/cuda/resident_model_bench.rs` (execution switch)
- `src/kernels.rs` (`StreamCheckRequest`, `stream_forward_check`)
- `tools/xtask/src/specialize.rs`, `tools/xtask/src/specialize/stream_check.rs`

## Open qualification work

Type-check, host unit tests, `qwen-stream-check` at short and 512-token prompts
with decode steps, compute-sanitizer memcheck/racecheck on the stream path,
and matched `qwen-model-bench` timings for both executions.
