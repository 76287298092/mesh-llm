# StreamForward: whole-model single-stream execution

Status: limited model-equivalence and timing evidence for an opt-in (2026-09-28, issue 1393), source
`ab33f730e`, PTX SHA256 `04f03b9b…`. Default remains `legacy`. Bounded sanitizer checks now pass with the scope below; graph capture
qualification is still pending. Full-row exact graph capture/replay is now implemented
in the unvalidated working tree; the measurements below are eager-stream evidence only.

## Qualification, 2026-09-28

Carrack RTX 5090 (GPU0), exclusive: Ninfer and ComfyUI stopped and restored.
`qwen-stream-check` compares legacy and stream forwards from identical state.
Both the 106-token prompt plus 32 decode steps and the 512-token prompt plus 8
steps pass: every step's selected token, BF16 logit hash and every KV, conv,
recurrent and other state-region hash are bit identical.

`qwen-model-bench`, 256 fixed outputs, three repetitions each (spread under
0.2%), matched prompts from `reassess-20260927/matched-prompts.json`:

| Prompt | Execution | Prefill tok/s | Decode tok/s | Decode step first → last |
|---|---|---:|---:|---:|
| story, 106 tokens | legacy | 238.9 | 20.35 | 43.6 → 54.4 ms |
| story, 106 tokens | stream | 306.2 | 26.48 | 32.3 → 43.1 ms |
| pg19, 512 tokens | legacy | 295.0 | 15.07 | 61.3 → 71.7 ms |
| pg19, 512 tokens | stream | 328.4 | 18.17 | 49.8 → 60.4 ms |

Generated tokens are identical between executions for both prompts. Removing
per-operation allocation, synchronization, per-launch context push and name
lookup gives +30%/+21% decode and +28%/+11% prefill. The remaining decode step
grows about 11 ms over 255 positions. Existing event profiles make FP64 causal
attention the leading explanation, but this trial did not isolate that cause.
Long-context model performance is not yet measured for this path. Matched Ninfer MTP0 decode is 76.3
tok/s at both prompt lengths ([matched reference](../findings/ninfer-matched-20260928.md)).
Evidence: `evidence/reassess-20260928/stream-trial-2/`.

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

- `MESH_SPECIALIZE_EXECUTION=legacy|stream|graph` (default legacy) selects the path in
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

## Bounded sanitizer follow-up

At source `ab33f730e`, `stream-sanitizers-1` passed memcheck on a 17-input,
two-decode-step legacy/stream comparison; scorer memcheck also passed on a
17-token window. Racecheck reached the 8 GiB cgroup memory cap and was OOM-killed
before completion, not a correctness pass. The failure is retained.

`stream-sanitizers-2` passed racecheck and synccheck on a two-input,
one-decode-step legacy/stream comparison and on the 17-token scoring window.
Racecheck used `--force-synchronization-limit 1 --racecheck-num-workers 1` to
bound instrumentation memory: zero errors/warnings. Synccheck used default
scheduling and reported zero errors. These short instrumented checks do not
cover the complete 106/512-input timing trials or prove arbitrary-context safety.
Both services were restored to their initial active states after each trial.

## Whole-model exact graph implementation, 2026-09-28

Status: source implementation only, based on the supplied parent HEAD `30658a926`.
No Cargo, PTX compilation, CUDA execution, sanitizer, or throughput result was run
by this worker. Parent owns validation. Device/driver/toolchain/clocks for this
change: not measured. Existing eager-stream evidence above does not qualify it.

`MESH_SPECIALIZE_EXECUTION=graph` now prefills eagerly on StreamForward, then
captures one entire M=1 forward, including embedding, all layers, final vocabulary
projection, and GPU greedy. Capture records kernels and stream-ordered DtoD copies
but must not execute them. The token upload, position upload, result readback,
optional full-logit readback, validation, and cursor commit remain outside capture.
No per-token graph instantiation or node parameter updates are used.

`GraphDecode::capture(&mut StreamForward, &mut Session)` returns an exclusive
lease. It cannot accept another session on replay. The lease prevents eager arena
reuse and state release while the graph exists; StreamForward's module/function
and weight borrows retain those owners as well. The device u32 position remains
allocated through graph destruction. Drop drains first, destroys the executable
and graph, then releases the position and owner borrows. Pageable token/position
bytes have their own tested completion guard, so an early return or unwind drains
before releasing upload storage. Failed replay transactions poison the cursor and
do not advance its committed position. Capture failures poison the bound session.

The intentionally fatal prototype policy is: a failed stream drain attempts a
context drain. If the context drain completes, return the original error and keep
the cursor poisoned. If both fail, log both errors and exit **this harness process**
with status70 without unwinding any CUDA owners. No undocumented fatal-CUDA-error
semantics are assumed. No external process or service is controlled by this path.

### Position ABI and arithmetic boundary

Three entries in `kernels/nvptx/graph_position.rs` delegate to unchanged shared
exact bodies extracted from the original entries. No new inline assembly is added.

- `attention_qk_prepare_position`: original eight-pointer/five-u32/FP32-epsilon
  ABI, followed by a fifteenth argument `*const u32 past`. Cos/sin now address
  table bases; the wrapper offsets each by `*past * rotary_dim/2` BF16 elements.
- `attention_kv_append_position`: original four-pointer/three-u32 arguments,
  then `*const u32 past` **instead of** the original by-value past, then capacity.
- `causal_attention_bf16_position`: original five-pointer/four-u32 arguments,
  then `*const u32 past` **instead of** by-value past, capacity, and FP32 scale.

All three use the original grids/blocks with rows fixed at1. The host validates
capacity before replay. A same-stream HtoD updates position and token before the
single graph launch. Q/K preparation keeps every rounding boundary and gate copy;
KV remains BF16 token-major; causal attention retains its exact FP64 reduction
and softmax body. The assembly inventory records these reused sites explicitly.

Graph rejects SplitDecode and all other nonexact arithmetic. Existing stream
SplitAttention remains available and unchanged. FP8 embedding and F32 GDN weights
are source formats, not arithmetic profiles: their existing consumers and dtype
bindings are reused in the captured full-row path.

### Cost scope and qualification hook

`qwen-model-bench` captures once for each fresh, eagerly prefilled repetition.
`repetitions[].graph.capture_seconds` and `instantiate_seconds` are recorded outside
both prefill and decode intervals. Setup does not hide a replay. The report also
records full `decode_setup_seconds` (allocation, handle resolution, capture,
instantiation, and setup drain), a setup-inclusive decode interval rate, and real
decode/sequence wall makespans and rates including per-session setup. The wall
boundaries include host-loop and memory-checkpoint overhead; interval sums exclude
those checkpoints. Thus replay-only throughput cannot hide the first-request
capture cost. Each measured decode includes the two small HtoD uploads, one graph launch, one stream completion,
and selection readback. Warmup protocol has explicitly changed for **all three
executions**: full requested prompt plus one actual decode on a disposable fresh
session. Graph warmup captures and actually replays once. Every timed repetition
still starts with a fresh session, captures a new session-bound graph, and includes
its first replay in the timed interval list. The report labels the new protocol
and records warmup graph replay count separately. Timed generated sequences are
unchanged. Driver-internal allocation/JIT work is not instrumented. The absence of explicit per-token device allocation/free is
**source-derived**, not a measured allocation counter. The benchmark now reports
actual GPU greedy for stream/graph and separately retains the configured flag.

Run the separate equivalence mode with the existing command and argument order:

```text
MESH_SPECIALIZE_EXECUTION=graph xtask specialize qwen-stream-check --artifact P --tokens IDS --ptx P --device N --output NEW_FILE --decode-steps N
```

Use at least two decode steps. This mode compares exact eager stream to actual
full-model graph replay from identical fresh sessions and input tokens. It retains
full resident-weight readback diagnostics. It checks selected tokens, CPU greedy
agreement, **complete BF16 logits**, committed cursor, and SHA256 of **all bytes of
all state regions**, including unused KV suffix. A separate before/after capture
snapshot checks that capture/instantiation did not execute or advance anything.
The report requires the successful actual replay count to equal requested decode
steps; a residual-only probe cannot satisfy this gate. Default qwen-stream-check
continues to compare legacy versus stream.

Host tests added: u64 position-ABI packing and scalar order, poisoned replay cursor,
exact-only profile rejection, complete-output/state comparison, mock successful/
failed/unwinding completion before host-storage release, and double-drain fatal
policy. `reference/graph_position.rs` independently checks base/compact RoPE,
Q gates, nonzero/final KV append addresses, and causal-prefix versus poisoned-tail
attention at positions0/1/7/16/32. These are host contracts, not device qualification.
Parent still needs serial check/Clippy/tests, Just PTX generation, graph equivalence
on real weights (including direct Ninfer source formats), all three sanitizers,
matched timings, and allocation/resource-release evidence. No speedup is claimed.


## September 28 measured follow-up

Graph result at4e75df069: wholemodelcapture/replay matches eagerstream logits/state; setup-inclusive rates26.478→26.635 and18.165→18.213tok/s (106/512inputs), onlyabout0.6%/0.3%. Not a majorremainingbottleneck. Memcheck passes; graphracecheck reaches16GiB cgroupOOM, retained as incomplete. No graphdefaultpromotion.
