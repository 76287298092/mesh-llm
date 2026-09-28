# Exact FP8 vector16 decode schedule

Status: implemented, not compiled or GPU-qualified by the worker. Default remains
baseline. Parent owns builds, GPU/reference/sanitizer trials, actual model checks,
and before/after model timing. No speedup or bandwidth result is established.

## Contract

`fp8_linear_exact_vector16` has the unchanged nine-argument ABI:
`(A:u8*, W:u8*, SA:f32*, SW:u16*, out:u16*, raw:f32*, M:u32, N:u32, K:u32)`.
A is K bytes and W is row-major N*K bytes. M must equal 1, N is 1..=262144,
K is a multiple of 16 in 16..=32768, and both A and W are 16-byte aligned.
All codes must be finite E4M3FN (not 0x7f or 0xff). Scale validity, nonaliasing,
output extents, lifetime, and element alignment match the exact control.
Launch grid is `[ceil(N/4), 1, 1]`, block `[128, 1, 1]`, dynamic shared bytes 0.

Each warp owns one channel. Lane L loads 16 bytes at K offsets `16*L + 512*j`,
using one aligned vector instruction for A and one for W. Four independent i64
chains each accumulate the four code pairs from one u32 word. Their sums feed the
unchanged integer warp reduction. K alignment prevents partial-vector overreads;
the N-tail branch is warp-uniform and invalid warps still join the reduction.

Finite E4M3FN magnitudes in units of 1/512 are at most 229376. For K<=32768,
the sum of absolute integer products is at most `32768*229376^2 < 2^51`.
Every partial and combined sum fits i64. Reassociation changes no integer result.
The control's rounded i64-to-FP32 conversion, power-of-two reconstruction,
FP32 row-scale then channel-scale multiplies, and BF16 RNE helper are reused
unchanged. No BF16/A16 activation, quantization, scale, or arithmetic-profile
change is included. The baseline kernel source is untouched.

## Admission and reporting

`MESH_SPECIALIZE_FP8_DECODE_SCHEDULE=baseline|vector16` is read once per process,
separate from `MESH_SPECIALIZE_FP8_PROFILE`. Default is baseline; unknown values
fail explicitly. Vector16 admits only `Profile::Exact`, M=1, supported K, and
aligned nonnull A/W. Unsupported K/alignment falls back to the baseline. Other
arithmetic profiles and multirow calls keep their previous kernels.

Wiring covers `resident_fp8::Projection` and `stream_forward` projections.
The stream resolves the additional function only for explicit opt-in, and
`stream.report.fp8_decode_schedule` records requested schedule, resolved handle,
actual host selection counts, and width/alignment fallbacks. These are enqueue
selection decisions, including graph capture but excluding graph replay, not GPU
launch counts. Legacy resident selection emits a debug event containing the
actual kernel and reason; launch profiling uses the actual kernel name.

Existing exact graph capture can record the selected schedule. Graph lifecycle,
positions, and state are untouched. Legacy MTP's calls through resident_fp8 may
select it only for exact M=1; multirow target verification is unchanged. Existing
MLP workspace/prepared projection paths are not wired and remain baseline, even
with opt-in. No claims of full-model or MTP equivalence have been measured yet.

## Independent trial and reproduction

Use the existing `fp8-exact-check` command with a separate explicit trial hook:

```sh
MESH_SPECIALIZE_FP8_DECODE_TRIAL=1 \
MESH_SPECIALIZE_FP8_DECODE_TRIAL_HEAD=1 \
  target/debug/xtask specialize fp8-exact-check \
  --ptx PATH_TO_CURRENT_PTX --device 0 --output NEW_VECTOR16_JSON
```

Use the parent-built binary path appropriate to the build. Set
`MESH_SPECIALIZE_FP8_DECODE_TRIAL_TIMING=0` for sanitizer/check-only runs.
The hook returns a new `fp8-exact-vector16-schedule-trial` JSON, rather than
changing the existing component `fp8_exact_trial::run` used by model/MTP checks.
The standalone trial calls both entrypoints directly regardless of schedule env.

Checks require exact raw FP32 bits and BF16 bits against both the unchanged GPU
control and the independent existing `reference/projections.rs::linear` oracle.
That oracle decodes logical E4M3 values and accumulates in FP64; it does not use
the candidate's packed loads or integer helpers. The trial covers:

- All 64,516 finite-code pairs, individually isolated across the 16 byte positions
  and fully checked by the CPU oracle. This includes negative zero/subnormals.
- Irregular finite inputs at K16/32/128/5120/6144/17408 with N1/3/4/5/17.
- Positive/negative even/odd BF16 midpoint ties, large cancellation with subnormal
  residuals, and maximum signed magnitudes at K16/128/32768.
- Synthetic real-model shapes N*K: 10240*5120, 6144*5120, 5120*6144,
  12288*5120, 1024*5120, 17408*5120, and 5120*17408.
- Optional vocabulary head 248320*5120. The head flag defaults to 0 and its skip
  reason is explicit. Opt-in requires GPU free-memory headroom; the largest host
  and device W payload is 1,271,398,400 bytes each, with no full-sized W clone.

Each case performs three runs per kernel with repoisoned outputs and 32-byte
canaries before/after every allocation. The trial checks all output bits against
the unchanged control, every repeated output, all output finiteness, and input/
output canaries. Canaries detect writes, not out-of-bounds reads; memcheck remains
required. The CPU oracle checks all outputs in small-N and finite-pair cases;
real-shape cases explicitly sample 257 evenly spread columns including the first
and last. JSON records exact sampled indices, output/product counts, and whether
CPU coverage is full. No full CPU head coverage is claimed.

Timing follows successful correctness checks at real model shapes: three warmups,
three samples of 20 launches per kernel, alternating control/candidate sample
order. Buffers, functions, and CUDA events are prepared outside event intervals.
Timings exclude CPU oracle, upload/readback, poisoning, and allocations. Event
batches can include submission gaps and establish operator timing only, not model
throughput or achieved bandwidth. Reported resources include JIT log, registers,
static shared bytes, local bytes, maximum block threads, and dynamic shared bytes.

## Assembly inventory and pending evidence

Two new inline assembly sites live in `fp8_linear_exact_vector16.rs`:

1. `coordinates`: reads `%laneid`, `%tid.x`, and `%ctaid.x`; no memory effects.
2. `load16`: `ld.global.v4.u32`; readonly 16-byte aligned global load, caller owns
   complete span/lifetime. Four output u32 registers, no shared/local allocation.

Integer product/add, shuffle, conversion, and FP32 scale sites are reused from
`fp8_linear_exact.rs` and remain inventoried there. Independent reference is
`reference/projections.rs`; scheduling coverage/bounds/admission unit tests live
in `src/kernels/fp8_decode_schedule.rs`.

Worker environment: local macOS source editing, 2026-09-28. Git revision,
PTX hash, compiler, GPU/driver, resource values, clocks, sanitizer results,
operator timings, and model comparisons: not measured by this worker. No Cargo,
Git, SSH, GPU, service changes, or NInfer compute source import was performed.
Parent should pin the integrated source/PTX in the resulting evidence and run
memcheck/racecheck/synccheck plus exact model logits/state/token and MTP recovery
checks before interpreting any timing. A component pass alone is not promotion.
