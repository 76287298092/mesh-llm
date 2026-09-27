# First representative kernel trial

Status: all 14 GPU workloads pass. Memory, race and synchronization checks pass;
one failed racecheck attempt is retained below. Full-model performance is pending.

The untuned NVFP4 kernel assigns one warp to a 16x8 output tile and loops over
64-element K tiles with four live FP32 accumulators. Inputs use the already
qualified packed fragment layout. It deliberately starts with ordinary global
loads; asynchronous shared-memory tiling remains a separate tuning task.

The RMSNorm kernel uses one 256-thread block per row, a shared-memory sum of
squares, rounded FP32 division/square-root, and learned-weight multiplication.
The independent host arithmetic accumulates in f64. Its acceptance tolerance is
`2e-6 + 2e-6 * abs(expected)`; all outputs must be finite.

GEMM workloads cover (M,N,K)=(1,8,64),(17,13,71),(32,24,192),
(1,5120,5120),(128,5120,5120). Signed FP4 values and varying power-of-two scales
exercise every output tile and padding. The logical reference exploits repeating
row/column classes to compute all expected outputs efficiently; it does not read
packed buffers. Small tests compare that shortcut with an independent dense
matrix product. The chosen values have exactly representable sums, so the GPU
comparison requires exact equality, including zero output in padded rows/columns.

RMSNorm workloads cover widths 1, 7, 255, 256, 257, 5120 and rows 1, 3, 17, 128, plus a
zero-input case. Both runners perform a numerical launch, ten warmup launches,
and five timed batches of 100 launches with CUDA events. Inputs stay resident;
event time includes default-stream host-submission gaps. Outputs record payload
bytes and driver-reported free memory with allocations live. Neither timing nor
memory is a full-model serving measurement.

Use `just specialize-ptx`, `just specialize-tools-build`, then
`xtask specialize workload-probe --ptx PATH --device 0 --output NEW_FILE`.
For sanitizers, use `workload-check` with the same arguments to execute each
numerical case once, with no timing repetitions. Independent CUDA-library
comparison is pending.
Nsight Compute/System executables were not found in carrack's PATH or the checked
`/opt/cuda/nsight*`, `/usr/local/cuda/nsight*`, `/opt/nvidia` locations; profiler
usability remains unqualified. System ptxas, nvdisasm and compute-sanitizer work.

## Measured trial

Ordinary trial source: `e0486677c`; host Rust 1.98.1 / LLVM 22.1.8.
Device source was unchanged from `91bccbd99`, compiled with
nightly-2026-09-25 / LLVM 23.1.1. PTX SHA256:
`227c8911e5ada033a679802934385cd5e007a28ba2fbb399eb9558b4dbb7999a`.
Linux xtask SHA256:
`6c3a031a92b9f3c53a16578a2037b2a75fde406ad57a3a2273295e33a3f78846`.
Hardware: carrack RTX 5090, SM12.0, driver 615.71.09, CUDA tools 13.4.92.
Trial interval: September 26, 23:24:33.091–23:24:33.682 EDT.

The [report](../evidence/rust-workloads-20260927.json) checks 1,489,305 outputs:
738,688 GEMM outputs including padding match exactly, and 750,617 RMSNorm outputs
pass the stated tolerance. RMSNorm's maximum absolute error is 4.76837158203125e-7.
Driver JIT reports 36 registers and no shared/local memory for GEMM; RMSNorm uses
22 registers, 1,024 bytes shared memory and zero local memory. The previous offline
assembly used a different register cap and reported 57 GEMM registers; that cubin
is not the artifact used for these launches.

| Workload | Median event microseconds per launch | Payload bytes |
| --- | ---: | ---: |
| GEMM M1 N5120 K5120 | 9.6896 | 20,039,680 |
| GEMM M128 N5120 K5120 | 44.1677 | 22,691,840 |
| RMSNorm 1 row, width 5120 | 4.4947 | 61,440 |
| RMSNorm 17 rows, width 5120 | 4.8640 | 716,800 |
| RMSNorm 128 rows, width 5120 | 5.2093 | 5,263,360 |

These are preliminary resident-input timings. A separate `target/release/mesh-llm`
process, PID 2585247, used 1,010 MiB on the GPU at trial start. It was not controlled
by this experiment and later exited. ComfyUI PID 448118 retained 498 MiB. Ninfer
was stopped. Therefore this was not an exclusive performance run. Six 100 ms
samples within the short trial reported 2,542–2,587 MHz, 51.68–58.22 W and 30 C;
these coarse samples cannot resolve per-kernel peaks or exclude interference.

Driver free memory was 31,155,093,504 bytes before loading the module and after
all cases. The lowest case observation was 31,125,733,376 bytes, a 28 MiB decrease.
Largest tensor payload was about 21.64 MiB. This includes neither model weights
nor KV/recurrent state, and says nothing about usable model context.

## Sanitizers and recovery

[Memcheck](../evidence/rust-workloads-memcheck-20260927.txt) passed all 14 cases
and timing repetitions with zero errors. The first racecheck attempt exhausted
host memory while tracing repetitions. Its [failed log](../evidence/rust-workloads-racecheck-failed-20260927.txt)
and [kernel OOM record](../evidence/rust-workloads-racecheck-oom-20260927.txt) are
preserved. See the [failure and fix](../dead-ends/racecheck-timing-repetitions.md).

Recovery source `38d656cd1` adds check-only execution; device PTX is unchanged.
Recovery xtask SHA256:
`cb1d5958aeb7433ca7f8c34adbeba45738d17ac0ba66169c1d8015c8b1ecaee3`.
Each checker ran in its own temporary user scope with `MemoryMax=8G`,
`MemorySwapMax=0` and a 90-second timeout. [Racecheck](../evidence/rust-workloads-racecheck-20260927.txt)
reported zero hazards; [synccheck](../evidence/rust-workloads-synccheck-20260927.txt)
reported zero errors. Both complete numerical reports passed. Only ComfyUI was
present before and after those runs; timings were disabled.

Ninfer was restored and reported engine-ready at 23:29:09 EDT, with its original
250,880-token FP8 KV allocation, 19.7 GiB weights and 9.27 GiB runtime allocation.
The process used 30,046 MiB, and its port 1235 listener was ready. Raw reports,
hashes, GPU samples and process snapshots remain on both hosts in
`target/specialize/workloads-20260927/`.

Validation: 24 macOS / 26 Linux unit tests, 18 focused xtask tests, focused Clippy
with warnings denied on both platforms, and the no-console-print repository
check passed. Linux exposed two mechanical integration fixes before launch
(visibility and fixed-array chunks); neither changed kernel arithmetic.

Durable rule: keep the initial kernel as a measured correctness baseline before
changing tiling, register pressure, or asynchronous-copy scheduling.
