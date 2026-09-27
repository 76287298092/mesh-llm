# First representative kernel trial

Status: source and host workloads integrated; GPU execution pending.

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

RMSNorm workloads cover widths1,7,255,256,257,5120 and rows1,3,17,128, plus a
zero-input case. Both runners perform a numerical launch, ten warmup launches,
and five timed batches of100 launches with CUDA events. Inputs stay resident;
event time includes default-stream host-submission gaps. Outputs record payload
bytes and driver-reported free memory with allocations live. Neither timing nor
memory is a full-model serving measurement.

Use `just specialize-ptx`, `just specialize-tools-build`, then
`xtask specialize workload-probe --ptx PATH --device 0 --output NEW_FILE`.
The source/PTX/toolchain/device revisions and numerical/timing results will be
recorded after the trial. Independent CUDA-library comparison is pending.
Nsight Compute/System executables were not found in carrack's PATH or the checked
`/opt/cuda/nsight*`, `/usr/local/cuda/nsight*`, `/opt/nvidia` locations; profiler
usability remains unqualified. System ptxas, nvdisasm and compute-sanitizer work.

Durable rule: keep the initial kernel as a measured correctness baseline before
changing tiling, register pressure, or asynchronous-copy scheduling.
