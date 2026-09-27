# Remaining instruction qualification

Status: source integrated; GPU execution pending. This entry records the next
bounded trial after the passing single-warp NVFP4 probe at `149e1aaa6`.

The Rust device module adds asynchronous global-to-shared copies, x2/x4 matrix
loads in both orientations, BF16/FP16/INT8 MMA, and register release/reacquisition.
Host fixtures generate independent logical expected outputs. The register probe
requires a complete 128-thread warpgroup and refuses to launch unless the driver
reports at least 64 registers per thread. A JIT register limit is a cap, so its
presence alone does not prove that precondition.

The same PTX compilation includes source for a first RMSNorm and tiled NVFP4
GEMM. Their host workloads and numerical execution are still pending. No kernel
timing is a model prefill/decode measurement.

Build with `just specialize-ptx` using nightly-2026-09-25, LLVM23.1.1, and
`sm_120a`. On carrack, build the host through `just specialize-tools-build`, then
run `xtask specialize instruction-probe --ptx PATH --device 0 --output NEW_FILE`.
`just specialize-sass PATH` invokes system ptxas and nvdisasm only on Rust-emitted
PTX for offline inspection; it compiles no CUDA C/C++ source. The launch path
continues to use the CUDA driver's JIT. Sanitizer checks will use the same typed
runner and a separate evidence output file.

Target: carrack RTX5090, SM12.0, driver615.71.09. Exact trial source revision,
PTX hash, actual tool versions, clocks and results will be recorded after running.
Model recipe: not applicable to instruction qualification. Throughput and context
capacity: not measured by these probes.

Durable rule: emitted PTX and passing host fixture tests are prerequisites;
qualification requires independent numerical results from the target GPU.
