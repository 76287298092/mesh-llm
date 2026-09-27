# Rust NVFP4 instruction probe

Status: source integrated, 14 independent arithmetic/packing tests passed on
macOS; CUDA execution pending. Base `dc1e24fe9`, Rust host 1.98.1, device compiler
nightly-2026-09-25 (rustc 1.100.0-nightly f7575a9da, LLVM 23.1.1).

The nested device workspace emits PTX8.7 for SM120a using Rust inline assembly.
The Linux host uses only dynamically loaded CUDA Driver API calls. It checks
two-context allocation isolation and rejects cross-context event timing. It
JIT-loads PTX with bounded logs and launches one 32-thread warp for each of four
logical fixtures and eight scale-selector pairs. Each fixture has 128 outputs.

The independent host reference decodes E2M1/UE4M3 logical row-major tensors and
accumulates a dense matrix product in f64. Packing uses the documented 16x8x64
lane mapping. Signed fixtures cover all 16 FP4 codes and nonuniform power-of-two
scales; unused scale lanes contain poison values. These dot products are exactly
representable in f32, so any numerical difference fails. This does not validate
all possible scales, GEMM performance, model inference, or context state.

Review corrections: CUDA context ownership needs scoped push/pop activation,
not just Rust borrows. JIT scalar options are pointer-sized values, not pointers
to integers. CUDA buffer copies check lengths and preserve the owner context.
No context/module/buffer/event may move between threads. Cleanup errors use
tracing and never panic.

Reproduction: `just specialize-ptx`, `just specialize-tools-build`, then
`target/release/xtask specialize nvfp4-probe --ptx target/specialize/probes.ptx
--device 0 --output NEW_FILE`. The JSON binds PTX SHA256, resource counts, JIT
logs, memory observations, every expected/actual output, and qualification status.
Kernel launch event times include submission gaps and are not model throughput.

References: [Rust NVPTX target](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html),
[NVIDIA matrix fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16864),
[CUDA module loading](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).
