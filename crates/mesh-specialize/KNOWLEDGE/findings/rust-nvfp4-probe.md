# Rust NVFP4 instruction probe

Status: executed successfully on carrack RTX5090 at revision
`149e1aaa6728eb3a786afd4c475eb0d740e54af6` on 2026-09-26. Rust host 1.98.1, device compiler
nightly-2026-09-25 (rustc 1.100.0-nightly f7575a9da, LLVM 23.1.1).

All 32 cases (4,096 output elements) matched exactly; maximum absolute error
was zero. Context isolation also passed. Driver 615.71.09 reports CUDA API
version 13040 and SM12.0. The driver JIT reported 20 registers, zero static
shared-memory bytes, and zero local-memory bytes. Event observations ranged
from 2.432 to 12.288 microseconds; this is correctness instrumentation, not a
stable kernel-throughput measurement. SASS and instruction-throughput profiling
remain pending.

Before module load and after case allocations were freed, `cuMemGetInfo`
reported 32,221,822,976 free bytes. Minimum during a case was 32,219,725,824:
a 2 MiB allocation-granularity difference for 1,536 bytes of payload. This
does not estimate full-model memory. Ninfer was stopped for the launch; only
the pre-existing ComfyUI process remained (498 MiB), and total device usage
before launch was 937 MiB. Idle SM clock was 195 MHz; clocks were not fixed.
Ninfer was restarted afterward and logged engine ready at 22:59:54 EDT,
PID 2519627. Its configuration was unchanged.

Evidence: [complete expected/actual results](../evidence/rust-nvfp4-probe-20260926.json).
PTX SHA256: `89aecb232e33dff2f66f3a81423a351541dcc260c61a4f55ebddcabc5543d0ee`.
Linux xtask SHA256: `945c250dc0bd5468c5825f8ece772243c1238329006142dd5de082a199e80cc0`.
Full device/process snapshots are retained at `target/specialize/probe-20260926/`
locally and on carrack. The PTX was cross-compiled locally and copied byte-for-byte;
the host harness was built on carrack from the pushed branch with `just`.

Validation: 14 specialized host tests plus 63 xtask tests passed locally; all
16 specialized tests passed on Linux. Both platforms passed focused Clippy with
warnings denied. Workspace roster, publish-chain, test-coverage and no-console
checks passed. Shellcheck passed for the roster script. No feature-branch CI
run was created by the pushes; this is local/remote validation, not green PR CI.

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
