# Specialized runtime development

Read `KNOWLEDGE/README.md`, `PLAN.md`, and relevant knowledge entries before editing.
The parent agent owns design decisions. Workers receive one bounded deliverable
with exact interfaces, owned paths, tests, and a stop condition. Do not expand a
worker's scope or dispatch further workers.

- The prototype is local/internal and must not enter native release catalogs.
- New engine and kernels use Rust only. No C/C++, nvcc, CMake, or vendored compute
  library. Dynamically loaded system CUDA driver functions belong under T0.
- `src/kernels/` owns device-specific behavior; `kernels/nvptx/` contains Rust
  device source compiled separately for NVPTX. Normal host builds need no GPU.
- `src/engine/` will own generic execution/session behavior. Model shapes and
  schedule belong under `src/packages/qwen3_8_27b/`. Do not create those modules
  before their gated implementation tasks.
- `reference/` contains independent arithmetic/oracle evidence. A kernel cannot
  certify itself. No NInfer source or `.ninfer` parser belongs here.
- Every PTX site needs an entry in `KNOWLEDGE/asm-inventory.md` and a reference.
- Record findings and dead ends with exact revisions, device/toolchain, commands,
  expected/observed behavior, and limitations. Never delete failed evidence.
- Keep functions within workspace Clippy limits and source files below 1,000 lines.
- Run Cargo serially. Workers must not run Cargo unless the parent assigns the
  build slot. The parent integrates, validates, commits, pushes, and uses SSH.
- Never report synthetic arithmetic or kernel timing as model prefill/decode.
- Add or update a knowledge entry for every change, or explain `kb: none`.
