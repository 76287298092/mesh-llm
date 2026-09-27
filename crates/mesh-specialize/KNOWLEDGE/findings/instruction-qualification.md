# Remaining instruction qualification

Status: all 29 cases pass on carrack, including memcheck, synccheck and racecheck.
This is the next bounded trial after the passing NVFP4 probe at `149e1aaa6`.

The Rust device module adds asynchronous global-to-shared copies, x2/x4 matrix
loads in both orientations, BF16/FP16/INT8 MMA, and register release/reacquisition.
Host fixtures generate independent logical expected outputs. The register probe
requires a complete 128-thread warpgroup and refuses to launch unless the driver
reports at least 64 registers per thread. A JIT register limit is a cap, so its
presence alone does not prove that precondition.

The same PTX compilation includes source for a first RMSNorm and tiled NVFP4
GEMM. Their host workloads are now being integrated; numerical execution remains
pending. No kernel timing is a model prefill/decode measurement.

Build with `just specialize-ptx` using nightly-2026-09-25, LLVM23.1.1, and
`sm_120a`. On carrack, build the host through `just specialize-tools-build`, then
run `xtask specialize instruction-probe --ptx PATH --device 0 --output NEW_FILE`.
`just specialize-sass PATH` invokes system ptxas and nvdisasm only on Rust-emitted
PTX for offline inspection; it compiles no CUDA C/C++ source. The launch path
continues to use the CUDA driver's JIT. Sanitizer checks will use the same typed
runner and a separate evidence output file.

## Measured evidence

Trial source: `817bfa2c0e823ee7fa45a72dff69360f1f74e789`; device source is unchanged
from `91bccbd99`. PTX SHA256:
`227c8911e5ada033a679802934385cd5e007a28ba2fbb399eb9558b4dbb7999a`.
Linux xtask SHA256:
`fd71cb52350db490f46f2fe14261ba1600b457904b9991848bf24b06d0e22d4a`.
Host Rust1.98.1/LLVM22.1.8, device nightly-2026-09-25/LLVM23.1.1,
offline ptxas13.4.92 and matching CUDA sanitizer tools. Target: carrack RTX5090,
SM12.0, driver615.71.09 (driver API13040). Trial began September26 at23:18EDT;
evidence filenames use September27 UTC. Before launch:195MHz,13.60W,29C,
937MiB total GPU memory; Ninfer stopped and ComfyUI's498MiB remained untouched.
No continuous clock/power sampling was collected for these correctness probes.

The [full report](../evidence/rust-instructions-20260927.json) contains3,712
exact output matches:16 shared-copy/load cases,12 ordinary MMA cases, and one
128-thread register-budget case. JIT resources: shared probe14registers/512bytes
shared; each ordinary MMA18registers; register probe64registers. All report zero
local memory. The JIT honored the register allocation required for this probe.

Separate instrumented runs produced zero errors in
[memcheck](../evidence/rust-instructions-memcheck-20260927.txt) and
[synccheck](../evidence/rust-instructions-synccheck-20260927.txt), and zero hazards
in [racecheck](../evidence/rust-instructions-racecheck-20260927.txt). All three
instrumented numerical reports also passed. Full raw artifacts remain on both
hosts under `target/specialize/instructions-20260927/`.

Offline SASS includes `LDGSTS.E.128`, `LDGSTS.E.BYPASS.128`, normal/transposed
`LDSM.16.M88/MT88.2/4`, `USETMAXREG.DEALLOC.CTAPOOL`,
`USETMAXREG.TRY_ALLOC.CTAPOOL`, BF16/FP16 `HMMA.16816` and signed
`IMMA.16832.S8.S8`. ptxas reports zero spills for all entries, including the
not-yet-executed GEMM/RMSNorm. This offline cubin was inspected, while actual
execution used driver-JIT PTX; they are separate artifacts.

The remote command's trailing log-display command failed because of a shell
`tail` interpretation; the four GPU runs had already succeeded. The EXIT trap
restored Ninfer, which logged engine-ready and its1235 listener at23:18:14EDT.
Linux20 unit tests and focused Clippy with denied warnings passed before launch.
No GitHub Actions run was required by the feature-branch push.

Model recipe: not applicable. Throughput and context capacity are not measured
by these probes. A complete performance kernel still needs independent library
comparison, profiling, and model integration.

Durable rule: emitted PTX and passing host fixture tests are prerequisites;
qualification requires independent numerical results from the target GPU.
