# Independent CUDA-library reference

Status: 14 independent library cases and 14 Rust workload cases pass on Carrack.

`mesh-specialize-validate` is a separate Cargo binary, gated by the explicit
`validation` feature and built with `just specialize-validation-build`. Its
cuBLAS module is not part of `src/lib.rs` and is never compiled into the runtime
library. The binary loads an explicit system cuBLAS path dynamically. It shares
the Rust driver ownership code and logical fixtures by source path; it does not
share GPU arithmetic with the Rust kernels. No new public device API is needed.

The binary first runs each Rust workload once, destroys that context, then runs
cuBLAS in a fresh context. Both paths compare all outputs with the same logical
f64 oracle. GEMM uses independently decoded FP32 matrices and pedantic SGEMM;
this is a numerical reference, not an NVFP4 performance competitor. RMSNorm
uses cuBLAS norm, diagonal multiplication and scaling, with a host f64 scalar
calculation between norm and scaling. Exact GEMM and the existing RMSNorm
absolute-plus-relative tolerances remain unchanged.

The wrapper selects host scalar pointer mode and pedantic math. Every call and
handle cleanup activates its borrowed driver context. These contracts follow
the installed `/opt/cuda/include/cublas_api.h` and the
[NVIDIA cuBLAS reference](https://docs.nvidia.com/cuda/cublas/).

The September 26/27 trial used source `f01846123753e61a4cac33704f995cff7cec4276`,
Rust 1.98.1 / LLVM 22.1.8, RTX 5090 SM120, driver 615.71.09 (CUDA API 13040),
and cuBLAS API version 130800. The pinned Rust device PTX remained unchanged at
SHA-256 `227c8911e5ada033a679802934385cd5e007a28ba2fbb399eb9558b4dbb7999a`.

Reproduce with `just specialize-validation-build`, then run the binary with
`--ptx target/specialize/probes-91bccbd99.ptx --library /opt/cuda/lib64/libcublas.so.13
--device 0 --output NEW_FILE` inside `systemd-run --user --scope` with
`MemoryMax=8G`, `MemorySwapMax=0`, and `timeout 90s`, while Ninfer is stopped.
The trial retained a shell exit trap to restart Ninfer. Only the pre-existing
ComfyUI process (498 MiB) remained on the GPU. Ninfer reported engine-ready at
23:47:16 EDT, with its original 250,880-token FP8 KV capacity.

All five cuBLAS GEMM cases matched the f64 oracle exactly. All nine cuBLAS RMSNorm
cases passed `2e-6 + 2e-6 * abs(reference)`; maximum absolute error was
`4.76837158203125e-7`. The Rust results independently passed their existing
criteria in the same executable. GEMM's cuBLAS comparison covers logical outputs;
the Rust test additionally checks padded tile outputs. There is no direct
cross-implementation output subtraction: both are checked against the independent
oracle. No model, throughput, clock, or profiler result is claimed by this trial.

A separate release library build with the `validation` feature enabled contained
no cuBLAS symbols (`nm -A`) or cuBLAS library/API strings (`strings`). This checks
the compiled runtime artifact in addition to the source-level binary boundary.
The local and remote raw evidence is under `target/specialize/library-20260927/`;
the full result and exact hashes are preserved in
[committed evidence](../evidence/library-20260927/results.json).
This closes the library-reference portion of the kernel gate. Profiler
qualification and full-model performance remain open.

The subsequent repository console-output check found a direct stdout handle in
the standalone validator's redundant success summary. That write was removed;
the durable JSON output file and failure exit status remain the result contract.
This changes reporting only, not the recorded GPU arithmetic or comparison.
