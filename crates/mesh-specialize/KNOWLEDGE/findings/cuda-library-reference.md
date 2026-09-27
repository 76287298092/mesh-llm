# Independent CUDA-library reference

Status: validation-only executable being integrated; GPU comparison pending.

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

Run the binary with `--ptx PATH --library /opt/cuda/lib64/libcublas.so.13
--device 0 --output NEW_FILE` inside a temporary memory-limited scope while
Ninfer is stopped. Results, actual library version, hashes and service restoration
will be recorded after execution. This closes only the library-reference portion
of the kernel gate; profiler validation and model performance remain separate.
