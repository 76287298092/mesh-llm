# llama-quant-ffi

`llama-quant-ffi` exposes the low-level Rust declarations for llama.cpp GGUF
quantization. It owns the native quantization types, symbol loading, and link
configuration used by [`skippy-quantize`](../skippy-quantize/README.md). Job
planning, conversion, manifests, and CLI behavior belong in `skippy-quantize`,
not this FFI crate.

The build script links the pinned patched llama.cpp static archives by
default. Prepare them with `just llama-build`, or point
`LLAMA_STAGE_BUILD_DIR` at a compatible native build. The optional
`dynamic-runtime` feature resolves quantization symbols from a compatible
runtime library instead of statically linking the archives. Native library
loading and FFI calls are unsafe; callers must use the same pinned ABI.

For the supported quantization workflow and build recipes, see the
[`skippy-quantize` guide](../skippy-quantize/README.md).
