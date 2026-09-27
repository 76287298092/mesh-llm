# F08 CUDA stream and graph primitives

Status: bounded driver primitives implemented; parent registration, host-only
compilation, GPU replay qualification, and performance measurement remain pending.
This entry does not establish model execution, decode improvement, or performance
parity.

## Scope

`src/kernels/cuda/driver_graph.rs` adds an owned nonblocking stream, thread-local
stream capture, captured graph and executable graph RAII wrappers, and
`Function::launch_on_stream`. The wrappers borrow the existing thread-bound CUDA
`Context`, which retains the dynamically loaded driver API and its library. Each
driver call and destructor activates that context. Failed stream creation,
instantiation, and capture cleanup release any handle returned alongside an
error. Dropping a live capture ends it and destroys the partial graph; if context
activation fails during that cleanup, the stream stays marked as capturing and
retries cleanup when explicitly aborted or dropped.

Capture uses `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`. The stream launch path checks
nonzero grid and block dimensions, enforces that the stream and function share a
context, and makes no profiler-event or synchronization calls. It exposes driver
operation guards for the existing default-stream launch and synchronous host-side
driver methods. Parent integration must call those guards before default-stream
profiling/launch and before allocation, copy, synchronization, or event work.

The CUDA graph stores raw kernel function and device-address values copied during
capture. Rust cannot infer the true lifetime of those values from the current
argument ABI, so capture finalization, graph instantiation, and graph replay are
unsafe. Callers must retain every referenced module and allocation at a stable
address until the graph and all executable graphs are destroyed and all replayed
work completes. Callers must also keep each launch's host argument storage alive
through the driver call. This primitive is only suitable for fixed-shape probes
whose workspace addresses are already stable.

## API contract checked

The FFI declarations were compared with NVIDIA's CUDA Driver API 13.4 reference:
[stream management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html),
[graph management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html),
and the [CUDA driver header reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h.html).
The verified signatures use `CUstream` plus flags for stream creation,
`CUstream` plus `CUstreamCaptureMode` for capture start, `CUstream` plus an
output `CUgraph` for capture end, `CUgraphExec*`/`CUgraph`/`unsigned long long`
for `cuGraphInstantiateWithFlags`, and the documented stream or graph handle
arguments for synchronization, launch, and destruction. The nonblocking stream
flag is `1`; thread-local capture mode is `1`.

## Validation state

- Source revision: the delegated worktree's final integration commit was not
  available because this task prohibited Git; the parent should attach its exact
  commit when integrating the file.
- Model and shape: none; no model path or graph probe ran.
- GPU architecture, driver, and CUDA runtime behavior: not measured.
- Rust command: `rustfmt --edition 2024
  crates/mesh-specialize/src/kernels/cuda/driver_graph.rs` completed with
  `rustfmt 1.9.0`.
- Cargo compilation and tests: not run, as the worker was explicitly barred from
  using the build slot. The parent must register the child module and compile it.
- Expected probe: launch a tiny fixed-shape kernel on the owned stream, capture
  it, instantiate it, replay it, synchronize the stream, and compare output with
  the existing independent reference. No observed GPU result is claimed.
- Throughput, host-submission savings, model prefill/decode, sanitizers, and
  Ninfer feature speedup: not measured.

## Durable rules

- Keep stream capture and replay bounded to stable buffers and modules; do not
  integrate whole-model capture from this primitive alone.
- Do not allocate or synchronize on the host during capture. If a host operation
  is attempted, reject it with the exported driver-operation guard; CUDA capture
  errors still require the capture guard's failure cleanup.
- Route captured kernel launches through `Function::launch_on_stream`. The
  existing default-stream launch path must check
  `ensure_default_stream_launch_allowed` before it starts event profiling.
- Synchronize the replay stream before releasing captured allocations or modules.
- These wrappers establish no throughput result. A fixed-shape replay probe is
  a prerequisite to any later model-level graph integration or performance
  attribution.

Parent integration wired the child module and capture guards in driver.rs, plus a fixed-address graph-check command. Review changed manual abort to require exclusive stream access, preventing an old guard from ending a later capture. Linux compilation and GPU execution are next; no performance claim.
