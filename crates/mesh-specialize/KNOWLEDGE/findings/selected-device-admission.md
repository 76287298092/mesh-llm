# Selected CUDA device admission

Status: H02 implementation in progress. No specialized model is loadable yet.

Driver-only runtimes explicitly declare `backend.cuda.driver_only`, a minimum
CUDA Driver API version and a minimum device-memory budget. The existing
`toolkit_major` is zero in this mode, so it cannot be confused with a dependency
on CUDA runtime libraries. `min_driver` must be absent; the numeric API floor is
the version returned by `cuDriverGetVersion`, not an NVIDIA driver release string.
The prototype also requires nonempty exact `serves` identities and numeric
compute architectures such as `sm_120`. The architecture-specific PTX target
`sm_120a` remains a kernel build/JIT contract, not a numeric device property.

Existing CUDA manifests without `driver_only` retain their current toolkit and
host-architecture policy. Driver-only manifests use the selected device's UUID,
CUDA-visible ordinal, architecture, driver API version, total memory and current
free memory. A second compatible GPU cannot qualify an incompatible selected GPU.
Unknown/malformed device evidence and insufficient memory reject the candidate.
Memory checks are snapshots; actual allocation and keeping execution bound to the
same context remain the engine's responsibility.

`model_selection::ModelRuntimeRequest` carries the exact model identity and optional
selected-device evidence. Identity-only calls cannot admit a driver-only runtime.
The main host's resident `.mspec` discovery/startup integration is still pending;
these pure APIs neither load a model nor download a runtime.

The Rust driver now queries `cuDeviceGetUuid_v2` from the same `CuDevice` used to
create its context. `kernels::device_probe` creates a temporary context, snapshots
its properties and memory, then destroys it without launching a kernel. The
`xtask specialize admission-probe` harness combines this live evidence with a
clearly labeled synthetic policy fixture. It does not verify any model artifact.

The [trial fixture](../fixtures/cuda-admission.json) requires SM120, CUDA API 13040
and 28 GiB total/free memory. These are experiment thresholds, not a measured
Qwen memory budget. Expected trials: reject the occupied 5090 while Ninfer runs;
admit that same UUID after Ninfer stops; reject the 3080 even when the host also
contains a 5090. All results must record `model_artifact_verified: false`.

GPU measurements and exact validation revisions will be appended after execution.
No new assembly site, vendor compute dependency or production catalog entry is
introduced by this change. The existing resolver remains oversized; the new
admission policy and tests live in separate capability-owned modules.
