# Selected CUDA device admission

Status: H02 policy and live selected-device trials pass. No specialized model is
loadable yet; resident discovery and actual engine/context binding remain pending.

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

## Live evidence

Carrack trial September 27, 00:09:09–00:09:21 EDT, source `7fa34d234`.
Host Rust 1.98.1 / LLVM 22.1.8, NVIDIA driver 615.71.09, CUDA Driver API 13040.
The built xtask SHA-256 is
`460f64fad2ac45dc7b193f87908bc5ac00cec0d44f854934f882e0b0cac1c3fc`.
Commands ran in temporary user scopes with `MemoryMax=8G`, `MemorySwapMax=0`
and a 90-second timeout. No profiler or clock measurement was needed for this
policy trial; no performance number is inferred from it.

| Actual selected device/state | Free bytes | Result |
| --- | ---: | --- |
| RTX 5090, Ninfer active | 693,960,704 | Rejected for insufficient free memory |
| Same RTX 5090 UUID, Ninfer stopped | 32,221,822,976 | Admitted for the synthetic 28 GiB requirement |
| RTX 3080, while 5090 also present | 10,158,145,536 | Rejected for SM86 and insufficient total/free memory |

The two 5090 snapshots report UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, matching the previous live inventory.
The 3080 reports `GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea` and 10,410,328,064
total bytes. The full [occupied](../evidence/admission-20260927/occupied-5090.json),
[free](../evidence/admission-20260927/free-5090.json), and
[wrong-GPU](../evidence/admission-20260927/3080.json) reports preserve the actual
host profile, selected-device evidence, exact rejection reasons and the fixture.
Toolkit independence is covered by unit tests with an empty toolkit inventory;
this live host has a toolkit installed, so the live trial alone cannot prove it.

All 14 representative kernel cases also passed with the refactored driver and
unchanged PTX. The [check-only report](../evidence/admission-20260927/workloads.json)
contains no new timings. ComfyUI PID 448118 remained running with 498 MiB, and no
other GPU compute process was present during the stopped-service checks. Ninfer
was restored and reported engine-ready at 00:09:31 EDT with its original
250,880-token FP8 KV capacity and port 1235 listener.

Validation: 70 native-runtime tests and 20 focused xtask tests pass locally and on
Linux; 26 macOS / 29 Linux kernel/reference tests pass. The 51 runtime-install
tests pass locally. Focused Clippy with warnings denied passes on both platforms,
including the Linux-only driver and validator. Repository no-console-print passes.
A bounded code review found no selected-device admission bypass. Bundle and cache
lookup now also require matching backend requirements; the regression checks
both sources. Full product build and real specialized startup remain unqualified.
Raw logs and process snapshots remain on both hosts under
`target/specialize/admission-20260927/`.

No new assembly site, vendor compute dependency or production catalog entry is
introduced by this change. The existing resolver remains oversized; the new
admission policy and tests live in separate capability-owned modules.
