# Full-decode kernel attribution

Status: the instrumented decode matches the control exactly. FP8 projections
account for 83.39% of the summed CUDA-event intervals in this short-prefix run.
The preceding [model timing](model-timing-20260927.md) measures about 645 ms per
short-context decode. That does not identify the bottleneck. Carrack has no
installed `nsys` or `ncu` command in its login environment or the inspected CUDA
installation locations.

The bounded profiler wraps one actual decode with CUDA events around every kernel
launch on the existing default stream. It groups count and GPU milliseconds by
kernel name, launch grid/block dimensions and dynamic shared memory. The capture
is scoped to one CUDA context on one thread, rejects nested captures and caps the
number of records. Normal execution creates no profiling events or record storage.
Errors and unwinding clear the capture. The driver retains the function-name
CString it already allocates rather than allocating an additional name.

Each profiled launch waits for its end event. This changes host scheduling and
wall time. The sum of GPU-event intervals attributes kernel work, not allocation,
copies, host logic or the added profiler overhead. Subtracting that sum from the
profiled wall duration does not isolate ordinary host overhead. Events can also
include GPU idle gaps while the CPU enqueues work. No inference-rate
claim is derived from the instrumented run.

`qwen-model-profile` uses a prefix of 1..128 tokens and profiles one subsequent
model-selected decode token. It compares with an independent session using the
same prefix and decode input without profiling. Prefill logits, decode logits,
selected token, cursor and every persistent-state byte must match exactly. The
warmup, weight loading, session allocation and state readbacks are outside the
profile interval. Memory must return to its pre-weight baseline. This validates
instrumentation equivalence, not an independent model-quality reference.

No kernel arithmetic, PTX instruction or exported ABI changes in this stage.
The existing `Event` ownership wrapper provides all needed CUDA calls; no new
CUDA library or driver entry point is introduced.

The first Linux all-feature build exposed the validator's separate compilation
of `driver.rs`: its module root also needs the private profiler module. The
validator now includes that same source alongside its driver. The initial failure
is retained in the raw Linux test log; Ninfer remained online during the build.

## Measured result

Prefix `[248044, 271]`, one subsequent decode, three processed positions. Both
sessions produce exactly the same prefill logits, decode logits, selected token,
cursor and complete named-state hash. All weight/state allocations return to the
pre-load baseline. The unprofiled decode takes 643.165858 ms; the instrumented
decode takes 647.777296 ms. The sum over 1,476 launch intervals is 633.564000 ms.
These are one sample each after warmup, without a long-context or prefill profile.

| Kernel | Launches | Summed event milliseconds | Share of event sum |
| --- | ---: | ---: | ---: |
| `fp8_linear_wide` | 233 | 528.329664 | 83.390% |
| `bf16_linear` | 96 | 71.357248 | 11.263% |
| `nvfp4_linear` | 168 | 24.348640 | 3.843% |
| All other kernels | 979 | 9.528448 | 1.504% |

The next bounded optimization should target decode-sized FP8 projections, followed
by the small BF16 gate projections. The profile identifies which kernel costs
time. It does not establish whether memory access, accumulator arithmetic, or the
ambiguous-rounding fallback causes that cost. Measure those alternatives against
the independent reference before choosing an implementation. Keep the existing
kernel as a control, validate full-model logits and state, then repeat the same
uninstrumented model trial and sanitizer checks. Do not infer the achievable final
model speed from removing one measured cost. Attention at three positions is not
evidence about attention cost at long context.

## Provenance and restoration

Source `55ee5ae56b21a09661d8b79199c73a9f2f999539`; release xtask SHA256
`67f9f807c1a544ff7cf2fb3d5b4b2c4dd5d8f4395b0c70644178a42fdb8099fc`;
unchanged PTX SHA256
`fa04eb2e19c22bcd47fc657c9adb6d8e079349719d31f7bbb213fe85a8a70ab6`.
The pinned raw-v1 artifact, RTX 5090 UUID, driver and toolchain match the
[first model timing](model-timing-20260927.md).

Linux tests pass with 271 library and 25 validation-executable tests. macOS passes
208 library tests. Both Clippy runs and the repository no-console check pass;
the Linux release build succeeds. No PTX arithmetic changed, so the previous
independent model and sanitizer qualification remains the kernel evidence. This
trial adds instrumentation equivalence, not broader model-quality qualification.

```sh
just specialize-tools-build
target/release/xtask specialize qwen-model-profile --artifact ARTIFACT --tokens 248044,271 --ptx PTX --device 0 --output NEW_FILE
```

The bounded service wrapper restores Ninfer at 07:45:48 EDT on September 27,
PID 3197048, HTTP 200, 30,046 MiB process memory. ComfyUI remains PID 448118 at
498 MiB. No GPU power or clock settings changed. The raw trial is in
`target/specialize/qwen-profile-20260927/` on both hosts; a sanitized committed
mirror, wrapper, complete report and derived totals are in
[the evidence directory](../evidence/qwen-profile-20260927/README.md).

The later performance iteration expands the bounded prefix limit to128 tokens
to attribute context-dependent costs. The original result above still used two.
