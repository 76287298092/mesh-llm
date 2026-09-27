# Full-decode kernel attribution

Status: instrumentation is being validated; no profiling result yet.
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
profiled wall duration does not isolate ordinary host overhead. No inference-rate
claim is derived from the instrumented run.

`qwen-model-profile` uses a prefix of 1..17 tokens and profiles one subsequent
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
