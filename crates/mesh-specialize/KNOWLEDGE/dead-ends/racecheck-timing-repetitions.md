# Racecheck traced benchmark repetitions until host OOM

Status: failed attempt preserved; bounded check-only recovery pending.

At source `e0486677c`, the ordinary workload trial passed all 14 cases, and a
separate memcheck run passed. Racecheck then traced the same numerical launch,
ten warmups and 500 timing launches per workload. On carrack at September 26,
23:26:10 EDT, the kernel OOM killer terminated xtask PID2586531. It reported
57,125,368 KiB anonymous RSS and 224,162,252 KiB virtual memory. Racecheck's
trailing zero-hazard line is not a pass: its target failed, and the output JSON
is empty. The SSH command returned9. Synccheck was not reached.

Device/toolchain: RTX5090, driver615.71.09, CUDA tools13.4.92, host Rust1.98.1;
Rust PTX hash `227c8911e5ada033a679802934385cd5e007a28ba2fbb399eb9558b4dbb7999a`.
Raw evidence: `target/specialize/workloads-20260927/racecheck.log` and kernel
journal excerpt. The workload GPU payload was at most22,691,840bytes; the OOM
was host-side during instrumented execution, not model or tensor residency.

The shell EXIT trap restarted Ninfer, which reported engine-ready23:26:19EDT
with its original250,880-token FP8 KV allocation. ComfyUI remained the same PID.
An unrelated transient mesh-llm process used1,010MiB at the ordinary trial start;
it had exited before this inspection and was not controlled by this experiment.

Recovery: `workload-check` runs each numerical fixture once without warmup or
timing repetitions, and records `timing_collected=false`. Execute sanitizer
checks in a temporary user systemd scope with `MemoryMax=8G`, `MemorySwapMax=0`
and a timeout. Do not instrument benchmark repetitions unless that repetition
is the behavior under test and its resource budget is bounded.
