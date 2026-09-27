# F07 reusable CUDA workspace primitive

Status: bounded planner and owner API added; parent integration and qualification pending.

This slice adds a capacity and FP8-geometry wrapper around the existing checked `engine::layout::Layout`, plus a CUDA owner wrapper for named scratch regions. This keeps alignment, name ordering, disjoint placement, and offset-overflow rules in one planner. The first layout accepts caller-supplied `rows`, `input_width`, and `output_width`, then plans FP8 codes (`rows * input_width` bytes), row scales (`rows * 4`), BF16 output (`rows * output_width * 2`), and FP32 diagnostics (`rows * output_width * 4`). Every region starts at the existing checked 256-byte boundary. Canonical name ordering makes offsets stable for the same region set; the workspace wrapper additionally bounds the aligned high-water allocation by caller capacity.

The CUDA owner allocates one `driver::Buffer` sized to the layout high-water mark. A step borrows the owner mutably; its non-owning region views carry that step lifetime and expose only checked pointer/byte pairs. The caller must call `complete()` after submitting the operation sequence. `complete()` synchronizes the owning CUDA context before releasing the mutable borrow. Dropping a step without completion attempts to drain the context and poisons the owner, so a possibly partial or failed step cannot reuse the arena. The buffer remains owned exactly once by the workspace and remains tied to its `Context`.

This is an integration primitive only. Existing projection wrappers still allocate their own buffers, so no repeated allocations have yet been avoided and no latency or throughput change has been measured. The live-output invariant for integration is: every kernel using a region must finish before the step is completed; no region pointer may be retained or launched after that lease ends. A future asynchronous graph path must preserve the same fixed allocation address for capture and replay and must keep the workspace lease alive until graph work completes.

## Qualification record

- Source revision: not assigned; parent integration is pending.
- Model and geometry: model independent; CPU tests use caller-provided small dimensions.
- GPU, driver, clocks, and CUDA sanitizer results: not applicable to the planner; GPU ownership wrapper has not been run.
- Rust tests: CPU-only cases are present for arithmetic overflow, rejected zero sizes and capacities, 256-byte-aligned offsets, disjoint regions, high-water accounting, stable offsets, and FP8 region extents. Parent owns the serial Cargo test slot; execution is pending.
- Allocation count: not measured. The current `resident_fp8::Projection::run` creates four temporary `Buffer`s per invocation; a later integration measurement must count allocation/free calls before and after using this owner on the same projection workload.
- Performance: not measured. No speed claim is made.
- Reproduction: after parent module registration, run the focused `mesh-specialize` workspace tests for `engine::workspace`; then qualify the integrated CUDA caller on the selected device with output lifetime checks and an allocation-call count. Compare repeated identical FP8 projection geometry and report median wall time plus range separately from device event time.

Durable rule: a workspace reset or next step is legal only after successful context synchronization. Region addresses are borrowed views, never separately owned `Buffer`s. If completion fails or the lease is dropped without explicit completion, reject reuse.

Parent integration: registered the host modules; 223 macOS library tests and
Clippy pass. F01 PTX compiles through `just specialize-ptx`. An initial oracle
fixture had N/K extents inconsistent with its two activation values; corrected
to a two-by-two diagonal case without relaxing its expected results. Linux CUDA
wrapper checks and GPU/model qualification remain pending. No resident default
was changed. Logs, including initial failures, are under
`KNOWLEDGE/evidence/features-20260927/`.

Parent GPU check, 2026-09-27: eleven synthetic F01/F02 cases passed on Carrack RTX5090, including M/N/K tails and K=5120. BF16 outputs matched the independent fixtures exactly; native FP8 raw FP32 scaled error was at most 9.58e-7. Workspace stable-address reuse and aborted-lease poisoning passed. All three CUDA sanitizer tools reported zero errors/hazards. Evidence: `../evidence/iterate-20260927/features-projection/`. PTX SHA256 `35c12bcfe57d282985b02bae256be0770991816519bc1a6cbf825a9cf369aa75`. JIT decode uses 33 registers and no local memory; prefill uses 56 registers, 64 local bytes and 6144 shared bytes. These are synthetic correctness checks, not model qualification or speed measurements. Ninfer and other GPU processes remained running.

## Parent real-weight MLP experiment

Prepared `resident_mlp_workspace.rs` and `mlp_workspace_projection.rs` queue an
entire exact FP8/NVFP4 MLP using one persistent named-region allocation. The
`mlp-workspace-check` command selects real layers0/56 weights and deterministic
signed BF16 inputs at rows1/5/128/512. It compares all seven exposed BF16/raw
projection and activation buffers against existing execution and repeats reuse.
Two schedules isolate persistent storage with old operator waits from one final
completion wait. Existing PTX/arithmetic/diagnostic writes and separate gate/up
quantizers remain unchanged; NVFP4 input scales stay projection-specific.
The experiment explicitly rejects non-exact profiles and split-K, preventing
accidental arithmetic/dispatch confounding. It is not model-integrated.

Initial timings are screening only: fixed schedule order, three repeats, no
allocator/driver tracing yet. Dynamic host argument vectors and function lookup
remain, and input values are deterministic fixtures, not recorded activations.
Linux compilation, GPU correctness/reuse, sanitizer and timing evidence remain
pending. No performance claim is made from this prepared implementation.

First normal GPU trial `workspace-check-1` at `670963c16` passed every case
and all seven exposed buffers exactly through repeated reuse. Representative
screening medians, existing/persistent-with-waits/one-wait milliseconds:
NVFP4 rows1 0.1947/0.1690/0.1554; rows128 1.8702/1.3141/1.2984;
FP8 rows1 0.3033/0.2827/0.2720; rows512 22.2203/21.0402/21.1282.
Fixed order and only three repeats prevent a stable speed claim. The largest
workspace was232.51MiB. Inputs are deterministic BF16 fixtures with real weights;
this is not measured whole-model throughput. PTX is unchanged `features-splitk`
SHA256 e18df0fb02524af65d00e3b313183da18ce549fce93a65a6a22011ccfc3c6136.
Ninfer remained inactive and ComfyUI remained resident. Mac259tests, Clippy on
both hosts and release tools build passed. Explicit address-stability checks and
post-gate abort/drain/poison checks were then added for the next sanitizer trial.
