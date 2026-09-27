# GDN recurrent matrix update

Status: real-weight recurrence, exact output/state checks and all three CUDA
sanitizers pass. This is not a complete attention layer or model.

The pinned [Transformers recurrent fallback](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L441)
decays the FP32 state matrix, predicts the current value from the key, applies a
beta-weighted correction and projects the updated state with the query. Layer
zero has 48 value heads and a 128 by 128 state per head. Each value head maps to
one of 16 unrepeated key heads by integer division by three.

The Rust kernel consumes resident normalized Q/K, beta and decay from the
qualified preparation kernels, and reads BF16 V from the retained convolution
buffer. No host reference output replaces those device buffers. State is FP32
and time-major outputs are BF16 with a separate FP32 diagnostic buffer. One block
owns each value head, and one thread owns a complete value column of its state.
It processes tokens sequentially, with explicit rounded FP32 multiply/add and
no FMA. The prediction and query reductions run in increasing key-coordinate
order. Disjoint column ownership needs no inter-thread state synchronization.
This first implementation prioritizes a checkable arithmetic contract; no speed
claim or efficient parallel prefill is implied.

The independent scalar oracle indexes logical row/head/key/value coordinates.
Its ordered FP32 mode supplies the exact arithmetic acceptance contract. Every
FP32 output and state element must match bit for bit; BF16 outputs must match
exactly. Hand-calculated scalar fixtures guard the equation and head mapping.
A second mode accumulates the two dot products in f64, then rounds to FP32.
The report preserves output/state error and BF16 differences from this wider
reduction. Those diagnostics are not an acceptance tolerance or evidence of
end-to-end model quality. A full-layer/logit comparison remains required.

Whole-sequence execution is compared with `[1, remaining]`, `[2, 1, remaining]`
and single-token partitions. The host reads state at each boundary for checking,
but never uploads a replacement between chunks. All output and final state bits
must agree between partitions. Dedicated signed fixtures use asymmetric nonzero
state, two key heads, six value heads, zero/one beta and decay, and a separate
zero-state reset. Host validation rejects invalid dimensions/extents, nonfinite
values, gate-domain violations and arithmetic overflow before launch.

Two bounded Luna-max workers own the device kernel and scalar reference. The
parent specifies arithmetic, ownership and ABI, then integrates retained inputs,
chunk lifecycle, diagnostics and deployment. Reproduction extends the existing
`qwen-projection-check` command to schema 5, with fresh evidence under
`target/specialize/qwen-gdn-recurrent-20260927/`. Prior preparation, convolution
and projection evidence remains intact. Model performance and context fields
stay null and `model_executable` stays false.

Local validation passes 107 library tests, focused all-target/all-feature Clippy
with warnings denied, and Rust PTX compilation. The initial test run exposed two
fixture errors: a query selects a state row across every value column, and a
rejection fixture accidentally used valid dimensions. Both expectations were
corrected without changing the recurrence. The initial failed log is retained.
The f64 diagnostic also rejects any product that would overflow FP32, keeping
its input domain aligned with the ordered contract. GPU fixtures additionally
cover widths one and 256, alongside the model's width 128.

A separate read-only host integration review found no ABI, chunk-offset, extent,
state ownership, retained-buffer or head-mapping defect. Linux passes 115 library
tests, 17 validator tests and focused all-target/all-feature Clippy with warnings
denied. The local repository no-console check also passes. No GitHub Actions
result is claimed.

## Carrack qualification

Source `19f1c8ccc75fff7d0e66282f2cf672b27099f8bd` passes on RTX 5090 UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120, driver 615.71.09 and
driver API 13040. Release xtask SHA-256 is
`7d522c3dd7d355772b10557a6ca8237111963b84b47f6e292f2c0db7825158b0`;
PTX SHA-256 is `496f2a3fdaeb2368a0e4d2bd49f576adf3d3f456991af0ed0cc45a9f51cd1b4d`.
The host used Rust 1.98.1 and LLVM 22.1.8. Device PTX used pinned
nightly-2026-09-25 with rebuilt NVPTX core on the Mac. CUDA 13.4.92 offline
ptxas reports 37 registers, no stack/spills and zero barriers. Driver JIT reports
35 registers and zero local/shared memory. These are distinct compilation paths;
the driver-JIT function is the one executed in this trial.

The [normal report](../evidence/qwen-gdn-recurrent-20260927/normal.json) passes
110,592 BF16 attention values and their FP32 precursors across one and 17 tokens.
Each matches the independent ordered-FP32 scalar oracle exactly. Both cases
check all 786,432 FP32 state values at every chunk boundary. Whole-sequence,
`[1,16]`, `[2,1,14]` and 17 single-token executions have exactly equal outputs and
final state. The six dedicated nonzero/reset fixtures also pass, covering widths
one, eight and 256 with zero/one gate endpoints and asymmetric head mapping.
Projection, convolution and preparation regressions remain green in the same run.

The wider-reduction diagnostic differs in six real BF16 outputs in total. Its
maximum FP32 output difference is `4.76837158203125e-7`; maximum final-state
difference is `3.337860107421875e-6`. These are measured diagnostics for the
selected sequences. They do not establish acceptable end-to-end logit drift or
long-context stability. No such acceptance claim is made.

[Memcheck](../evidence/qwen-gdn-recurrent-20260927/memcheck.log),
[racecheck](../evidence/qwen-gdn-recurrent-20260927/racecheck.log) and
[synccheck](../evidence/qwen-gdn-recurrent-20260927/synccheck.log) each report zero
errors or hazards. Every numerical and partition check passes in all three runs.
Each command has an 8 GiB host-memory limit, zero swap and a 240-second timeout.
The normal command takes 12.576 seconds including artifact verification, uploads
and CPU references. No kernel or model throughput is inferred from that duration.
Driver free memory before and after temporary allocations is 32,221,822,976 bytes.
This is not a model allocator peak. Model prefill/decode/context remain null.

After `just specialize-ptx` and `just specialize-tools-build`, the qualified
normal invocation is:

```sh
./target/release/xtask specialize qwen-projection-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/qwen-gdn-recurrent-20260927/probes.ptx \
  --device 0 --output NEW_REPORT.json
```

The parent runs it inside the bounded systemd scope, with Ninfer stopped and a
restart trap installed. Sanitizer runs prefix that invocation with
`/opt/cuda/bin/compute-sanitizer --tool TOOL --error-exitcode 42` and use a new
report path each time.

Ninfer restarted at 02:34:41 EDT on September 27 as PID 2873395, reached
engine-ready at 02:34:46 and returned HTTP 200 from `/health`. Its sampled
allocation is 30,046 MiB. ComfyUI PID 448118 remains resident at 498 MiB.
The [evidence directory](../evidence/qwen-gdn-recurrent-20260927/) contains raw
reports, service records and test summaries. Build logs and exact PTX remain
under `target/specialize/qwen-gdn-recurrent-20260927/` on both hosts.

The subsequent [GDN output trial](gdn-output.md) connects gated normalization
and output projection with component checks. Independent full-layer/logit parity,
full-attention layers, MLPs, full-model schedule,
tokenizer/sampling, ABI integration and the requested model performance/context
comparison remain open.
