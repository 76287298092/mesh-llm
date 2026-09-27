# Resident FP8 MLP execution

Status: resident FP8 MLP execution and independent comparisons pass for layers
56 and 63, with one/17 tokens and all three sanitizers. Full decoder pending.

Mac tests (181) and Clippy pass. The first Linux test run passes 209 library and
20 validator tests, but Clippy found an unused import in the Linux-only wrapper.
It is removed before deployment; the failed log is retained. Ninfer remained online.

The last eight decoder layers use FP8 gate/up/down weights, unlike the first
56 NVFP4 MLPs. Pinned checkpoint `config.json` group_0 explicitly selects layers
56–63 with dynamic per-token FP8 inputs and per-channel FP8 weights. The execution
path borrows validated tensor views
from the full resident weight arena. It performs GPU input quantization, refined
FP8 matrix products, BF16 SiLU/product and output projection. It has no scalar
reference calls, host readbacks, or weight uploads. Temporary allocations and
explicit synchronization remain in this first implementation; it is not tuned.

A separate trial compares layers 56 and 63 with one and 17 rows of deterministic
synthetic normalized hidden input. All intermediate reference values come from
original inputs/weights through the independent scalar chain. GPU readbacks never
replace inputs. Gate, up, activation and down each use the unchanged aggregate and
per-token 1% normalized-L2 / 0.9999 cosine bounds, with exact BF16 rounding checked
against each matrix's unrounded output and bit differences retained as diagnostics.

This adds no new PTX. It reuses the qualified refined FP8 matrix entrypoint and
SiLU-product kernel. Source metadata, shape, byte extents and vocabulary-independent
input bounds are checked before exposing resident addresses. Full-model scheduling,
logits, usable inference context and prefill/decode comparisons remain pending.

Reproduction uses `xtask specialize qwen-fp8-mlp-check` with the same ordered
`--artifact`, `--ptx`, `--device`, `--output` arguments as other Qwen trials. Build
through Just.

## Carrack qualification, 2026-09-27

Implementation `4863bf6ef564510186a36544e596452e8d684979`; qualified source after
the Linux import fix `baf1cd881b6cbe754ef50cb861c827d5f8654344`. Release xtask
SHA256 `c43c40b594f6e09cea71075d074c9afd7e89ea121ec7aadb952eac7b4deddb4c`.
PTX is unchanged from the preceding trial, SHA256
`ee13b6bf3d34ee2ccceeaf4b9420c32ddc0fee2c97fc7f7d529f86ba06c306b9`.

GPU0 is RTX5090, SM120, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`,
driver 615.71.09 / Driver API 13040. Host Rust 1.98.1 / LLVM 22.1.8 and CUDA
13.4.92. Idle pretrial sample: P8, graphics 375 MHz, memory 405 MHz, 11.68 W,
600 W power limit. No steady-state clock or throughput measurement was collected.

All 1,620 text tensors are loaded once into the resident weight arena. The GPU
chain makes no intermediate data copies to the host. It synchronizes before
releasing temporary quantization/diagnostic allocations; this is structurally
usable execution, with allocation reuse and launch optimization still pending.
Five negative live checks reject foreign input buffers, modules or resident
weight owners before kernel execution. This explicitly covers both projection
and activation wrappers; a failed extent check cannot substitute for rejection.

| Layer / tokens | Down output BF16 differences | Aggregate normalized L2 | Worst-token normalized L2 |
| --- | ---: | ---: | ---: |
| 56 / 1 | 0 of 5,120 | 0 | 0 |
| 56 / 17 | 15 of 87,040 | 0.0000115135433 | 0.0000521334504 |
| 63 / 1 | 0 of 5,120 | 0 | 0 |
| 63 / 17 | 0 of 87,040 | 0 | 0 |

All 1,253,376 gate/up outputs match the independent reference exactly in BF16.
Each 17-token activation has three BF16 differences (six of 626,688 activation
values across all cases). These precede the 15 layer-56 down differences; layer-63
down remains exact. All 184,320 down outputs meet the unchanged aggregate and
per-token budgets; minimum token cosine is 0.9999999986411725. All matrix outputs
round exactly from their actual FP32 values. These comparisons are synthetic-input
MLP branches, not evidence for preceding attention or whole-model logits.

Normal, memcheck, racecheck and synccheck reports all pass. Harness durations are
25.001507776 / 31.565675647 / 25.923413371 / 25.166436428 seconds respectively,
including independent CPU references, artifact verification and uploads. They are
not inference throughput. Memcheck and synccheck report zero errors; racecheck
reports zero errors, warnings or hazards. Refined FP8 JIT uses 46 registers,
zero static shared memory and zero local bytes. No new assembly site was added.

CUDA free memory is 32,221,822,976 bytes before weights and after release;
10,575,020,032 with the full text weights; 10,568,728,576 at the 17-token output
checkpoint. These are checkpoints, not peak measurements. This trial allocates
no persistent attention/GDN state and establishes no usable model context.

Mac: 181 library tests, Clippy and no-console checks pass. Linux: 209 library
tests, 20 validator tests, Clippy with warnings denied and release build pass.
The first unused-import Clippy failure remains in the evidence directory.

[Raw evidence](../evidence/qwen-fp8-mlp-20260927/) contains all four JSON reports,
build/sanitizer logs, toolchain/device observations, exact script and service
restoration evidence. Raw working copies remain under
`target/specialize/qwen-fp8-mlp-20260927/` on both hosts; PTX is retained there and
identified by hash. Every invocation was bounded to 240 seconds and 8 GiB host
memory with no swap. Ninfer was paused at 05:30:10 EDT, active at 05:32:00, engine
ready at 05:32:06 and health HTTP 200. Ninfer PID 3083563 uses 30,046 MiB; ComfyUI
PID 448118 is unchanged at 498 MiB. Original Carrack branch remains at
`d8949ab608a8771115b8e9d2bc23aefad94a9cf9`.

Remaining: reference-free GDN, attention and NVFP4 operation composition using
the persistent state/weight arenas; ordered 64-layer execution, final norm and
logits; independent model-level comparison; then prefill/decode/context trials.
