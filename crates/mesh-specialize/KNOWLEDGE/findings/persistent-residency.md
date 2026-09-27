# Persistent text weights and state allocation

Status: all-text-weight residency, state allocation, entry operation and three
CUDA sanitizer checks pass on Carrack. Full model execution and all inference
performance measurements remain pending.

The previously qualified GDN and full-attention trials reload subsets of weights
and maintain operation-local allocations. The next gate stores all 1,620 text
tensors (21,646,480,768 raw bytes) in one 256-byte-aligned device allocation. The
verified artifact reader streams and rehashes each object; the GPU trial reads
every tensor back in bounded chunks and compares its original artifact SHA256.
The separate 15 MTP tensors are deferred until base-model execution works.

The generic engine owns checked named allocation placement. The Qwen package
owns its 64-layer schedule: 48 GDN / 16 full-attention blocks, 56 NVFP4 / 8 FP8
MLPs. Each GDN block has three BF16 convolution-history rows and a FP32 recurrent
matrix. Full-attention blocks have separate BF16 K/V arrays. State storage is
153,944,064 + context_capacity * 65,536 bytes. At capacity 131,072 this is
8,743,878,656 bytes, initialized and fully read back as zero.

Admission checks aligned weights plus state bytes and a 1 GiB workspace reserve
against current free memory after module loading. The reserve is an admission
margin, not an allocated workspace or an observed inference peak. CUDA free
memory snapshots are allocation checkpoints, not a continuous peak measurement.

The first embedding/input-normalization operation uses pointers inside the full
resident weight arena, including vocabulary rows zero and 248,319. Its independent
CPU reference retains only those selected source rows while hashing the complete
embedding object. Neither CPU expected outputs nor device readbacks feed GPU
computation. Persistent state capacity is not tested usable inference context.

Reproduction (after building through the existing Just recipes):

```sh
target/release/xtask specialize qwen-residency-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/probes.ptx --device 0 --output NEW_FILE
```

No device assembly changes were required. The trial reuses the previously
qualified PTX, with SHA256
`ee13b6bf3d34ee2ccceeaf4b9420c32ddc0fee2c97fc7f7d529f86ba06c306b9`.

## Qualified run: 2026-09-27

Implementation commit `54abbb4738144193013c5e4dd17b60b6edb7736c`, with additional
normalization-view boundary assertions in trial source
`4128e1cabf3bc53da04aeeeb09ae48f3b8d07c69`. Carrack release xtask SHA256:
`4840c7478aac4f582993d492f6f7c4a7cbeae3e383405e800fac79b10b460f2d`.
Artifact identity remains the pinned raw-v1 identity recorded by each JSON report.

GPU0 was RTX5090, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120,
driver 615.71.09 / Driver API 13040. Host Rust was 1.98.1, LLVM 22.1.8 and
CUDA assembler 13.4.92. The pretrial idle sample was P8, graphics 195 MHz,
memory 405 MHz, power 12.66 W, limit 600 W. Clocks during work were not sampled;
this is a transfer/correctness test, not a performance benchmark.

| Gate | Observed result |
| --- | --- |
| Text tensor readback | All 1,620 original object SHA256 hashes match |
| Weight arena | 21,646,588,928 bytes including alignment |
| State arena | 8,743,878,656 bytes across 128 regions, zero nonzero bytes |
| Combined arena payload | 30,390,467,584 bytes / 28.3033 GiB |
| Entry through resident views | 10,240 outputs, zero FP32 error and zero BF16/residual differences |
| Released arenas | Free memory returns to the preallocation value |
| Normal harness elapsed | 36.836796140 seconds |
| Memcheck harness elapsed | 36.378958031 seconds; zero errors |
| Racecheck harness elapsed | 35.721459762 seconds; zero errors/warnings/hazards |
| Synccheck harness elapsed | 35.768909325 seconds; zero errors |

Elapsed time includes artifact verification, transfers, every weight hash, state
initialization/readback and the tiny entry operation. It is not prefill or decode.
The aligned weight arena adds 108,160 bytes to the original text tensor payload.

CUDA reported 32,221,822,976 free bytes before allocation and after release.
With weights it reported 10,575,020,032 free bytes; with weights and state it
reported 1,829,896,192 free bytes (1.7042 GiB). With the entry's temporary buffers
it reported 1,827,799,040 free bytes. The observed arena allocation delta was
30,391,926,784 bytes; allocator granularity exceeds the logical arena payload.
These are checkpoints, not a measured peak. The 1 GiB workspace margin was not
allocated. MTP, complete activation workspaces, scheduling and logits are pending.

Mac validation: 178 library tests, Clippy with warnings denied, no-console check,
formatting/diff checks and release xtask build pass. The release build emitted the
existing local compiler-builtins deployment-target linker warning (26.0 object
versus default 11.0); the tests and Clippy explicitly used target 26.0.
Linux validation: 201 library tests, 20 validator tests, Clippy with warnings
denied, release xtask build and offline PTX assembly pass. Source changes require
no new assembly inventory entries. Review caught a duplicate-row streaming
reference bug before tests/deployment; the collector now copies every overlapping
duplicate before advancing and has a split-chunk regression fixture.

Raw reports, sanitizer/build logs, exact trial script, hashes and service evidence
are in [qwen-residency-20260927](../evidence/qwen-residency-20260927/), mirrored
from `target/specialize/qwen-residency-20260927/` on both hosts. PTX remains in the
raw output directories and is identified by hash rather than committed again.
Ninfer was paused from 05:14:38 EDT and active again at 05:17:05, with engine ready
at 05:17:11 and health HTTP 200. New Ninfer PID 3067554 uses 30,046 MiB. ComfyUI
PID 448118 remains unchanged at 498 MiB. Carrack's original branch remains at
`d8949ab608a8771115b8e9d2bc23aefad94a9cf9`.

This closes only the persistent allocation/transfer gate. The next gate must use
these resident views to execute the ordered decoder schedule, including the final
eight FP8 MLPs, final norm and logits, with an independent model-level reference.
Only then can model prefill, decode and usable context be compared with Ninfer.
