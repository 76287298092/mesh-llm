# A16 small-batch sliced-K head candidate

Status: source candidate only, 2026-09-27. Compilation, CPU tests, PTX assembly,
GPU execution, sanitizers and timing NOT RUN by this worker. Before/after latency,
register usage, spills, occupancy, memory traffic and model quality are unknown.
No dispatch or runtime integration is included. Parent owns those gates.

## Contract and schedule

`kernels/nvptx/fp8_a16_head.rs` exports `fp8_a16_head(input, weight, weight_scale,
out, unrounded, m, n, k)`. Inputs are row-major BF16 `[M,K]`, finite E4M3FN
`[N,K]`, and finite signed BF16 scales `[N]`. Outputs are BF16 and FP32 `[M,N]`.
M is 1 through 8, N is a multiple of 8 through 262144, K is a multiple of 16
through 32768. Zero extents are rejected by the independent reference and must
be rejected by the future launcher. Device pointers must be aligned for their
element types, disjoint, complete, and live through completion.

Launch grid `[N/8,1,1]`, block `[512,1,1]`. A CTA computes 16 token rows by eight
output columns, zero-padding token rows beyond M. Sixteen warps each own K16
tiles at offsets `warp*16 + iteration*256`. Idle warps still publish zero partials
and participate in the single barrier. All 32 lanes of an active warp execute
every MMA. Each warp stores 128 FP32 results into its own 512-byte shared range.
After a converged `bar.sync 0`, warp zero sums partials from warp zero through
warp fifteen in ascending order with explicit `add.rn.f32`. Shared storage is
8192 bytes per CTA. No atomics, global scratch, asynchronous copies or shared
operand staging are used in this first candidate.

For lane `l`, group `g=l/4` and pair `p=2*(l%4)`, packed A BF16 register pairs
address `(g,p)`, `(g+8,p)`, `(g,p+8)`, `(g+8,p+8)` in the local 16x16 tile.
Packed B pairs address K positions `p,p+1` and `p+8,p+9` of weight column `g`.
The four accumulator values map to `(g,p)`, `(g,p+1)`, `(g+8,p)`, `(g+8,p+1)`.
Global output columns add `8*blockIdx.x`. These mappings come from the public
PTX m16n8k16 BF16 fragment specification; asymmetric GPU fixtures remain a gate.

Every finite E4M3FN code has an exact BF16 representation. The kernel widens
normal codes through exponent/fraction bits and subnormals through a seven-value
lookup. Negative zero survives widening. The host must reject codes 0x7f/0xff.
The FP32 reduced dot is multiplied once by the represented BF16 scale using
`mul.rn.f32`, then BF16 RNE. Signed and zero scales are legal. The host must
reject nonfinite outputs, including BF16 overflow; finite inputs alone do not
exclude intermediate FP32 overflow.

The independent `reference/fp8_a16_head.rs` computes each logical dot in FP64,
scales in FP64, then rounds to FP32 and BF16. It does not reproduce warp slicing,
fragments, or reduction. Tests cover M1/5/8, signed and zero scales, cancellation,
E4M3 subnormals/extremes, BF16 ties, shape/extents, nonfinite inputs and overflow.
This is not an exact-A8 profile and does not promise CPU/GPU bit equality. MMA
floating-point accumulation ordering requires qualification on the selected GPU.

## PTX inventory handoff

Parent must add these new sites to the shared assembly inventory before admission.
All sites are in `kernels/nvptx/fp8_a16_head.rs`.

| Function | Instructions | Independent check |
| --- | --- | --- |
| `coordinates` | `mov.u32` special registers | Tile coverage against logical row/column oracle |
| `partial_base` | `.shared .align 4 .b8`, `mov.u32` symbol address | 2048 FP32 slots; bounds/race/sync sanitizer checks |
| `store_partial` | `st.shared.f32` | Disjoint warp/lane/slot map |
| `load_partial` | `ld.shared.f32` | Post-barrier initialized slots and logical oracle |
| `add_rn` | `add.rn.f32` | CPU FP64 dot comparison and signed cancellation fixtures |
| `scale_rn` | `mul.rn.f32` | Signed/zero BF16 scaling oracle |
| `mma` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | Logical FP64 matrix product; asymmetric fragment fixtures |
| kernel barrier | `bar.sync 0` | All 512 threads converge; synccheck/racecheck |

Ordinary Rust pointer reads/writes generate global loads/stores. Inspect emitted
PTX and SASS as part of compilation; the source inventory cannot establish the
compiler's final instruction selection or resource use.

## Provenance and remaining gates

Technique reference, read-only Ninfer source pin
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`:
`target/specialize/ninfer-deep-dive/src/ops/linear/fp8/fp8_a16_sliced_k_mma.cuh`,
especially the widening/MMA loop and CTA partial reduction. This implementation
was written independently in Rust with token-major A fragments, direct global
loads, strided K16 ownership and an ascending sixteen-way reduction. It does not
copy Ninfer's staged/swizzled schedule or reduction implementation.

Instruction reference:
[NVIDIA PTX ISA, warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
Local instruction precedent is `kernels/nvptx/ordinary_mma.rs`; represented-input
semantics also follow the existing A16 decode reference without changing it.

Parent integration gates: register source/oracle and assembly inventory; compile
through approved Just recipes; inspect emitted PTX; exhaustively compare all
254 finite E4M3 codes against logical decode; validate asymmetric fragments and
K16/K240/K256/K272 tails, M1/5/8, signed cancellation and scales; run all CUDA
sanitizers; then test real same-input projections at 5120x6144, 5120x17408 and
248320x5120. Freeze tolerance and quality policy before examining results.
Measure against exact split-four and existing A16 GEMV separately. Keep this
profile opt-in and requalify ordinary/MTP agreement, teacher-forced logits and
full model quality before any performance claim or dispatch promotion.

Device, driver, toolchain and clocks: not measured. Model weights: not loaded.
Reproduction command: no build/run was authorized for this worker. Source
formatting used `rustfmt --edition 2024` on the two owned Rust files only.
Durable rule: sliced-K changes floating-point association; schedule evidence
cannot certify arithmetic or model quality.

The worker's live PTX documentation fetch timed out. Parent must verify the
fragment mapping against a retrieved official specification as well as the
independent GPU fixtures; the link above is a reference target, not a successful
fetch record.

Parent retrieved NVIDIA PTX ISA9.4 section9.7.16.5.8 and checked the candidate's
BF16 A/B and accumulator coordinate formulas against its published lane mapping.
This resolves the worker's documentation-fetch gap, not GPU qualification.
Parent registered the separate source and reference for compilation; current
resident dispatch is unchanged.


Parent compilation checkpoint: 276 host tests and host Clippy pass; three
Clippy findings in the A16 reference were repaired without arithmetic changes.
`just specialize-ptx` passes with nightly2026-09-25, SM120a. Retained PTX
`target/specialize/iterate-20260927/features-head-pipeline.ptx` SHA256
`e8beed4f5bba069809a7ff3d80398bea6f3980aa77a9a9f7352c4753f6ec401a`.
Both expected entry symbols are present. PTX declares local storage,40bytes for
A16 and144bytes for NVFP4; these are virtual PTX declarations, not measured JIT
register/spill/resource usage. GPU qualification and resident integration remain
pending. Existing baseline PTX remains retained separately.

## Synthetic launch trial, source handoff

Added `src/kernels/cuda/a16_head_trial.rs` with parent-owned registration pending.
Its `run(ptx, device)` records SM120 device information, function resources and
37 case results. One M8/N256/K16 one-hot case covers all 254 finite weight codes
in every token row, with K-dependent cyclic column shifts. Twenty-four asymmetric
small-dyadic cases cross M1/5/8, N8/24 and K16/240/256/272, including idle warps and
partial final K groups. Their FP32 arithmetic is exactly representable, so both
FP32 bits and final BF16 bits must equal the independent oracle. Twelve general
cases cross M1/5/8, N8/24 and K272/5120 with every finite weight code, signed
activations and nonuniform signed scales.

Before any run, the general-case acceptance limits are fixed at BF16 relative L2
<=0.01 and FP32 maximum absolute error divided by max(1, maximum absolute oracle)
<=1e-4. Every case also requires finite FP32/BF16 outputs and BF16 bits exactly
equal to RNE of its stored FP32 output. Reports retain BF16 and FP32 difference
counts, both normalized L2 errors, maximum raw/scaled error, completion and poison
status. Numeric failure returns `all_passed:false` with evidence. Driver or
reference errors still return an error; no successful completion is claimed then.
Quiet NaN poisons detect unwritten results. All buffers remain owned through
synchronization and both downloads.

Worker validation for this addition: rustfmt only. Cargo, compilation, execution,
sanitizers and GPU access NOT RUN. No timing is taken. Parent reported earlier
kernel/source integration checks at `daabf18c3`; this new launch trial has not yet
been included in those checks. Representative resident head timing and real-weight
coverage remain separate parent gates after synthetic and sanitizer qualification.
