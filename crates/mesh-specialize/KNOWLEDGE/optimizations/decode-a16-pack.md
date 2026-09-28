# BF16 A/B FP32 decode candidate

Status: implemented, unqualified, 2026-09-28. The coordinator narrowed
`feature_decode_a16_pack` to the small BF16 A/B projection bottleneck. No FP8,
NVFP4, norm/gate fusion, scorer, attention, or stream-executor changes are included.
Default arithmetic and resident dispatch are unchanged.

Source context: coordinator supplied HEAD `ab33f730e`; this worker did not run
Git to verify it. Model target is Qwen3.8-27B, current GDN A/B shapes each
[48,5120]. Weights are separate resident row-major BF16 matrices, without scales,
from `linear_attn.in_proj_a.weight` and `linear_attn.in_proj_b.weight`.
Input is the already BF16-rounded input RMSNorm output. No mspec layout or
recipe transformation is needed. GPU target is SM120; driver, clocks, JIT
resources and performance are not measured. Host compiler and PTX builds,
CPU tests, GPU trials and sanitizers were not run. Only rustfmt was run on
three new Rust files. There is no before/after result or bandwidth-attainment claim.

## Schedule and arithmetic

`kernels/nvptx/bf16_ab_decode_fp32.rs` exports one paired projection launch.
Each CTA computes one head of either A or B. The current shape launches 96
CTAs, rather than two launches of 12 CTAs each. This still cannot occupy all
170 SMs simultaneously. No bandwidth target follows from the schedule.

Four warps within each CTA divide K into contiguous eight-BF16 groups per
thread. Both activation and weight groups use explicit 16-byte loads. Each
thread has four independent FP32 FMA chains. At K5120 each chain receives ten
FMAs; a pairwise local sum and warp shuffles produce four warp partials. One
CTA barrier and 16 bytes of static shared memory combine those partials.
No atomics, global scratch, allocation, or inter-CTA synchronization is used.
This is CTA-local sliced K, justified by N48 rather than large-N GEMV tuning.

The exact control uses FP64 and remains unchanged. The earlier FP8 A16 GEMV
mostly saved activation-quantization launches, not dot time. This candidate
instead removes FP64 from A/B and increases CTA count, with a separate numerical
contract. Its performance and model impact require measurement.

## Exact ABI and parent integration

```text
bf16_ab_decode_fp32(
  input: *const u16, weights_a: *const u16, weights_b: *const u16,
  output_a: *mut u16, output_b: *mut u16,
  raw_a: *mut f32, raw_b: *mut f32,
  n: u32, k: u32)
```

M is fixed at one. Grid `[n,2,1]`, block `[128,1,1]`, dynamic shared 0,
static shared 16 bytes. N in 1..=256, K divisible by eight in 8..=32768.
The three input bases need 16-byte alignment. Input spans K BF16 values;
each matrix spans N*K BF16 values. Each output spans N values of its declared
type. All pointers are disjoint and remain live through completion. Finite
inputs, finite intermediate FP32 sums, finite FP32/BF16 results, and checked
shape/address arithmetic are host obligations. BF16 output is RNE of stored raw
FP32 output. There is no residual, row scale, or activation quantization.

Parent can pre-resolve this function and enqueue one pointer-only operation
against preallocated output views. Scalars N/K are fixed launch metadata.
Reuse the normalized input already consumed by QKV/Z. Bind weights from
`resident_bf16::Projection::new` semantics, replacing only the two A/B calls
in `resident_gdn::Layer::execute` for an explicitly non-exact M1 experiment.
The existing `gdn_gates` consumes the candidate's BF16 A/B outputs unchanged.
No stream integration is supplied. Mixed M1/non-M1 scheduling is not
partition-equivalent; MTP and quality admission remain parent-owned.

## Independent oracle and fixed trial gates

`reference/bf16_ab_decode_fp32.rs` accumulates decoded products sequentially
in FP64, casts once to FP32, and rounds once to BF16. It does not reproduce
the device reduction. Host fixtures have hand-computed signed dots, absolute
product sums, cancellation, BF16 tie-to-even boundaries, invalid extents,
nonfinite inputs, and overflow rejection. These tests are authored, not run.

The GPU trial has eight cases: N1/K8, N3/K24, N48/K1024, N48/K1032,
N48/K5120 exact dyadics, two distinct seeded N48/K5120 signed cases, and
N48/K32768. A and B have different values. Small-dyadic cases require exact
raw and BF16 bits. The unchanged FP64 baseline always requires exact oracle
bits on these bounded fixtures. General candidate cases use per-output
`gamma(s)*sum_abs_products + 1e-37`, where `s=2*ceil(K/1024)+14`,
`gamma(s)=s*2^-24/(1-s*2^-24)`. This covers chain depth, both reductions,
and oracle rounding. BF16 absolute error may additionally include the sum
of the two endpoint half-ULPs. A universal one-BF16-ULP bound near cancellation
is not justified. Every output must be finite, BF16 must equal RNE of stored
raw, and repeated launches must reproduce all output bits. This is only an
operator gate; `findings/quality-gates.md` is still required before promotion.

Harness `src/kernels/cuda/bf16_ab_decode_trial.rs` poisons outputs and reports
candidate/control metrics, determinism, device identity, JIT resources and logs.
It runs three warmups, then three event batches of ten A/B pairs, for both the
single candidate launch and the two unchanged baseline launches. Reports include
logical weight GB/s using `4*N*K` bytes per pair. The 983,040-byte current weight
pair fits cache; repeated timing is warm-cache, not measured DRAM bandwidth.
Event batches include host submission gaps, and candidate-first order is fixed.
No real model weights or model throughput are measured by this harness.

Parent reproduction after approved host/PTX builds:

```text
cargo xtask specialize bf16-ab-decode-check --ptx PATH --device 0 --output NEW_FILE
```

This mirrors `a16-head-check` through the existing xtask `run_probe` helper,
which uses `create_new(true)`, adds the PTX SHA256, retains failure JSON, and
refuses overwrite. Parent must run host tests/type-check, PTX/JIT inspection,
memcheck/racecheck/synccheck, same-input real A/B comparison, then matched timings.
All generated evidence belongs in a fresh parent-selected directory. No trial
report exists yet.

## Assembly inventory handoff

All new inline-asm sites are in `kernels/nvptx/bf16_ab_decode_fp32.rs`.
Parent should add these to the shared inventory before admission.

| Function/site | Instructions | Independent evidence needed |
| --- | --- | --- |
| `coordinates` | `mov.u32` tid.x/ctaid.x/ctaid.y | Separate A/B row coverage, N1/N3/N48 |
| `load8` | `ld.global.v4.u32` | Logical BF16 oracle, K8/K24/K1032, alignment and memcheck |
| `fma` | `fma.rn.f32` | FP64 dot and absolute-product error bound |
| `add` | `add.rn.f32` | Same oracle, signed cancellation |
| `sum_warp` | `shfl.sync.bfly.b32` full mask | Both reductions, all-lane participation |
| `shared_base` | `.shared .align 4 .b8`, symbol-address `mov.u32` | 16-byte extent and JIT resource report |
| `store_partial` | `st.shared.f32` | Distinct warp slots, racecheck |
| `load_partial` | `ld.shared.f32` | Initialized slots after barrier, racecheck |
| kernel barrier | `bar.sync 0` | CTA-uniform guards, synccheck |

Compiler-selected global stores, BF16 bit operations and final register/spill
usage also need emitted PTX/SASS inspection. No instruction qualification is
claimed from source inspection alone.

Durable rule: removing FP64 changes arithmetic; operator error bounds do not
certify gate sensitivity, model quality, MTP equivalence, or serving performance.

## Paired FP64 continuation

Status: implemented for parent compilation, unqualified, 2026-09-28. Parent
supplied source context `30658a926`; the worker did not run Git to verify it.
New device source is `kernels/nvptx/bf16_ab_decode_fp64.rs`. The existing FP32
candidate, `bf16_linear_decode` control, and resident/default dispatch remain
unchanged. No graph, stream-forward, model-score, or model-source changes.
GPU architecture target is SM120. Driver, clocks, JIT resources, GPU numerical
results, sanitizers, and model performance are not measured by this worker.
No before/after speedup is established. Parent retains the build/GPU slots.
Direct `rustfmt --edition 2024 --config skip_children=true` and its `--check`
pass on the new kernel, extended trial, and module registry. Their lengths are
239, 541, and 131 lines respectively. No Cargo, PTX compilation, tests, Git,
SSH, or GPU execution was performed by this worker.

### Schedule and ABI

```text
bf16_ab_decode_fp64(
  input: *const u16, weights_a: *const u16, weights_b: *const u16,
  output_a: *mut u16, output_b: *mut u16,
  raw_a: *mut f32, raw_b: *mut f32,
  n: u32, k: u32)
```

Exactly the same nine arguments and M1 shape contract as the paired FP32
candidate: grid `[n,2,1]`, block `[128,1,1]`, dynamic shared zero, N1..256,
K8..32768 divisible by eight. The three BF16 input bases are 16-byte aligned,
with extents K, N*K, N*K. Output extents are N BF16, N BF16, N FP32, N FP32,
all naturally aligned. All seven buffers are disjoint and live through completion.
The host checks dimensions, extents, address arithmetic, finite input values,
and finite FP32/BF16 output values.

One CTA owns one head/projection, giving 96 CTAs at N48 instead of two launches
of 12 CTAs. This changes parallelism, not the GPU's FP64 throughput. Each thread
loads eight adjacent BF16 activations and weights with `ld.global.v4.u32`, widens
them exactly, and updates four independent FP64 FMA chains. K5120 gives ten
FMAs per chain. Two FP64 local adds per contribution's path combine the chains;
a paired-word shuffle reduces each warp. Four FP64 warp totals use 32 bytes of
static shared memory, one CTA barrier, and a final FP64 warp reduction. Lane zero
casts once to FP32 with `cvt.rn.f32.f64` and stores that raw result plus the same
RNE BF16 encoding as the control. No FP32 accumulation, atomics, or global scratch.
JIT register/shared/local statistics must confirm the source-level expectations.

### Precision bounds and cancellation limits

Every finite BF16 product is exact in FP64: at most 16 significand bits, with
product magnitudes inside FP64's normal range. FMA therefore does not introduce
a product-rounding difference from the control's separate FP64 multiply/add.
Summation order still changes. FP64 has only 53 significand bits; large exponent
spreads can lose small terms, and cancellation can expose those losses. Neither
FP32 nor BF16 bit identity is guaranteed for arbitrary finite BF16 inputs.

A sufficient exact-sum condition is that every product is an integer multiple
of some `q=2^e` and `sum(abs(products))/q < 2^53`. Then every partial sum in any
of these schedules is exactly representable in FP64. This condition holds for
the deliberately bounded fixtures, but is not assumed for actual model inputs.

Outside that sufficient condition, a conservative pre-cast absolute error bound
for the candidate is `gamma_d * sum(abs(products))`, with
`d=2*ceil(K/1024)+12`, `gamma_d=d*2^-53/(1-d*2^-53)`. This counts chain FMAs,
two local adds, and two five-stage warp reductions, including zero partners.
The sequential oracle has its own `gamma_K` bound; their pre-cast difference is
bounded by the sum of these envelopes. Comparing stored FP32 values also needs
the two cast-rounding errors, including gradual underflow, and BF16 comparison
needs both BF16 rounding errors. No relative-error or one-BF16-ULP guarantee
survives arbitrary near-zero cancellation or an output at a rounding boundary.
These analytical bounds are explanatory only: the new FP64 fixture gate uses
zero tolerance, not a widened envelope.

### Extended bounded trial and gates

The existing `bf16-ab-decode-check` command now emits schema version 2 with
separate `fp32` and `fp64` results in each case and separate top-level
`fp32_all_passed` / `fp64_all_passed` flags. Both candidates run against the
unchanged two-launch FP64 control and the unchanged independent sequential
oracle in `reference/bf16_ab_decode_fp32.rs`. The eight original fixtures and
their seeds remain, with two additional N48/K5120 cases:

- Large cancellation: repeating `[2^20, r, -2^20, r]`, where
  `r=(head+1)*2^-16`, activation one, with opposite-signed A/B matrices.
  Each row's exact residue is `2560*r` for A and its negative for B.
- Bounded magnitude range: signed BF16 values with all eight significand bits
  and exponents -5..5. Products lie on a `2^-24` lattice, and their absolute
  sum in lattice units stays below `2^53`.

FP64 and the control require raw-FP32 and BF16 bit identity against the oracle
for every case, including the original seeded fixtures. FP64 additionally
requires direct bit identity with control outputs. FP32 keeps its original
exact-dyadic and analytical-error gates. Differences are counted separately for
raw and BF16 values; up to eight failed output bit patterns per projection are
retained in JSON. Numerical failures return report data with `all_passed=false`;
they are not discarded or used to loosen the gate.

All four outputs of both candidates and their controls start as NaN poison with
16-byte prefix/suffix guards. The harness checks first and repeated-run guards,
re-poisons before repeats, checks candidate and control determinism, and verifies
finite output plus BF16 RNE consistency. Guard bytes cannot detect all reads or
out-of-allocation writes; memcheck/racecheck/synccheck remain required.
Host tests cover guard detection, strict raw-bit rejection despite equal BF16,
NaN/extent rejection, hand-computed cancellation residues, and the magnitude
fixture's exact-lattice bound. Tests are authored, not run by this worker.

Reports include JIT log and registers/static-shared/local bytes for all three
symbols. Timing stays at three warmups and three CUDA-event batches of ten A/B
pairs per kernel/control. Fixed order is FP32, control, FP64, control for each
fixture. Only event times and logical byte counts are reported in schema 2;
the former logical GB/s estimate is removed. These weights are cache-resident
in the repeated trial, and event batches include host submission gaps. This
is neither a DRAM-bandwidth measurement nor a whole-model throughput claim.

Parent reproduction uses the existing command after building updated PTX:

```text
cargo xtask specialize bf16-ab-decode-check --ptx PATH --device 0 --output NEW_FILE
```

The existing command wrapper retains JSON/PTX identity evidence without
overwriting prior reports. Parent must compile/type-check and run the host
tests, inspect emitted PTX/JIT for the intended vector loads/four FP64 chains/
resources, run the bounded trial and all three sanitizers, then separately
qualify same-input actual-model A/B, full-model state/logits, and timings.
No resident integration or model qualification is supplied here.

### FP64 assembly sites and reference gates

All eleven inline-assembly sites are registered in `KNOWLEDGE/asm-inventory.md`.

| Site | Instructions | Required evidence |
| --- | --- | --- |
| `coordinates` | special-register `mov.u32` | N1/N3/N48 coverage of both projections |
| `load8` | `ld.global.v4.u32` | Oracle values, K8/K24/K1032 tails, alignment, memcheck |
| `widen` | `cvt.f64.f32` | Independent decoded BF16 products, emitted no-FTZ conversion |
| `fma` | `fma.rn.f64` | Exact bounded fixtures, inspect four live chains |
| `add` | `add.rn.f64` | Cancellation and bounded-magnitude oracle comparisons |
| `shuffle` | full-mask `shfl.sync.bfly.b32`, low/high halves | Both FP64 reductions, all-lane participation |
| `shared_base` | `.shared .align 8 .b8 [32]`, address `mov.u32` | JIT static-shared size, slot bounds |
| `store_partial` | `st.shared.f64` | One writer per warp slot, racecheck |
| `load_partial` | `ld.shared.f64` | Four initialized slots after barrier, racecheck |
| `narrow` | `cvt.rn.f32.f64` | Exact raw bits plus BF16 RNE of those bits |
| kernel barrier | `bar.sync 0` | Uniform guards, synccheck |

Durable rule: retaining FP64 permits a different parallel schedule, not a
universal bit-identity claim. Keep strict bounded failures and actual-model
qualification separate from microkernel timing.
