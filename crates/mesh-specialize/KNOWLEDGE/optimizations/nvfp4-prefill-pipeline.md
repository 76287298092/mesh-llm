# NVFP4 prefill pipeline candidate

Status: source candidate, September 27, 2026. No compilation, host test, GPU
launch, sanitizer or timing run was performed by this worker. Parent integration
and qualification are required. Before/after performance, register usage, driver,
toolchain, clocks and device observations are not measured. Intended target is
SM120a. This entry does not close deep-dive experiment 4.

The independent Rust implementation uses the existing logical packed matrices,
the existing native K64 MMA helper, and the existing FP32 multiply/BF16 epilogue.
It adds one separate `nvfp4_prefill_tiled` symbol. It does not change the baseline
or add runtime dispatch. The scalar reference delegates decoded products to
`nvfp4_linear_reference`, which does not use GPU fragment indexing. Its f64 dot
is an independent numerical oracle, not a promise of native-MMA bit identity.
The parent's pre-existing native-multirow versus integer-one-row mismatch at
layer 22 down projection, row 101 after online attention remains unresolved by
this work. No universal integer-exact claim applies to this candidate.

## Layout and lifetime proof

A CTA has 256 threads and computes 32x32 outputs. Warp `w` computes rows
`16*(w/4)..+16` and columns `8*(w%4)..+8`. Each warp retains the original 16x8
fragment mapping and accumulates exactly one K64 MMA per iteration in ascending
K order. Output guards cover M and N tails without removing threads from MMA
or barriers. Admission is M 1..512, N 8..32768 divisible by 8, K 64..32768
divisible by 64. Launch grid is `[ceil(N/32),ceil(M/32),1]`, block `[256,1,1]`.

The 4,608-byte shared allocation contains two 2,304-byte stages. Each stage has
A codes at 0..1024, W codes at 1024..2048, A scales at 2048..2176, and W scales
at 2176..2304. A code row contains 32 bytes for K64; a scale row contains four
bytes. A rows are reused by four warps and W rows by two warps. Producer thread
t owns words t, t+256, and t+512 while in range 0..576. These sets are disjoint
and cover the full stage. All source and destination addresses are four-byte
aligned because base pointers are aligned and K is divisible by 64.

Missing M/N rows use `cp.async` source size zero, with the live allocation base
as source. Thus no out-of-bounds Rust pointer is formed. Both codes and scales
are zero-filled for missing rows. Their MMA results are discarded. Valid row
scale words are copied unchanged; there is no scale permutation or K tail.

The prologue issues and commits stage zero. Every iteration waits for all of
its thread's copies and then executes a CTA barrier, publishing all producers'
words to every consumer. It then issues and commits the next stage before
loading current fragments and executing MMA. The terminal CTA barrier retires
all readers before that slot can be overwritten two iterations later. There is
at most one pending group per thread. Final iteration has no next copy and ends
with no pending group. All barriers and loop counts are CTA-uniform. The async
copies can overlap current-stage computation; useful overlap is not measured.

## PTX inventory for parent integration

Instruction references use the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
Every new assembly block below uses a compiler memory clobber through absence of
`nomem` and `readonly`; synchronization cannot be optimized past shared accesses.

| Site | Instructions and guarantee | Safety contract |
| --- | --- | --- |
| `coordinates` | `.shared .align 16 .b8`, `mov.u32`, `%tid.x`, `%ctaid.x/y`; per-CTA allocation and thread/block identifiers. See state spaces and special registers. | One inlined allocation site; exactly 256 x threads and no y/z thread dimension. |
| `copy_word` | `cvta.to.global.u64`; `cp.async.ca.shared.global ...,4,src-size`. [Async copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async) permits four-byte copy and zero-fills bytes beyond source size. | Aligned disjoint shared words, valid global base, source size 0 or 4, no shared reader before completion. |
| `issue_stage` | `cp.async.commit_group`; commits the executing thread's prior uncommitted copies. | Each destination written once per group; each thread commits after issuing its words. |
| `await_stage` | `cp.async.wait_group 0`; completes this thread's committed groups. | All producers wait before CTA barrier; a wait alone is not cross-thread publication. |
| `barrier` | `bar.sync 0`; CTA synchronization and memory ordering. | All 256 threads reach matching barriers; no early return. Also protects completed readers against slot reuse. |
| `shared_word` | `ld.shared.b32`; reads a four-byte shared word. | Aligned address in published current stage; next copies target the other slot. |

The reused `nvfp4_linear::mma_nvfp4` owns the existing
`mma.sync.aligned.m16n8k64...block_scale.scale_vec::4X` site and its scale/lane
contract. `store_scaled_output` owns `mul.rn.f32` and BF16 RNE. Their inventory
entries remain applicable; there is no new floating-point arithmetic instruction
site in this file. No TMA, tensor-map binding, producer-specialized warp,
scale permutation, fusion or imported Ninfer code is included.

## Provenance and qualification

Technique reference is Ninfer `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`,
`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:130-220`, read from the parent's local
checkout. Its producer/consumer stage lifetimes informed the investigation;
this implementation uses ordinary async word copies and CTA barriers with the
repo's existing MMA, not Ninfer's TMA implementation or container format.

Parent must register the device module, scalar reference and PTX inventory, then
compile through the repo's Just lane. Run independent signed/nonuniform-scale
operator fixtures at K64,128,192 and larger K, M1,17,31,32,33,128,512 and N8,24,
32,40 plus real 5120/17408 dimensions. K192 explicitly exercises slot reuse.
Compare both raw FP32 and BF16 against the unchanged native kernel on identical
inputs, then separately compare the independent oracle under the parent's
frozen arithmetic policy. Run memcheck, racecheck and synccheck, including tails
and repeated launches. Real-weight same-input layer audits, whole-model state,
quality and paired uncontended 128/512-token timings remain mandatory. Tests in
the reference file check logical mapping/tails, signed cancellation, nonuniform
scales and shape admission; passing them cannot certify device indexing.


Parent compilation checkpoint: 276 host tests and host Clippy pass; three
Clippy findings in the A16 reference were repaired without arithmetic changes.
`just specialize-ptx` passes with nightly2026-09-25, SM120a. Retained PTX
`target/specialize/iterate-20260927/features-head-pipeline.ptx` SHA256
`e8beed4f5bba069809a7ff3d80398bea6f3980aa77a9a9f7352c4753f6ec401a`.
Both expected entry symbols are present. PTX declares local storage,40bytes for
A16 and144bytes for NVFP4; these are virtual PTX declarations, not measured JIT
register/spill/resource usage. GPU qualification and resident integration remain
pending. Existing baseline PTX remains retained separately.

## Bounded operator trial source

Added `src/kernels/cuda/nvfp4_pipeline_trial.rs` against parent-provided HEAD
`2e5f77d2f`. The worker ran only rustfmt, not Cargo, CUDA, or remote commands.
The parent reported host tests, Clippy and PTX compilation of the earlier
candidate; this new trial still needs parent registration, build and execution.

`run(ptx, device)` loads both symbols and records device, JIT log and resources
for each. Nineteen deterministic cases total 11,302,912 scalar products, below
the enforced 25-million-product budget. The shape list includes M1,17,31,32,33,
128,512 and N8,24,32,40, with K64,128,192 and selected small-M/N K5120 and
K17408 cases. One-hot K192 inputs expose every E2M1 code across 16 columns.
Dense signed dyadic cases have exactly represented sums and scales; general
cases vary finite E4M3 scales over rows and 16-value groups. K192 and long-K
cases exercise shared-slot reuse. Both symbols receive the same uploaded
codes/scales and factor 1.0. CPU oracle Matrix globals are both exactly 1.0.

Frozen admission requires finite outputs and stored BF16 equal to RNE of raw
FP32 for both symbols. Exact cases require raw/BF16 bit equality with the scalar
oracle. General cases require BF16 normalized L2 at most 0.01 and raw maximum
absolute error divided by max of 1 and maximum absolute oracle value at most
1e-4. Every case additionally requires candidate raw/BF16 bit equality with the
old native kernel. The JSON preserves numerical failures as `all_passed:false`,
including comparison counts, mismatch counts and errors. Nonfinite output
invalidates a case and its error metrics are null. CUDA/API failures return
errors; failed launches explicitly drain before buffers drop. Outputs begin as
quiet NaN poison. This is operator qualification source without timing or model
claims, and does not resolve the existing native-versus-integer arithmetic audit.

## Fixed producer paths resource experiment

The parent reports that the original candidate at `5dbe00e1d` passed all 19
operator cases normally and under memcheck, racecheck and synccheck, including
raw/BF16 bit equality with the native baseline. Reported CUDA resources were
38 registers, 4,608 shared bytes and 144 local bytes for the candidate, versus
53 registers and zero local bytes for the baseline. Those are parent-provided
observations; this worker did not rerun them. Original PTX is retained as
`features-head-pipeline.ptx`, SHA256
`e8beed4f5bba069809a7ff3d80398bea6f3980aa77a9a9f7352c4753f6ec401a`.

Replaced the dynamic matrix/row/origin array indexing in `issue_stage` with
named `MatrixStage` fields and separate fixed A/W paths. Thread t now writes
A code word t and W code word t. Threads 0..31 additionally write A scale word
t and W scale word t. This redistributes the original scale-copy producers
without changing any destination byte or logical source. All 256 threads
reconverge before their single group commit. Tail rows still use source size
zero with the live allocation base. Stage offsets, waits, publication and
reader-retirement barriers, MMA order and output ownership remain unchanged.
The existing PTX sites and safety contracts apply; no instruction was added.

The intended effect is to let scalar replacement remove dynamically indexed
local arrays. Local-byte removal, register use, numerical qualification and
performance are unverified for this revision. Only rustfmt and its check were
run. Parent must compile and compare resources, rerun the unchanged numerical
and sanitizer trial, and measure original versus revised PTX before selecting
this scheduling change. Arithmetic and qualification limits were not changed.


Parent retained completed `nvfp4-pipeline-check-1` evidence at5dbe00e1d.
All19 cases pass normal, memcheck, racecheck and synccheck with zero reported
errors/hazards. Candidate and native control raw/BF16 outputs agree bit-for-bit.
Maximum independent-oracle scaled raw error is8.729596730380663e-6, within the
frozen1e-4 gate. Original resources are38registers/4608shared/144localbytes.
Ninfer stayed inactive and ComfyUI498MiB was preserved.

Parent added `MESH_SPECIALIZE_NVFP4_PROFILE=baseline|tiled-prefill`, absent means
baseline. Tiled dispatch covers16..512 rows, N divisible by8 and K divisible by64
within the kernel's dimensions. Smaller or unsupported shapes retain baseline;
one-row integer decode is unchanged. Both ordinary allocations and MLP workspace
use one checked schedule selector. Bench/profile/logit manifests record the NVFP4
profile independently of FP8 and attention. MTP rejects tiled-prefill until its
own recovery qualification. Existing strict model partition checks are unchanged.

The fixed-producer revision compiles with Just, nightly2026-09-25 sm_120a.
Retained artifact `features-nvfp4-tiled.ptx` SHA256
`83f52bb4e754f336f46f8d2a333daa4d08e27a02dceb4504f1aeeb5087ec7513`.
Host tests and Clippy pass. GPU resource, repeated operator/sanitizer checks and
original-versus-revised model timing remain pending at this compilation checkpoint.


`nvfp4-pipeline-check-2` at e68e93d95 passes the unchanged19 cases and all three
sanitizers. Raw/BF16 bits still match native control. Revised JIT resources are
37registers,4608shared bytes,112local bytes. Parent inspected PTX local accesses:
the remaining stack storage belongs to the final array-of-four-output-tuples
iterator. Four explicit stores are a possible follow-up, pending measured model
results for the existing versions. 278 host tests and host/Linux Clippy pass.
Model comparison `nvfp4-tiled-model-1` uses128/512 inputs, exact FP8/attention,
MLPworkspaceon, and retained original/revised PTX as separate schedules. This
entry records the planned scope, not a completed result.

## Separate 32x128 register-reuse candidate

Parent-reported full-model prefill medians for baseline/original/fixed32x32
were 277.414/215.605/278.192 tokens/s at 128 inputs and
305.580/233.389/306.601 at 512 inputs. The fixed32x32 schedule did not establish
a meaningful win. These are parent observations, not measurements by this
worker; the original and fixed kernels/PTX remain retained.

Added only `kernels/nvptx/nvfp4_prefill_wide.rs`, symbol
`nvfp4_prefill_wide`. The six-pointer/MNK/factor ABI, admitted dimensions and
arithmetic contract match `nvfp4_prefill_tiled`. Launch differs: grid is
`[ceil(N/128),ceil(M/32),1]`, block `[256,1,1]`. Existing
`nvfp4_prefill_tiled_reference` is the independent logical oracle for the
identical input representation and shape bounds; no new fragment-derived oracle
is introduced.

Warp w owns rows `16*(w/4)..+16` and columns `32*(w%4)..+32`. Four named
accumulators cover offsets 0,8,16,24 within those columns. `load_a` loads the
four A registers and scale once per K64 step. Four explicit `accumulate_b`
calls reuse those registers against separate B fragments. Every output still
receives ascending K64 MMA steps through the existing `mma_nvfp4` helper.
The epilogue makes four explicit `store_fragment` calls, each containing four
explicit existing `store_scaled_output` calls. It has no tuple-array iterator
or dynamically indexed accumulator array. Global-factor RN and BF16 RNE remain
owned by the existing helper. Actual register reuse and absence of local
storage must be checked in emitted PTX and JIT resources.

Each 5,760-byte stage contains A codes at 0..1024, W codes at 1024..5120,
A scales at 5120..5248 and W scales at 5248..5760. Total shared storage is
11,520 bytes. Thread t writes A word t and W words t,t+256,t+512,t+768.
Threads 0..31 write A scale word t; threads 0..127 write W scale word t.
These producer sets cover every aligned word exactly once, including tails.
Invalid rows use zero source size with the live allocation base. No invalid-row
pointer is formed. K64 admission means no partial code or scale word is read.
The maximum A fragment accesses row31/code byte31 and scale byte127; maximum W
fragment accesses row127/code byte31 and scale byte511, within their regions.

Every thread commits one group after its five to seven copies. The original
wait-group-zero and CTA publication barrier precede consumption; next-slot
copies precede four current-slot MMAs. The terminal CTA barrier retires all
readers before slot reuse. Tail threads participate in every MMA and barrier.
Output columns owned by different warps/fragments do not overlap, and existing
store guards remove M/N tails.

PTX inventory alias for parent registration: `coordinates`, `copy_word`,
`issue_stage`, `await_stage`, `barrier` and `shared_word` have the instruction
semantics and compiler clobbers listed in the initial inventory above. The
unique static shared symbol is `nvfp4_prefill_wide_stages[11520]`; its extent
and producer proof are replaced by this section. Native MMA and output
instruction sites remain in `nvfp4_linear`. No additional PTX instruction kind,
TMA, swizzle, scale permutation or fusion is introduced.

Technique references at Ninfer `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`
are `src/ops/linear/nvfp4/nvfp4_a4_mma.cuh:191-275` and
`nvfp4_a4_tma.cuh:250-309`. These retain A fragments while iterating multiple
output-column MMAs. This Rust candidate uses the repo's existing logical layout
and MMA helpers; no Ninfer implementation is imported.

Only rustfmt and its check were run. Compilation, resources, sanitizer results,
operator comparisons and model timing remain unmeasured for the wide symbol.
Parent must extend the unchanged numerical trial to N120,128,136, including
M tails and K64/128/192 plus bounded long-K cases. Require independent oracle
gates and raw/BF16 bit equality with the original native kernel, followed by
all three sanitizers and paired full-model timing. No speedup is presumed.

### Parent wide-candidate registration and preflight

Registered the separate symbol and all six assembly sites. The trial now uses
33 fixtures and 23,515,136 independent CPU products, reusing each oracle output
for baseline, 32x32 and 32x128 comparisons. Added N120/128/136, M boundary cases,
and bounded long-K wide-output fixtures. Both candidates must retain raw/BF16
bit equality with native baseline; existing oracle limits are unchanged.

At parent base `72d7a4e2e`, `just specialize-ptx`, macOS Clippy, all 278 host
library tests and `just no-console-print` pass. Preserved new PTX as
`target/specialize/iterate-20260927/features-nvfp4-wide.ptx`, SHA256
`dd54c51a775eb95634b456caa03198e53954646727658de126e4ed5771bae162`.
The new entry has no PTX local declarations or local load/store operations.
This is compiler evidence only; JIT resources and GPU checks remain pending.
Prior PTXs and full-model evidence are preserved. Remote preflight found Ninfer
active and ComfyUI resident at 498 MiB; the next trial must restore the service
if it stops it.

### Wide operator qualification and separate model profile

`nvfp4-pipeline-check-3` at source `475c1b35f` passes all 33 cases in normal,
memcheck, racecheck and synccheck execution. All sanitizer summaries report zero
errors/hazards. Wide and tiled outputs match the native baseline in raw FP32 and
BF16 bits. Maximum wide raw error against independent FP64, scaled by maximum
oracle magnitude, is 8.729596730380663e-6 under the unchanged 1e-4 limit.
JIT resources for wide are 62 registers, 11,520 shared bytes, zero local bytes.
Linux Clippy and Just release tool build pass. Ninfer was active before this
trial; the bounded stop was restored with health HTTP200. ComfyUI remained
resident at 498 MiB. This operator trial measures no performance.

Added `MESH_SPECIALIZE_NVFP4_PROFILE=wide-prefill`, separately named
`nvfp4-tiled128-prefill-integer-decode-v1`. It uses the existing admission of
16..512 rows, N divisible by8 and K divisible by64, each <=32768. One-row
integer decode and smaller-batch native dispatch are unchanged. Both ordinary
and workspace MLP paths already share the profile selector. MTP remains gated
to baseline. Default remains baseline; full-model qualification is pending.
