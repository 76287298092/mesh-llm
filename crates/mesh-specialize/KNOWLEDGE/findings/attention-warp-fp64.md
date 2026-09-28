# Exact-order FP64 warp attention

Status: implemented, unqualified. Default attention is unchanged. No GPU or model
performance is claimed. This work starts from parent checkpoint `56d426cb0` in
`/Users/ndizazzo/dev/worktrees/ninfer-direct-runtime` on 2026-09-28.

## Schedule and arithmetic contract

`MESH_SPECIALIZE_ATTENTION_PROFILE=warp-fp64` selects the report profile
`bf16-warp-fp64-exact-order-v1`. Legacy and eager-stream execution use it only
for M=1 with 24 query heads, 4 KV heads, and D=256. Every larger row count uses
the unchanged `causal_attention_bf16` kernel. Existing online and FP32 split
profiles are unchanged. Graph capture and MTP continue to reject this profile.
No device-position warp variant is qualified or supplied.

`causal_attention_warp_fp64` retains the control ABI: five pointers (Q, cache K,
cache V, BF16 output, raw FP32 output), six u32 values (rows, query heads, KV heads,
width, past, capacity), then FP32 scale. Launch grid is [96,1,1], block [32,1,1].
One warp owns one (query head, 64-channel output shard). Each lane owns two V
output channels, but loads all eight Q/K stripes at lane+32*c. All four shards
repeat the complete QK dot and visit keys strictly in ascending order.

The local dot tree is exactly
`((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7))`, followed by paired-word FP64 DOWN shuffles
at offsets 16,8,4,2,1 and a lane-zero broadcast. The existing rounded FP64
multiply/add/subtract/divide and conversion helpers are reused with visibility
changes only. The original `exponential::exp_nonpositive` implementation is
unchanged. Maximum, denominator, and each channel's accumulator preserve the
control's online recurrence and operation order. No split sequence, approximate
exp2 substitution, BF16 beta, FMA contraction, shared memory, or CTA barrier is
introduced. Each lane duplicates the softmax scalar recurrence; its runtime cost
and compiler resource consequences require measurement.

The intended tradeoff is four times as many CTAs (24 to 96) and no per-key CTA
barriers, at the cost of duplicated K reads/dots. This is a scheduling hypothesis,
not evidence of a speedup. The failed FP32 split quality gate is not relaxed.

## Required qualification

The `attention-warp-check --ptx PATH --device ORDINAL --output NEW_FILE` xtask
subcommand preserves a report on numerical failure. It requires exact equality
of **every raw FP32 bit and BF16 output bit** to the unchanged GPU control, plus
the existing independent logical FP64 oracle's component and per-head budgets.
The host operation-order simulator is supplemental evidence, not that oracle.

The synthetic matrix covers M=1 at past 0,1,32,127,512,8191,32767, plus uniform
scores, signed cancellation, signed zeros, and wide-exponent finite BF16 inputs.
All cases use NaN-poison unused KV tails, leading/trailing output guards, complete
input/cache readback, and same-address repeated execution. Host tests compare
the independent 256-slot tree mapping with the eight-register/32-lane mapping,
including finite BF16 exponents -120 through 120, and check the complete recurrence.
Plan tests reject M=5 and malformed geometry/capacity. Source checks assert the
fixed geometry and forbidden barrier/shared/approximate-exponential instructions.

Only passing cases receive event timing: three warmups and five repetitions of
each control and candidate, separate CUDA-event sample arrays and medians. The
report records driver-JIT resources for both symbols. Times are synthetic kernel
latencies, never model prefill/decode. Optional 131K GPU coverage is not included.

Parent qualification remains required: serial host/Linux checks and Clippy, PTX
and SASS inspection (including arithmetic order, register/local memory and absence
of barriers), all three CUDA sanitizers, identical whole-model input/logit/state
comparisons, and repeated matched whole-model timings. Preserve raw failures.
Device, driver, toolchain, clock/power, and uncontended model timing are **not
measured** in this implementation handoff. No promotion is authorized by source
or synthetic host checks alone.


## Measured rejection, September28

At aac74c4c5, all11operatorcases and106/512-input modelchecks preservecontrol
raw/logit/state bits. Memcheckpasses; unfilteredracecheck timesout at1200seconds.
Candidate-filtered racecheck/synccheck thenpass. Those filterlimits are explicit.
Balanced modelmedians REGRESS:26.479→25.603tok/s after106inputs and
18.166→17.034 after512inputs,256outputs,foursamples/profile. Do notpromote.
Evidence: evidence/reassess-20260928/warp-attention-1 and warp-attention-2.

Both originalandwarpkernels use152localbytes. PTX725e13e... shows a136-byte
Taylorcoefficientarray copiedbyte-by-byte into localmemory insideeachnontrivial
exponential call, then17local FP64loads/mul/add steps. Warpduplicates thiswork
acrosslanes/outputshards. This is source/assemblyevidence, not an independent
wall-time attribution. A separate unrolledHorner experiment will preserve
coefficientbits, operationorder androunding to isolatethis generatedtraffic.
