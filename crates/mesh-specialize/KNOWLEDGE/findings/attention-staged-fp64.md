# Three-stage exact-order FP64 attention

Status: implemented, unqualified, 2026-09-28. This is a separate M=1 schedule,
not a default change or a claim of faster model decode. Earlier warp, unrolled,
FP32 split and online profiles remain available and unchanged.

The parent reported bit-identical whole-model results with modest gains from
unrolling the exponential (26.47 to28.04 tokens/s at106 inputs;18.17 to20.24 at512).
Those measurements belong to the preceding experiment, not this schedule. The
new hypothesis is to remove per-key CTA barriers without duplicating dots or
exponentials and without reassociating a single recurrence.

## Ordered stages and ABI

Fixed geometry: M1,24 query heads,4 KV heads,D256, scale1/16. `length=past+1`.
Source K/V stays token-major BF16. All launches use the SAME CUDA stream:

1. `attention_staged_scores_fp64(Q,K,workspace,length,capacity,scale)`:
   grid[length,24,1], block[32,1,1], one warp per key/head. Each lane forms eight
   rounded FP64 products. Reuses the exact warp tree
   `((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7))`, then paired-word DOWN16/8/4/2/1 and
   lane-zero broadcast. Only lane zero stores `mul.rn(dot,scale)`.
2. `attention_staged_coefficients_fp64(workspace,length,capacity)`:
   grid[24,1,1], block[32,1,1]. Only lane zero visits keys in ascending order.
   Maximum, alpha, beta and normalizer use the original operation order and the
   separately qualified unrolled exponential. The normalizer is NEVER parallel
   reduced. Writes each alpha/beta and the final normalizer.
3. `attention_staged_values_fp64(V,workspace,BF16_out,FP32_out,length,capacity)`:
   grid[24,4,1], block[64,1,1]. Each thread owns one output channel. It visits
   keys strictly ascending, using separate rounded FP64
   `add(mul(acc,alpha),mul(beta,V))`, then the original division/FP32/BF16 rounding.

Pointer parameters are64-bit; length/capacity areu32 and scale isFP32. No stage
uses CTA barriers or changes the addition tree. Inter-stage visibility comes
from ordered launch completion, not an in-kernel global synchronization trick.

## Workspace, admission and lifetime

One aligned allocation holds FP64 scores, alpha and beta, each[24,capacity],
followed by24 FP64 normalizers. Extent is `8*(3*24*capacity+24)` bytes:75,497,664
bytes at131,072 capacity, or72 MiB plus192 bytes. Pure host planning checks M1,
fixed geometry, positive bounded capacity, past overflow, workspace extents,
pointer overflow, alignment and pairwise nonaliasing of all six live ranges.
Key grid uses x rather than y, avoiding CUDA's65535 y-grid limit at large capacity.

Every stage accesses only each head's[0,length) prefix. All relevant scores,
coefficients, normalizers and outputs are overwritten on EVERY invocation.
Capacity stride never changes when rewinding length. Unused workspace and KV
suffixes are not read or written. Prior-layer contents have no semantic role.

Legacy creates scratch and resolves all stage handles before KV append; it keeps
scratch, inputs and outputs through the final success/error context drain.
Eager-stream execution allocates one persistent workspace and resolves handles
once at construction, reused across layers/forwards with no enqueue allocation.
Admission checks session capacity before enqueue. Every failure path drains the
same stream while all owners remain live, including a failure after stage1/2.
Chunked-benchmark memory admission includes the full persistent workspace.

`MESH_SPECIALIZE_ATTENTION_PROFILE=staged-fp64` reports
`bf16-staged-fp64-exact-order-v1`. Only M1 uses the three stages. Every larger
batch uses original `causal_attention_bf16`, not the unrolled candidate. Graph
capture and MTP reject this profile pending separate position/recovery proof.

## Qualification and sanitizer scope

`attention-staged-check --ptx PATH --device ORDINAL --output NEW_FILE` requires
ALL raw FP32/BF16 bits equal original control plus the unchanged independent
FP64 attention oracle budgets. Inputs include uniform scores, cancellation,
signed zeros, wide BF16 exponents, and poisoned unused KV capacity. Tests inspect
leading/trailing guards and every workspace suffix word; repeat on the same
addresses; rewind long-to-short while future KV remains initialized; poison ALL
workspace intermediates and repeat the short prefix; then restore the long prefix.
All input/cache bytes must remain unchanged. Resource records cover all3 stages.

Defaults are intentionally sanitizer-safe:

- `MESH_SPECIALIZE_STAGED_TRIAL_SCOPE=short`: past0/1/32/105/127 (lengths1/2/33/106/128), plus special fixtures.
- `MESH_SPECIALIZE_STAGED_TRIAL_TIMING=off`: no warmup/event timing repetitions.

For normal full qualification, explicitly select scope `full` to add past511/8190 (lengths512/8191).
After the parent's whole-model bit-identity gate, timing `on` adds three warmups
and five complete-chain CUDA-event samples for each passing candidate/control
case. Operator timing is not model throughput. Do NOT enable full/timing-on
inside sanitizers merely to reproduce a normal timing report.

Parent gates remain: host/Linux tests and Clippy; emitted PTX/SASS order/resources;
short timing-off memcheck/racecheck/synccheck; full normal context matrix; same
whole-model inputs/logits/all-state identity before matched model timing. No
Cargo, PTX build, SSH, GPU execution, exact source revision lookup or performance
measurement was run by this worker. Device/driver/clock results for this schedule
are not measured. Preserve failed evidence and retain all earlier candidates.
