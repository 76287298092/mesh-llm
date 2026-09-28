# Isolated unrolled FP64 exponential

Status: implemented, unqualified on 2026-09-28. Default attention, original
`exponential.rs`, and the separate warp/split/online candidates remain unchanged.
This is an independent coefficient-delivery experiment, not further warp tuning.
Parent checkpoint at assignment: `b1e7`; no build or GPU run by this worker.

## Emitted-control evidence

Inspection of `target/specialize/probes.ptx`, SHA-256
`725e13e9637b6b91ea9a3ed19216fb984711a91dfb1339f7b807a00fa64e82c7`,
confirmed that control and warp attention each declare a 152-byte local frame.
The coefficient bytes occupy offsets16..151; no explicit access uses the leading
16 bytes. PTX line20's 136-byte constant exactly decodes to all17 original FP64
coefficient bits. For control, lines15721..15750 and15793..15822 show per-call
136-iteration global-byte-load/local-byte-store copies followed by17 local FP64
loads and separate rounded multiply/add Horner stages. Warp has the same pattern
at16688..16717 and16760..16789. No `fma` occurs in either attention entry.
These are explicit local array accesses, not evidence of Q/K register spills.

The parent reported that warp retained complete tested output/state bits but
regressed whole-model decode:26.479 to25.603 tokens/s at106 inputs and18.166 to
17.034 at512. The failed candidate remains intact. The local-copy evidence alone
does not prove the regression's cause and does not establish an unrolling gain.

## Independent candidate

`exponential_unrolled.rs` preserves branch order, NaN return payload, positive/zero
shortcut, cutoff below-745, range reduction, cast behavior, initial positive zero,
17 coefficients and Horner stages, exponent splitting and final multiplication
order. Coefficients are fixed binary64 encodings of the original divisions.
Device operations reuse explicit `mul.rn.f64`, `add.rn.f64`, and `sub.rn.f64`
helpers; no fused multiply-add or coefficient array iterator is introduced.
Host strict-FP64 implementations exist only for supplemental order checks.

`causal_attention_unrolled_fp64` uses the original CTA256 attention body with a
compile-time true exponential selector. Original and device-position entries
explicitly select false and continue calling the untouched exponential. Nothing
else in attention's dot, reduction, online recurrence or output rounding changes.
The unchanged five-pointer/six-u32/FP32-scale ABI launches grid[rows*Qheads,1,1]
and block[256,1,1]. Source API specializations still require emitted-code review.

`MESH_SPECIALIZE_ATTENTION_PROFILE=unrolled-fp64` selects report name
`bf16-unrolled-fp64-exact-order-v1`. Legacy and eager-stream dispatch use it for
all admitted row counts with original geometry. Stream construction resolves an
optional handle once and reports its presence. Graph and MTP continue rejecting
the nondefault profile until separate position/recovery qualification.

## Qualification tools and remaining gates

- `exponential-unrolled-check --ptx PATH --device ORDINAL --output NEW_FILE`:
  standalone old/new GPU helper comparison using explicit raw u64 FP64 bits.
  Inputs cover dense[-128,0] at1/1024 spacing, a dense neighborhood of-745,
  representable neighbors of range-reduction boundaries, gradual underflow,
  signed zeros, infinities, and signed quiet/signaling NaN payloads. Reports all
  mismatch counts, first mismatching bits, special-input bits, guards, three
  repeats, immutable input checks, and both resource records.
- `attention-unrolled-check` with the same arguments first requires the helper
  bit gate, then compares every raw FP32/BF16 output bit with unchanged control
  plus the existing independent FP64 oracle and unchanged component/head budgets.
  Cases include M1 past0/1/32/127/512/8191/32767, uniform/cancellation/signed-zero/
  wide-exponent inputs, M17 past32, and M128 past0. Guarded outputs, poison tails,
  repeated same-address execution and complete input/cache readback are required.
  Passing cases get three warmups/five CUDA-event samples for each kernel.

Parent must inspect new PTX/SASS for zero coefficient copies/local arrays and
unchanged rounded arithmetic, then qualify resources, all three sanitizers,
whole-model same-input/logit/state identity and repeated matched model timings.
No candidate PTX build, device/toolchain/clock record, GPU equivalence or model
speedup is established here. Host comparisons do not substitute for device bits.


## Measured isolated result

At b58527962, helper bit checks, strict attention checks and same-input native
model checks pass. Candidate PTX has no local coefficient array/loads/stores and
no FMA; control retains 152 local bytes. Balanced 256-output medians: 106-input
26.473→28.038 tok/s; 512-input 18.170→20.238 tok/s. This is a modest gain, not
the large context gap. Memcheck passes; broad candidate racecheck timed out at
1200 seconds. A bounded retry instruments 48 launches covering past 0/1/32/127,
plus one M17/past32 launch; it passes, as does candidate-filtered synccheck.
Preserve these coverage limits and the failed run. No default change. Evidence:
reassess-20260928/unrolled-attention-1 and -2.
