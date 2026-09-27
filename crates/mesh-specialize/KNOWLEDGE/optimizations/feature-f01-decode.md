# F01: FP8 A16 decode GEMV candidate

Status: synthetic GPU checks and sanitizers pass. Parent one-row resident
integration is authored behind `MESH_SPECIALIZE_FP8_PROFILE=a16-decode`;
initial real-model timings and same-input logit differences are measured, and
one-row model sanitizers pass. Bounded complete-answer checks are recorded;
broad quality and MTP qualification remain open.
This is a separate A16 arithmetic profile; it does not replace or relax any A8
decode qualification gate.

## Candidate contract

The NVPTX entrypoint is `fp8_a16_decode` in `kernels/nvptx/fp8_a16_decode.rs`:

```text
fp8_a16_decode(
    input: *const u16,
    weight: *const u8,
    weight_scale: *const u16,
    out: *mut u16,
    unrounded: *mut f32,
    n: u32,
    k: u32,
)
```

Input is one BF16 activation row of length K, without activation quantization.
Weights are row-major E4M3FN bytes with shape N by K. Each output row has one
represented BF16 scale, applied after its dot. The outputs are scaled FP32 and
the same values rounded to BF16 with round-to-nearest-even. The kernel launch is
`grid = [ceil(n / 4), 1, 1]`, `block = [128, 1, 1]`: four warps per block, one
output row per warp. The host must use checked shape arithmetic and reject zero
N or K before launch.

Each lane owns four adjacent K elements per iteration and advances by 128
elements. Four independent FP32 FMA chains accumulate those positions before a
full-warp XOR reduction. The aligned path loads four BF16 values in one 64-bit
load and four E4M3 codes in one 32-bit load when the activation base is
8-byte-aligned and that output row's weight start is 4-byte-aligned. Other rows
use guarded scalar loads. The final partial group uses scalar loads with per-K
guards. Invalid N rows contribute zero, and every lane reaches the warp
collectives.

The reference in `reference/fp8_a16_decode.rs` decodes the logical E4M3FN
sign/exponent/fraction fields, accumulates BF16 times E4M3 in FP64, applies the
represented BF16 scale, then converts to FP32 and BF16. The candidate instead
uses FP32 FMA chains and a tree reduction. Differences are expected, especially
under cancellation and near BF16 rounding boundaries. The profile is distinct
from both the existing A8 decode path and the CPU FP64 oracle. Finite E4M3FN
codes and finite BF16 activations/scales are required; positive and negative
zero scales are accepted.

## Integration and qualification

Do not mix this profile with existing A8 target verification. Initial host
integration must use the same per-row A16 arithmetic for ordinary decode and
MTP target verification. Until a separate small-batch A16 kernel is qualified,
verification can batch repeated rows through this entrypoint. No default
dispatch change is part of this candidate.

Before dispatch, the parent should register the NVPTX source and CPU reference,
compile PTX, JIT and launch the entrypoint, and check generated resource usage.
Compare against the independent oracle using nonuniform signed values, positive
and negative zero scales, E4M3 subnormals and maximum finite codes, odd K tails,
and cancellation fixtures. Record maximum absolute and relative FP32 error,
per-output BF16 mismatches, and the predeclared A16 acceptance bounds. Keep all
existing A8 checks unchanged. Run the relevant CUDA sanitizers after numerical
checks. Only then measure the isolated kernel and matched model impact with the
same inputs, weights, scales, device, and timing boundary; report the A16 result
separately from A8.

## Evidence record

- Model and real weights: not used; this source candidate has no model run.
- GPU architecture, driver, clocks, and Rust/PTX toolchain: not recorded; no GPU
  or compiler run was performed by this worker.
- Reproduction command: pending parent registration and build integration.
- CPU cases: source tests cover zero and signed scales, finite FP8 extremes,
  subnormals, an odd K tail, cancellation, and BF16 RNE ties; tests were not run.
- PTX/JIT correctness, sanitizers, registers, occupancy, memory, and latency:
  not measured.
- Expected result: compilation and a finite, tail-safe output for valid inputs;
  numerical differences from A8 and the FP64 oracle are evaluated under a
  separately declared A16 gate.
- Observed result: source and reference candidate only; no runtime result.
- Revision and evidence artifact: parent should record the integration revision
  and qualification artifact when the candidate is built and exercised.

Durable rule: an A16 speed claim requires a measured same-shape comparison, and
an A16 correctness claim requires its own numerical gate. Neither follows from
the existing A8 qualification or from this source implementation.

Parent integration: registered the host modules; 223 macOS library tests and
Clippy pass. F01 PTX compiles through `just specialize-ptx`. An initial oracle
fixture had N/K extents inconsistent with its two activation values; corrected
to a two-by-two diagonal case without relaxing its expected results. Linux CUDA
wrapper checks and GPU/model qualification remain pending. No resident default
was changed. Logs, including initial failures, are under
`KNOWLEDGE/evidence/features-20260927/`.

Parent GPU check, 2026-09-27: eleven synthetic F01/F02 cases passed on Carrack RTX5090, including M/N/K tails and K=5120. BF16 outputs matched the independent fixtures exactly; native FP8 raw FP32 scaled error was at most 9.58e-7. Workspace stable-address reuse and aborted-lease poisoning passed. All three CUDA sanitizer tools reported zero errors/hazards. Evidence: `../evidence/iterate-20260927/features-projection/`. PTX SHA256 `35c12bcfe57d282985b02bae256be0770991816519bc1a6cbf825a9cf369aa75`. JIT decode uses 33 registers and no local memory; prefill uses 56 registers, 64 local bytes and 6144 shared bytes. These are synthetic correctness checks, not model qualification or speed measurements. Ninfer and other GPU processes remained running.


## Parent resident integration

`a16-decode` selects the candidate only for one-row FP8 projections, including
one-row output-head calls. Multirow projections retain exact A8 arithmetic.
This hybrid experimental profile is not partition-equivalent by construction;
it does not relax the exact profile's checks. MTP rejects it until a consistent
verification profile is qualified. The default remains exact A8.

Optional `MESH_SPECIALIZE_LOGIT_DUMP_DIR` on model-profile diagnostics creates a
fresh directory containing raw BF16 logits for whole and token-partitioned
prefill and one subsequent decode, plus hashes and input IDs. It does not change
model inputs or outputs. Cross-process exact/A16 comparisons must use identical
prompt and teacher-token IDs and artifact/PTX identities; within-profile partition
metrics alone are not an A16-versus-exact comparison. Neither throughput nor
quality improvement is claimed before real-model evidence exists.

Initial Linux compilation caught unsupported hexadecimal formatting on the current SHA digest type in the diagnostic exporter. It now uses the existing hex encoder; no GPU trial had started.


## First resident ablation, 2026-09-27

Source `a17e7b8a73c5cff4d77a0bb7ba042009a9a08f13`, unchanged PTX
`089e027984eb7937fef47f5de3c01e92f08108d832eda1937ffae68eddbfd782`.
Three repetitions per profile, two natural-language prompts, 32 fixed output
tokens, same verified artifact and GPU0. Sequential exact/A16 batches share the
GPU with ComfyUI; these are not matched Ninfer measurements.

| Prompt | Exact median tokens/s | A16 median tokens/s |
| --- | ---: | ---: |
| Python Fibonacci | 24.25 | 26.42 |
| Bicycle prose | 24.18 | 26.65 |

Cross-process raw BF16 logit exports verify equal weights, PTX, prompts and the
same teacher-forced continuation token. Python prefill/next-position KL(exact||A16)
is 0.00000471 / 0.00000782; prose is 0.002103 / 0.00005645. Greedy tokens agree
at both sampled positions. Python's next-position relative L2 is 0.24091 despite
its small KL: unnormalized-logit norm drift alone is not semantic-quality
measurement. Python's first 32 generated tokens match; prose diverges at index8.
Two positions per prompt and two short completions do not establish broad quality.

Same-profile control/profile output and state are exact. Whole-versus-token
partition checks fail as expected for hybrid A16/A8 scheduling and are not
relaxed. One-token prefill plus one teacher-forced decode uses A16 throughout;
full-model memcheck, racecheck and synccheck all pass with zero errors/hazards,
and that one-row schedule's strict same-profile checks pass. This is not MTP
qualification; non-exact MTP remains rejected.

Evidence: `../evidence/iterate-20260927/a16-model-2/`, including raw logits,
hashed manifests, comparison script, exact input IDs, process sampling and
compiler logs. The preceding `a16-model-1/` stopped on incorrect placement of
the optional `--teacher-token` suffix before A16 execution; it is retained.
Ninfer remained inactive before/after, and unrelated processes were preserved.
No default promotion. Full answers and broader teacher-forced coverage remain open.


The current Python teacher-token profile has 1,476 exact launches and 31.63 ms
recorded event time versus 1,243 A16 launches and 28.60 ms. FP8 projection time
itself is 9.91 versus 9.47 ms; eliminating 233 activation-quantization launches
removes another 2.61 ms. Thus this sample's gain is mainly avoided quantization,
not a large speedup of the projection dot. NVFP4 decode accounts for 8.24 ms in
the current exact profile. The much older 83.39% FP8 attribution in the initial
model profile must not be treated as the current optimized decoder breakdown.
CUDA event intervals are instrumentation evidence, not isolated hardware
instruction occupancy or bandwidth measurements.


### Complete answer follow-up

`../evidence/iterate-20260927/a16-answers-1/` records same-source exact/A16
512-output trials for both prompts, pinned to the same tokenizer used in earlier
text trials. All four answers reach end-of-turn. Python answer text before the
first end-of-turn is identical across profiles; after inspection, both functions
pass independent checks at n=-1,0,1,2,3,10,100. Prose differs: the A16 answer
contains the incomplete phrase "a few working together". Complete generation is
therefore not itself a qualitative pass. Scientific accuracy is not independently
qualified here. Raw post-end-of-turn output is retained but excluded from answer
assessment. These samples do not qualify serving EOS behavior or broad model
quality, and A16 remains experimental. Initial service state was inactive and
was preserved; unrelated processes remained running.
