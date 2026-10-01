# Synthetic Q8 FC candidate qualification harness

Status: the synthetic Q8 FC candidates pass the recorded GPU qualification in an
isolated, uncommitted snapshot. This qualifies only the six exercised synthetic
cases; it does not admit native MTP or qualify real weights, a model, or throughput.

## Interface and ownership

`src/kernels/cuda/native_mtp_q8_fc_operator.rs` exposes
`pub(in crate::kernels) fn run(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value>`.
The parent owns module registration, command registration, compilation, PTX
identity, build validation and GPU execution. No existing files were changed.
The harness JIT-loads supplied PTX and resolves only the two new FC entries.
It requires SM80+ instruction support; CUDA JIT remains responsible for rejecting
PTX compiled for an incompatible specific architecture. This is not admission.

Both entries launch grid `[320,1,1]`, block `[256,1,1]`, no dynamic shared memory.
The ABI is separate codes, FP16 scales, BF16 input, BF16 output pointers followed
by one `u32` token count. C4 receives T1; C8 receives T5. Every plane is a full
5120-by-10240 allocation or its full corresponding scale/input/output extent.
Device allocations and host vector reservations return checked errors. Module
context and plane alignment are checked. Launch success and failure both drain
the context before buffers leave scope. Each fixture launches twice into the same
output allocation, freshly poisoned with different BF16 NaNs before each launch.

## Coverage and arithmetic

Six cases cover three fixtures for each entry. The dense fixture includes signed
extremes -128/127, row/K-varying signed codes, neighboring positive power-of-two
FP16 scales and signed BF16 activations. Five columns use distinct multipliers
`1,-1,2,-2,0.5`. Every K iteration, split, CTA and live output is executed.

The dense fixture's terms are multiples of 1/8. Except for the first two codes,
code magnitudes are at most seven. Activations have magnitude at most two and
scales at most two. The absolute sum in units of 1/8 is less than 2^24 even over
all 10240 terms. Therefore group dots, split sums and reductions are exact FP32
for any summation order. Scalar-reference BF16 bit equality is valid for these
fixtures, without a tensor-core internal rounding assumption or new tolerance.

The last-K fixture has only K=10239 nonzero. A dense pair-cancellation fixture
uses row-dependent signed codes, alternating positive/negative activations,
and varying group scales. Every pair cancels except the last pair, whose final
activation is zero. Its surviving last-K contribution has a closed-form answer.
Both simple fixtures check all 5120 rows of every live column exactly on both
repeats. They include the last row, row-8/16 tile boundaries, multiple CTAs,
all eight split warps and the last staging iteration.

All six cases check every output for finiteness and exact repeat agreement.
The existing independent scalar/FP64 reference checks 15 selected rows:
`0,1,7,8,15,16,17,31,32,2559,2560,5103,5104,5118,5119`.
The dense case does NOT independently check every row's numerical result.
No dense dequantized weight matrix exists. The full packed object is retained
for reference validation, which still scans the full plane; expensive scheduled
and FP64 dot products are limited to the selected rows.

Reports retain case failures, mismatch counts, first 16 numerical mismatches,
selected-row scheduled FP32 values, independent FP64 values and the original
mathematical error bounds. Those bounds are evidence, never GPU acceptance
budgets. Infrastructure errors within a case become failed case records so
other cases can remain visible. Context/JIT/function/resource setup errors return
`Err`. The caller must honor `all_passed`, not just a successful `Result`.

## GPU qualification evidence

Two host tests cover independent cancellation/simple-oracle agreement and
poison/repeat failure detection at the final row. Host tests and Clippy results are
not part of this GPU trial. The parent reports that the Rust PTX build passed;
LSP diagnostics timed out, so there is no clean LSP type-check result.

Trial B passed normal execution, memcheck, racecheck and synccheck for all six
cases, with two repeats per case. Every case had zero selected-oracle BF16 bit
mismatches, non-finite outputs, repeat mismatches and simple-row mismatches.
The two signed-dense-dyadic cases compare the 15 selected rows against the
independent scalar/FP64 reference; they do not independently check every dense
row. The four `last-k-all-rows` and `dense-pair-cancellation-all-rows` cases also
check every row against their simple exact oracle. The independent FP64 error
bounds remain diagnostic and are not GPU acceptance tolerances.

| Entry | Fixture | Tokens | Trial B result |
| --- | --- | ---: | --- |
| `native_mtp_q8_sliced_k_fc_c4` | `signed-dense-dyadic` | 1 | Pass; 15 selected oracle rows, 5,120 outputs/repeat |
| `native_mtp_q8_sliced_k_fc_c4` | `last-k-all-rows` | 1 | Pass; all 5,120 rows checked per repeat |
| `native_mtp_q8_sliced_k_fc_c4` | `dense-pair-cancellation-all-rows` | 1 | Pass; all 5,120 rows checked per repeat |
| `native_mtp_q8_sliced_k_fc_c8` | `signed-dense-dyadic` | 5 | Pass; 15 selected oracle rows, 25,600 outputs/repeat |
| `native_mtp_q8_sliced_k_fc_c8` | `last-k-all-rows` | 5 | Pass; all 25,600 rows checked per repeat |
| `native_mtp_q8_sliced_k_fc_c8` | `dense-pair-cancellation-all-rows` | 5 | Pass; all 25,600 rows checked per repeat |

Sanitizer summaries report zero memcheck errors, zero racecheck hazards/errors/
warnings and zero synccheck errors. The complete JSON results and logs are retained
under [`trial B evidence`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-b/).
The tested PTX SHA-256 is
`850a8b0e8e93957631938b065d201122d85db6489737dc7b5a3f977f8391af88`; the
`xtask` executable SHA-256 is
`9162a07abb1271ad89d5cda80222c65b7d6767881aa6eb49658affe90613d867`.

Trial A is preserved, not overwritten. Normal execution, memcheck and racecheck
passed, but synccheck found divergent threads reaching a barrier in C4 and failed
with exit 97; the follow-on module query returned CUDA error 719. Its PTX hash was
`878fe90cd00f7f232b0fa69dd135f0f350d39c6aaaaac910cd204bb6bff5cfa0`. The source
correction made the store uniformly predicated while keeping the barrier
unconditional. Trial B's new PTX hash above passes synccheck. See
[`trial A synccheck log`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-a/synccheck.log)
and [`trial A results`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-a/results.txt).

The trial used an isolated snapshot based on baseline
`b188c2925aa05108ebb5b1fe7afd195c60254e6c`. The snapshot was uncommitted, not a
clean commit. Source manifests: trial A
[`fc-q8-source-20260930.sha256`](../evidence/fc-q8-20260930/fc-q8-source-20260930.sha256)
and trial B
[`fc-q8-source-20260930-b.sha256`](../evidence/fc-q8-20260930/fc-q8-source-20260930-b.sha256).
The recorded device was an NVIDIA GeForce RTX 5090 (SM 12.0, UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`) with driver 615.71.09; CUDA toolkit
13.4.92, Rust 1.98.1 and Compute Sanitizer 2026.3.0.0 are recorded in
[`versions.log`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-b/versions.log).
Both Ninfer and ComfyUI services were restored active, and Ninfer health returned
HTTP 200. See [`units-after.txt`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-b/units-after.txt)
and [`ninfer-health-restored.txt`](../evidence/fc-q8-20260930/fc-q8-trial-20260930-b/ninfer-health-restored.txt).

The candidate remains synthetic-only and is not promoted or admitted for native
MTP. No real-weight, whole-model, MTP execution/admission, or model-throughput
claim follows from these results. Inactive shared columns remain intentionally
unstaged in the device candidate; this trial does not qualify their use.
