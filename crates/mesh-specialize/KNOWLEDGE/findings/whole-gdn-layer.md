# Independent whole GDN layer reference

Status: independent real-weight whole-layer comparison passes the fixed criteria.
This gate addresses accumulated error through the entire first decoder layer.
It does not execute or qualify the full 64-layer model or its logits.

The independent Rust CPU path receives only the verified checkpoint weights,
original token IDs and zero initial convolution/recurrent state. It composes the
existing scalar references from embedding and input norm through QKV/Z/A/B,
convolution, GDN preparation, ordered-FP32 recurrence, gated norm, attention
output projection, first residual and post-attention norm, NVFP4 MLP and final
residual. Every intermediate comes from that CPU path. It never receives a GPU
output or a device intermediate as an oracle input.

The chosen arithmetic profile remains explicit: f64 dot/norm/transcendental
oracles with the documented BF16 boundaries, ordered-FP32 recurrent reductions
and FP32 activation local scales. This compares the implemented quantized model
profile, not a full BF16 checkpoint or another runtime's exact numerical profile.

Before seeing real results, the parent sets an engineering error budget of
normalized L2 at most 0.01 and cosine similarity at least 0.9999. Both must pass
for the final hidden vector, raw convolution history and recurrent state.
The same criteria also apply to each token's hidden vector, each history row
and each recurrent head separately, preventing good aggregate metrics from
masking an isolated failed partition. For a zero reference vector, only a zero
actual vector passes. Finite values and exact extents are mandatory. Reports
include maximum/RMS absolute error and exact FP32/BF16 element counts; these
counts remain diagnostics rather than an exact-parity claim.

The generic comparator uses compensated f64 energy and dot-product sums. A unit
fixture deliberately changes one small partition while keeping aggregate error
small; the partition gate must reject it. Other tests cover scaling, angle,
signed zero, zero energy and malformed/nonfinite vectors. The layer composition
has a tiny zero-branch fixture with nonzero embeddings and explicit residual
identity, plus missing/invalid connection cases.

GPU wrappers now retain the actual final hidden words, convolution history and
recurrent state readbacks for this comparison. They do not replace any device
inputs. The existing component checks remain in place. A failed layer budget
sets the case and overall report false; the command persists the full numerical
report before returning failure, so failed evidence is not lost.

Two bounded Luna-max tasks own the CPU layer composition and generic comparison.
The parent owns their interfaces, fixed thresholds, per-token/head grouping,
retained readbacks, report integration and deployment. No device kernel or
launch changes are included. Existing sanitizer evidence for the identical
kernels and launches remains in the MLP qualification entry; this change needs
fresh host tests/Clippy and a normal real-weight whole-layer trial.

Reproduce with `just specialize-tools-build`, then the existing
`qwen-projection-check` command using the unchanged MLP-trial PTX. Schema 9 adds
`whole_layer_reference`. Fresh evidence belongs under
`target/specialize/qwen-layer-reference-20260927/`. Model performance/context
fields remain null. Timing from this harness includes independent CPU execution
and cannot represent inference throughput.

Local tests (143) and Clippy passed; Linux passed 154 library and 17 validator
tests, then its CUDA-only Clippy path rejected placing `compare_layer` after the
test module. Moving that helper before the tests fixes source ordering without
changing behavior. Preserve the initial Linux Clippy log alongside the rerun.

## Carrack qualification

Source `44a114486be73bbab325b2f6ebb0685f738ea222` passes the fresh normal
trial on RTX5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, SM120,
driver 615.71.09/API 13040, with Rust 1.98.1/LLVM 22.1.8. Release xtask SHA-256
is `32fdb3df9190b164b0a55a0d6e9dcf0da804302df54b90a04aa7cec04209615b`.
The unchanged PTX SHA-256 is
`e75136f4874be3765180d3ab3c2dd1c6fd70cb8415a87f7a2fa8e77b5febcb8f`.
Linux tests (154 library and 17 validator), corrected Clippy, Mac tests (143),
Mac Clippy, no-console and formatting checks pass. Initial Linux Clippy failure
and successful rerun are both retained.

Both original token batches and every component fixture pass. Independent
whole-layer results are:

| Tokens | Boundary | Aggregate normalized L2 | Worst partition L2 | Minimum partition cosine | Bit differences |
| --- | --- | --- | --- | --- | --- |
| 1 | Final hidden, 5,120 elements | 0 | 0 | 0.9999999999999999 | 0 BF16 |
| 1 | History, 30,720 elements | 0.0000236932 | 0.0000236932 | 0.9999999997193 | 3 BF16 |
| 1 | State, 786,432 elements | 0.000000408646 | 0.00000552874 | 0.9999999999847 | 86,774 FP32 |
| 17 | Final hidden, 87,040 elements | 0.0018555911 | 0.0068821543 | 0.9999763227 | 9,127 BF16 |
| 17 | History, 30,720 elements | 0.0000700366 | 0.0000984175 | 0.9999999951593 | 7 BF16 |
| 17 | State, 786,432 elements | 0.0001917382 | 0.0040002992 | 0.9999987312716 | 578,442 FP32 |

Partitions are one hidden vector per token, three history rows and 48 recurrent
heads. Every partition passes the original 0.01 L2 / 0.9999 cosine gates. The
17-token aggregate hidden cosine is 0.9999982784; its maximum absolute error is
0.00732421875. These are bounded one-layer differences under the chosen quantized
arithmetic profile, not exact end-to-end parity or proof of 64-layer quality.
No tolerance was changed after observing the trial.

The trial ran in a user scope capped at 8 GiB RAM, zero swap and 240 seconds.
Elapsed 65.660093602 seconds includes artifact verification, upload, GPU work,
component checks and independent CPU execution. It is not prefill/decode speed.
Driver free memory before and after temporary allocations was 32,221,822,976
bytes; this does not measure peak full-model memory. Model prefill, decode and
usable context fields remain null.

Ninfer restarted at 03:43:16 EDT on 2026-09-27, PID 2999518, and reported engine
ready at 03:43:23. A fresh `/health` returned HTTP 200. Its sampled GPU allocation
is 30,046 MiB. ComfyUI PID 448118 remained present at 498 MiB. The original Carrack
branch remains preserved; the experiment checkout follows the worktree branch.

Evidence is committed under `KNOWLEDGE/evidence/qwen-layer-reference-20260927/`;
raw files and PTX remain under `target/specialize/qwen-layer-reference-20260927/`
on both hosts. The next implementation gate is full attention, followed by the
reusable model execution/schedule, full-model/logit qualification and serving ABI.
