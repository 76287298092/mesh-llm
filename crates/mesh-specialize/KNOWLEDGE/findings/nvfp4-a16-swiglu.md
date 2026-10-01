# NVFP4 A16 SwiGLU

Status: candidate kernel, independent reference, operator-check commands, and
opt-in legacy decode dispatch are implemented in the current worktree. No Cargo
build, PTX regeneration, GPU operator check, sanitizer, whole-model comparison,
quality score, or timing run was performed for this change. The profile remains
experimental and disabled by default.

## Scope and arithmetic

The explicit `MESH_SPECIALIZE_NVFP4_MLP_SCHEDULE=a16-swiglu` setting selects a
fused gate/up projection only in the legacy resident model when an NVFP4 MLP is
executed with exactly one row and `past > 0`. This admits decode rows, not the
one-row prompt/prefill at `past == 0`. Multirow calls, FP8 MLPs, and the baseline
profile retain their existing path. The fused gate/up output feeds the existing
NVFP4 down projection and residual path.

The device kernel reads BF16 activations directly, computes separate gate and up
FP32 FMA reductions using each projection's packed E2M1 weights, E4M3 K16 scales,
and reciprocal weight global scale, then writes BF16 diagnostics and one
FP32-SiLU/raw-product BF16 activation. It does not use the NVFP4 activation
quantizer for gate/up. The baseline instead quantizes activations to A4 and
rounds projection and SiLU intermediates through BF16. This is a separate
arithmetic profile, not an exact schedule replacement.

The reference at `reference/nvfp4_swiglu_a16.rs` decodes logical weight nibbles
and scales independently and accumulates each dot in FP64. It computes SiLU and
the product in FP64, then rounds the product to FP32 and BF16. Device dots and
the product therefore use a different reduction/activation order. Operator
limits in `src/kernels/cuda/nvfp4_swiglu_a16_operator/compare.rs` are initial
screening checks, not established model-quality thresholds.

## Current integration and exclusions

- Ordinary legacy model execution uses the profile stored in
  `src/kernels/nvfp4_mlp_schedule.rs`. The profile parser defaults to `baseline`.
- The one-token fused branch is wired through `resident_gdn::Layer` and
  `resident_attention::Layer` into `resident_mlp::Mlp::run_with_past`.
- The stream/graph executor remains on its original planner and kernel sequence;
  it rejects `a16-swiglu` rather than using the legacy-only candidate.
- MTP/speculative recovery, the MLP workspace, NVFP4 projection audit, exact
  whole-model comparison, isolated layer control diagnostics, and model
  component-control trials reject this profile until they have candidate-aware
  contracts. Teacher-forced scoring admits it only with the explicit
  `MESH_SPECIALIZE_SCORE_MODE=decode` route, which prefills each window's history
  and uses sequential single-token decode for the remaining scored rows. This
  enables candidate-aware score collection; it does not qualify model quality.
- Model profile, model benchmark, and score reports include the configured MLP
  profile. Run one-token-prefix model profiling in separate baseline and
  candidate processes. The replay result is same-profile only, not a
  cross-profile arithmetic or quality check. A one-token prefix's prefill stays
  baseline; the profiled next-token decode exercises the candidate.

## Parent qualification plan

Run these serially on the parent-selected RTX 5090/SM120 environment after
reviewing the candidate code and the updated source tree. Use new report paths,
the exact intended artifact, and preserve every failing report.

1. Build PTX and the checker with `just specialize-ptx` and
   `just specialize-tools-build`.
2. Run the synthetic operator cases:
   `target/release/xtask specialize nvfp4-a16-swiglu-check --ptx target/specialize/probes.ptx --device 0 --output /tmp/nvfp4-a16-swiglu-synthetic.json`.
3. Run the real layer-zero gate/up check:
   `target/release/xtask specialize nvfp4-a16-swiglu-real-check --artifact ARTIFACT --ptx target/specialize/probes.ptx --device 0 --output /tmp/nvfp4-a16-swiglu-real.json`.
4. Repeat the synthetic and real checks under CUDA memcheck, racecheck, and
   synccheck, each with a fresh output path and
   `/opt/cuda/bin/compute-sanitizer --tool TOOL --error-exitcode 42`.
5. Run the same verified artifact and one-token prefix in separate baseline and
   candidate processes using `xtask specialize qwen-model-profile`. Its replay
   comparison is same-profile only, not an independent arithmetic oracle or a
   cross-profile quality check.
6. Use teacher-forced scoring with `MESH_SPECIALIZE_SCORE_MODE=decode` for a
   candidate-aware fixed-token collection, and full-logit hashes for determinism
   checks. The scorer's operator/record checks do not produce a model-quality
   verdict. Apply the frozen gates in `findings/quality-gates.md` with an exact
   baseline collection, then run the fixed objective task set and matched
   baseline/candidate model checks. `qwen-model-check` remains the exact-profile
   control and intentionally rejects this experimental schedule. Reject
   promotion on any unexplained regression.
7. Benchmark the ordinary legacy profile against baseline with repeated paired
   requests, then record the actual model throughput. Operator checks and
   kernel-event durations alone do not establish model speedup.

No candidate quality result, baseline-versus-candidate model result, sanitizer
result, resource count, or performance measurement is currently available.
