# Specialized runtime knowledge

Start with [the implementation plan](../PLAN.md) and the
[feasibility assessment](../../../docs/design/assessments/issue-1393/README.md).

| Entry | Status |
| --- | --- |
| [Performance iteration](optimizations/decode-projections.md) | Matched medians: 20.07 short decode, 18.15 after128 inputs, 136.83 prefill tokens/s; exact checks, sanitizers and memory-release checks pass |
| [Ninfer source comparison](findings/ninfer-performance-comparison.md) | Pinned shape dispatch, GEMV, tiling, fusion and graph/workspace differences; parity remains open |
| [Full-decode kernel profile](findings/model-profile.md) | Control/profile logits and state exactly agree; FP8 projections account for 83.39% of short-prefix event time |
| [First model timing](findings/model-timing-20260927.md) | Raw-token 128-token prefill 19.2 tokens/s, decode 1.55 tokens/s; 135 positions exercised, no matched Ninfer comparison |
| [Resident full model](findings/resident-model.md) | One/two-token hidden/logit fixtures bit exact; two-token state equivalence and all three sanitizers pass; wider quality/context pending |
| [Resident decoder connection](findings/resident-decoder.md) | Resident GDN and attention blocks, whole/chunk/token state equivalence and sanitizers pass; full decoder pending |
| [Resident FP8 MLP](findings/resident-fp8-mlp.md) | Layers 56/63 one/17-token scalar comparisons and three sanitizers pass; reference-free device execution |
| [Persistent residency](findings/persistent-residency.md) | All 1,620 text weight hashes, 131K-capacity state allocation, entry operation and sanitizers pass; full model pending |
| [Initial constraints](findings/initial-constraints.md) | Original assessment; execution evidence now in probe entry |
| [Assembly inventory](asm-inventory.md) | NVFP4 probe executed and numerically checked |
| [Baseline harness](findings/baseline-harness.md) | 18 focused tests and successful live baseline |
| [Ninfer baseline](findings/ninfer-baseline-20260926.md) | Nine measured requests; tested through 42,837 input tokens |
| [Rust NVFP4 probe](findings/rust-nvfp4-probe.md) | 4,096 exact output matches on RTX5090 |
| [Remaining instruction qualification](findings/instruction-qualification.md) | 29 GPU cases pass; three sanitizer tools clean |
| [Representative kernels](findings/representative-kernels.md) | 14 GPU cases pass; preliminary timings; sanitizer recovery complete |
| [CUDA-library reference](findings/cuda-library-reference.md) | 14 independent cuBLAS cases pass; runtime library has no cuBLAS references |
| [Model identity selection](findings/model-identity-selection.md) | Exact policy and legacy fallback; resident discovery/startup pending |
| [Selected CUDA device admission](findings/selected-device-admission.md) | Occupied/free 5090 and wrong-GPU 3080 trials pass; startup integration pending |
| [Mspec format](findings/mspec-format.md) | Reader/writer and content identity pass macOS/Linux tests and low-descriptor checks |
| [Pinned checkpoint intake](findings/checkpoint-intake.md) | Real Carrack import/readback passes: 1,635 tensors, 22.52 GB; model execution pending |
| [Qwen entry operation](findings/qwen-entry.md) | All 1,635 tensor metadata entries match; 696,320 real-weight embedding/norm values pass on GPU; sanitizers clean |
| [Qwen projections](findings/qwen-projections.md) | Resident FP8 QKV/Z and BF16 A/B pass 296,640 real outputs and all three sanitizers |
| [Causal convolution](findings/causal-convolution.md) | 184,320 real outputs pass; whole/chunk/token state exactly agrees; all three sanitizers clean |
| [GDN normalization and gates](findings/gdn-preparation.md) | 73,728 real Q/K and 2,592 gate values pass; all three sanitizers clean |
| [GDN recurrence](findings/gdn-recurrence.md) | 110,592 real outputs and recurrent state match scalar exactly; partition equivalence and sanitizers pass |
| [GDN output](findings/gdn-output.md) | 110,592 gated-norm and 92,160 output-projection values pass component bounds; sanitizers clean |
| [Post-attention operations](findings/post-attention.md) | 92,160 residual/norm values and both MLP input quantizations pass; sanitizers clean |
| [Layer-zero MLP](findings/qwen-mlp.md) | 718,848 matrix outputs, SiLU product and final residual pass component checks; sanitizers clean |
| [Full attention layer](findings/full-attention-layer.md) | One/17-token complete block and independent layer/state comparisons pass fixed budgets; sanitizers clean; full model pending |
| [Causal attention and KV](findings/causal-attention.md) | 110,592 real outputs pass; whole/chunk/token outputs and KV bits agree; sanitizers clean |
| [Full-attention preparation](findings/attention-preparation.md) | 258,048 layer-3 projection and 129,024 prepared Q/K values pass; all sanitizers clean |
| [Whole GDN layer reference](findings/whole-gdn-layer.md) | Independent real-weight CPU layer comparison passes all token/history/head budgets; full model remains pending |
| [Prebuilt core target mismatch](dead-ends/prebuilt-core-target-mismatch.md) | Resolved by rebuilding core; emitted PTX executed |
| [Racecheck timing repetitions](dead-ends/racecheck-timing-repetitions.md) | Failed with host OOM; bounded check-only recovery passed |

New entries belong in `findings/`, `pitfalls/`, `optimizations/`, `dead-ends/`,
or `ah-ha/` according to their subject. Each entry records status, exact model and
recipe, GPU architecture, driver/toolchain, clocks, commit, reproduction command,
expected/observed result, evidence location, and a durable rule. Use `not measured`
or `not applicable` explicitly when appropriate. Optimization entries need measured
before/after results. Superseded entries retain a link to their replacement.
