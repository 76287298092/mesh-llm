# Specialized runtime knowledge

Start with [the implementation plan](../PLAN.md) and the
[feasibility assessment](../../../docs/design/assessments/issue-1393/README.md).

| Entry | Status |
| --- | --- |
| [Initial constraints](findings/initial-constraints.md) | Source-backed; GPU execution unproven |
| [Assembly inventory](asm-inventory.md) | NVFP4 probe emitted as PTX; GPU untested |
| [Baseline harness](findings/baseline-harness.md) | 18 host tests and Clippy pass; live baseline pending |
| [Ninfer baseline](findings/ninfer-baseline-20260926.md) | Nine measured requests; tested through 42,837 input tokens |
| [Rust NVFP4 probe](findings/rust-nvfp4-probe.md) | Host arithmetic/packing tests pass; CUDA execution pending |
| [Prebuilt core target mismatch](dead-ends/prebuilt-core-target-mismatch.md) | Compiler failure; recipe correction under validation |

New entries belong in `findings/`, `pitfalls/`, `optimizations/`, `dead-ends/`,
or `ah-ha/` according to their subject. Each entry records status, exact model and
recipe, GPU architecture, driver/toolchain, clocks, commit, reproduction command,
expected/observed result, evidence location, and a durable rule. Use `not measured`
or `not applicable` explicitly when appropriate. Optimization entries need measured
before/after results. Superseded entries retain a link to their replacement.
