# Specialized runtime knowledge

Start with [the implementation plan](../PLAN.md) and the
[feasibility assessment](../../../docs/design/assessments/issue-1393/README.md).

| Entry | Status |
| --- | --- |
| [Initial constraints](findings/initial-constraints.md) | Original assessment; execution evidence now in probe entry |
| [Assembly inventory](asm-inventory.md) | NVFP4 probe executed and numerically checked |
| [Baseline harness](findings/baseline-harness.md) | 18 focused tests and successful live baseline |
| [Ninfer baseline](findings/ninfer-baseline-20260926.md) | Nine measured requests; tested through 42,837 input tokens |
| [Rust NVFP4 probe](findings/rust-nvfp4-probe.md) | 4,096 exact output matches on RTX5090 |
| [Prebuilt core target mismatch](dead-ends/prebuilt-core-target-mismatch.md) | Resolved by rebuilding core; emitted PTX executed |

New entries belong in `findings/`, `pitfalls/`, `optimizations/`, `dead-ends/`,
or `ah-ha/` according to their subject. Each entry records status, exact model and
recipe, GPU architecture, driver/toolchain, clocks, commit, reproduction command,
expected/observed result, evidence location, and a durable rule. Use `not measured`
or `not applicable` explicitly when appropriate. Optimization entries need measured
before/after results. Superseded entries retain a link to their replacement.
