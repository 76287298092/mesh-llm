# Resident FP8 MLP execution

Status: implementation in progress; no new live qualification yet.

Mac tests (181) and Clippy pass. The first Linux test run passes 209 library and
20 validator tests, but Clippy found an unused import in the Linux-only wrapper.
It is removed before deployment; the failed log is retained. Ninfer remained online.

The last eight decoder layers use FP8 gate/up/down weights, unlike the first
56 NVFP4 MLPs. The execution path now being added borrows validated tensor views
from the full resident weight arena. It performs GPU input quantization, refined
FP8 matrix products, BF16 SiLU/product and output projection. It has no scalar
reference calls, host readbacks, or weight uploads. Temporary allocations and
explicit synchronization remain in this first implementation; it is not tuned.

A separate trial compares layers 56 and 63 with one and 17 rows of deterministic
synthetic normalized hidden input. All intermediate reference values come from
original inputs/weights through the independent scalar chain. GPU readbacks never
replace inputs. Gate, up, activation and down each use the unchanged aggregate and
per-token 1% normalized-L2 / 0.9999 cosine bounds, with exact BF16 rounding checked
against each matrix's unrounded output and bit differences retained as diagnostics.

This adds no new PTX. It reuses the qualified refined FP8 matrix entrypoint and
SiLU-product kernel. Source metadata, shape, byte extents and vocabulary-independent
input bounds are checked before exposing resident addresses. Full-model scheduling,
logits, usable inference context and prefill/decode comparisons remain pending.

Reproduction uses `xtask specialize qwen-fp8-mlp-check` with the same ordered
`--artifact`, `--ptx`, `--device`, `--output` arguments as other Qwen trials. Build
through Just. Exact revision and evidence will be recorded after qualification.
