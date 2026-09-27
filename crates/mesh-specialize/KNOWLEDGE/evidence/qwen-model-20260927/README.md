# Connected model experiment, 2026-09-27

Compact evidence for [resident-model findings](../../findings/resident-model.md).
Reports preserve failed trials as well as successful checks. Log copies normalize
trailing blank lines and omit authentication-related service-journal lines; JSON
reports remain unchanged. Full reference arrays
and PTX remain in `target/specialize/qwen-model-20260927/` in the isolated local
worktree and Carrack checkout. `reference-manifest.json` records reference hashes,
extents, identity and locations; each GPU trial records binary/PTX/reference hashes,
GPU state, and Ninfer restoration evidence. Later scripts also record source HEAD.

| Directory | Source revision | Result |
| --- | --- | --- |
| `one-normal` | `10c5e95b05ba85dae3f361015113a1a8d7d3f3ee` | Wrong old PTX rejected before model execution |
| `one-normal-corrected-ptx` | same | All layers execute; first divergence at layer 2 |
| `layer2-diagnostic` | `f50e7189a5c86dfcde011880593803525dbb65b2` | Isolated first difference in convolution SiLU |
| `one-silu-normal` | `b800646d9ff122b7fb367143c5dce37f4fcefa5f` | One-token hidden/logits bit exact |
| `two-silu-suite` | same | Two-token first divergence at layer 12; stopped before sanitizers |
| `layer12-diagnostic` | same | Isolated first difference in BF16 A projection |
| `two-bf16-suite` | `a6003edcbf7a96b331dc6a500188afc972c99fea` | Two-token first divergence moved to layer 47; stopped before sanitizers |
| `layer47-diagnostic` | `e8b63b8566107b9608153e5bf11c43e8164b120c` | Isolated first difference in attention output |
| `two-attention-wide-suite` | `d99306f87cee4f669590e6b2ae259c96d1f150ab` | All hidden/logits/state exact; memcheck, racecheck and synccheck clean |
| `bench-two-eight` | same | Two-input fixed-eight-output timing; 9 processed positions |
| `bench-128-eight` | same | 128-input fixed-eight-output timing; 135 processed positions |

Correctness harness elapsed times include loading and diagnostic work; they are
not prefill/decode rates. Isolated diagnostics use independent CPU hidden inputs
and are not full-model execution. Allocation capacity is not validated context.
Numerical checks use an independent quantized arithmetic contract, not upstream
BF16 quality evidence. Ninfer's measured deployment uses MTP4 and FP8 KV; this
prototype has neither. No matched performance comparison is implied.
