# Resident full-model connection

Status: first 64-layer GPU execution completed, independent numerical comparison
failed. No full-model correctness or performance claim.

The decoder follows the compiled 64-layer schedule with persistent checkpoint
weights and one session state arena. A transaction advances the sequence cursor
only after all layers, the final normalization, vocabulary projection and greedy
selection succeed. A partially failed step poisons the session and rejects reuse.
The LM head selects the last hidden row on device. Intermediate hidden rows remain
on device except for optional read-only observers in the correctness trial.

An independent CPU composition loads the original quantized checkpoint one layer
at a time and saves every layer's BF16 output and the final logits. It runs while
Ninfer remains online. GPU qualification consumes this saved reference, verifies
its model/weights identity and extents, compares every layer and final logits with
the existing fixed 1% relative L2 and 0.9999 cosine budgets, checks greedy choice,
and compares full-batch against token-by-token logits and complete state hashes.
An injected failure checks rejection of partially updated sessions. Reference
files and trial reports are created without replacing existing evidence.

Commands, after `just specialize-tools-build`:

```sh
target/release/xtask specialize qwen-model-reference --artifact ARTIFACT --tokens 248044 --output REFERENCE
target/release/xtask specialize qwen-model-check --artifact ARTIFACT --reference REFERENCE --ptx target/specialize/probes.ptx --device 0 --output REPORT
```

CPU references are bounded to 1..17 tokens. This comparison uses our explicit
quantized arithmetic contract. It is not upstream BF16 quality validation.
The trial includes layer readbacks, state hashing and repeated independent
sessions. Its elapsed time is not model prefill or decode throughput. No device
kernel or PTX instruction site changes in this stage. Current performance, peak
memory, usable long context, tokenizer/chat serving, MTP and ABI integration
remain unmeasured or incomplete.

Exact source revisions and live results will be recorded after the first trial.

## First execution and diagnostic follow-up

Source `10c5e95b05ba85dae3f361015113a1a8d7d3f3ee` passed 199 macOS library
tests, 254 Linux library tests, 20 Linux validator tests, Clippy on both hosts,
the console-output policy and release xtask builds. The one-token CPU reference
for `[248044]` completed all 64 layers in 111.789145637 seconds, with Ninfer online.
It selected token 271. The first GPU attempt accidentally selected an old root
PTX probe containing only `probe_nvfp4_mma`; module lookup rejected it before model
execution. The retry script pins the previously qualified PTX SHA256
`ee13b6bf3d34ee2ccceeaf4b9420c32ddc0fee2c97fc7f7d529f86ba06c306b9`.
Both attempts and restoration records remain under
`target/specialize/qwen-model-20260927/` on Carrack.

The corrected GPU trial executes every layer and chooses token 271, with exact
repeat-session logits/state. Its final-logit relative L2 is 0.13722781637368325
and cosine is 0.9914219092625522, failing the unchanged numerical budget. The first
nonexact hidden output is layer 2: 13 of 5,120 BF16 values differ, relative L2
0.00007030544160344343. Fifty-seven layers eventually exceed the budget. Harness
duration is 21.595276324 seconds including weight loading, comparisons and repeated
sessions; this is not a throughput measurement. Ninfer was restored, HTTP 200,
PID 3117119 at 06:36:35 EDT; ComfyUI PID 448118 stayed unchanged at 498 MiB.

The follow-up adds optional read-only BF16 observers to the independent CPU MLP
and GDN compositions and the resident GDN wrapper. Normal execution passes no
observer and adds no host readback. `qwen-model-trace` enriches the saved reference
with one GDN layer's independent intermediate results and checks the recomputed
hidden output against the existing reference. A diagnostic `qwen-model-check`
replays only that layer from CPU hidden input, comparing operation boundaries to
locate the first cause. It is explicitly marked as isolated layer execution.
No tolerance or device arithmetic has changed.

### First cause: convolution SiLU rounding

Diagnostic source `f50e7189a5c86dfcde011880593803525dbb65b2` passed 257 Linux
library tests, 20 validator tests and Clippy. Layer 2's input normalization,
QKV/Z/A/B projections are bit exact. The first difference is the convolution
activation: ten outputs round to `0xbaff` instead of reference `0xbb00`.
The same run has eight gated-output differences and thirteen final hidden
differences; MLP gate/up/activation/down happen to be exact for this input.
This isolates an operation mismatch without propagated input error. The trial
restored Ninfer at 06:41:50 EDT, PID 3120992, HTTP 200, ComfyUI unchanged.

The old SiLU uses approximate FP32 exp2. At input -1/256, independently evaluated
SiLU is approximately -0.001949310307585006. Rounding it to FP32 lands exactly on
the BF16 midpoint -0.001949310302734375, which rounds to `0xbb00` by ties-to-even.
Small FP32 exponential error can place it on the other side of that midpoint.

The proposed correction shares one pure Rust SiLU calculation across convolution,
MLP activation and gated normalization. It evaluates the stable sigmoid in FP64
using ln(2) range reduction and a degree-16 Taylor polynomial, then converts once
to FP32. No host table or reference function enters device execution. An exhaustive
host test and device probe compare every finite BF16 input with the independent
libm-backed reference. The model trial must pass this probe before loading weights.
Device validation and the unchanged full-model comparison remain pending.
