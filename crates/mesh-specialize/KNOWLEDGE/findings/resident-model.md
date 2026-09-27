# Resident full-model connection

Status: one-token 64-layer hidden/logit comparison is bit exact after the SiLU
correction. Multi-token qualification and model performance remain open.

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

### Corrected one-token run

Source `b800646d9ff122b7fb367143c5dce37f4fcefa5f`, release xtask SHA256
`bd0060d1d06fb778c7e610373e0bdb73c188c63abc6ee856ba8a013ba4206454`, PTX SHA256
`7e7d363e34779f0efdd85dade38437362f3c219842e187c0773695cf68522a3d`.
The unchanged CPU reference SHA256 is
`2ad8afddf4afdb9959f8eac45ac10956ddf4b3df2d88f13745c0b38d22d4a809`.

All 65,280 finite BF16 inputs now produce exactly the reference FP32 SiLU result
on the RTX5090. All 64 model layers and 248,320 final BF16 logits match the saved
independent reference bit-for-bit for `[248044]`. Greedy output is 271, repeated
session logits/state are exact, cursor commit and failed-session rejection pass,
and device memory returns to the pre-weight baseline after release. The trial
took 21.841696232 seconds including load/checks/repeats, not an inference timing.

macOS passes 205 tests and Clippy; Linux passes 260 library and 20 validator tests,
Clippy and the release build. Rust PTX and offline assembly succeed. Probe,
MLP-SiLU, GDN-gated-norm and convolution use 28/28/32/34 registers with no stack
frame or spills; only gated norm uses shared memory, 1,024 bytes. This is resource
evidence, not performance evidence. Ninfer resumed at 06:49:04 EDT, PID 3133806,
HTTP 200; ComfyUI PID 448118 remained at 498 MiB.

The next reference appends the independently selected token: `[248044,271]`.
It will test actual recurrent/KV history across calls, not just repeated empty
sessions. One token does not establish multi-token quality, usable context or
throughput. All failed attempts remain preserved.

### Two-token qualification and BF16 gate projection

The independent `[248044,271]` reference finished in 197.56316971 seconds while
Ninfer stayed online. The initial SiLU-corrected GPU run is exact through layer 11;
layer 12 differs in three of 10,240 hidden values, all in the second token. By the
last layer, logits fail the unchanged budget: relative L2 0.11124162377990778,
cosine 0.9953314708288528. Greedy selection still agrees at 271. GPU full-batch
and token-by-token logits plus all recurrent/KV state are bit exact. Sanitizers
were not run because the normal comparison failed. Ninfer resumed with HTTP 200.

An independent layer-12 replay identifies BF16 A projection output 88 as the first
different boundary: device 0.8828125, CPU 0.88671875. Normalization, QKV, Z, B,
convolution and all MLP boundaries match. The gate error propagates into recurrent
output and produces the three hidden differences. This is not a cursor or cache
partition issue. The diagnostic restored Ninfer at 06:54:52 EDT, PID 3137439,
ComfyUI unchanged.

The next correction retains BF16 tensor-core tiles but starts each K16 tile from
zero and accumulates its result in FP64. Positive-product tiles estimate an error
interval; BF16-ambiguous outputs use an ordered scalar FP64 dot from the original
BF16 values. The interval is deliberately conservative empirical protection, not
a formal floating-point proof. Tail and cancellation fixtures plus unchanged
model references qualify the change. Performance cost remains unmeasured.

### BF16 correction and next attention boundary

Source `a6003edcbf7a96b331dc6a500188afc972c99fea`, PTX SHA256
`4a5b55a1d4a29325b1b357f4f822e815caab9f9f1648c05b4632d898ba40ad77`,
passes Linux tests, Clippy, release build and offline assembly. All three BF16
projection fixtures, including cancellation and partial tiles, match independent
FP64 dots exactly in FP32 and BF16. The kernel uses 45 registers, no local or
shared memory. The exhaustive SiLU probe remains exact.

The unchanged two-token reference now matches hidden outputs through layer 46
bit-for-bit. The first difference is attention layer 47 (17/10,240 BF16 values);
layer 49 first exceeds the fixed budget. Final-logit normalized L2 is
0.06517841049660603 and cosine 0.9979641448221243, still failing. Greedy selection
and full-batch/token-by-token logits and state agree exactly. Harness duration
22.929358517 seconds is not throughput. The suite stops before sanitizers on this
numerical failure. Ninfer resumes at 07:05:07 EDT, PID 3145777, HTTP 200; ComfyUI
PID 448118 remains unchanged. Raw evidence: `two-bf16-suite/`.

The next diagnostic extends the existing read-only boundary observer to attention
blocks. It replays a selected layer from the saved independent previous-layer
hidden values and records Q/K/V projection, prepared Q/K/gate, attention, output
and MLP boundaries. Saved reference structure remains compatible; no arithmetic,
tolerance or normal execution readback changes are made.

### Attention output rounding

Diagnostic source `e8b63b8566107b9608153e5bf11c43e8164b120c` passes 206 macOS
and 261 Linux library tests, 20 Linux validator tests, Clippy and the release
build. Layer 47's normalization, Q/K/V projections, prepared Q/K and query gate
are exact. The attention output first differs in three BF16 values (indices
10291, 10318 and 11622), which propagate through gating/projection/residual into
17 final hidden differences. The MLP branch remains exact for this fixture.
Raw evidence: `layer47-diagnostic/`, duration 20.297814654 seconds. Ninfer resumes
at 07:09:37 EDT, PID 3152310, HTTP 200, ComfyUI unchanged.

The correction under test widens dot-product reduction and online softmax/value
accumulation to FP64, uses a Rust range-reduced exponential polynomial, and rounds
once to FP32 before BF16. It retains the same causal cache traversal and kernel
ABI. This is an untuned correctness baseline; FP64 cost on this consumer GPU must
be measured, and subsequent optimization must retain independent numerical checks.

### Bounded model timing harness

`qwen-model-bench` measures the connected 64-layer model without layer observers,
CPU reference execution or state hashing. It accepts 1..128 raw prompt tokens,
2..16 fixed-length greedy output tokens and 1..3 fresh sessions. Weight loading,
JIT warmup and session allocation precede timing. Prefill includes final logits and
the first greedy token; decode counts the remaining output-token intervals. The
harness records generated IDs, wall-clock intervals, state capacity, weight/state
allocation and memory samples after completed forwards. Those samples cannot
establish transient peak allocation. Fixed-length raw-token execution does not
certify tokenizer/chat quality or a matched Ninfer workload. Numerical correctness
and timing results are reported separately; an executed benchmark is not a parity
claim. No device instruction sites are introduced by the timing harness.
