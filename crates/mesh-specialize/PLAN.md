# Specialized Qwen runtime implementation plan

Status: connected 64-layer GPU decoder and dedicated decode optimization qualified.
The retained two-token hidden/logit/state reference matches exactly and all three
CUDA sanitizers pass. Current medians are 25.26 short-prefix decode and 22.37
after 128 inputs, up from the original 1.55 tokens/s. Whole/token partition checks
also pass at 512 input tokens. Prefill now reaches 259.38 tokens/s at 128 inputs and 296.58
at 512 with exact tensor-core tiles and all sanitizer checks passing. See [dedicated decode](KNOWLEDGE/optimizations/dedicated-decode.md)
and [larger prefill](KNOWLEDGE/optimizations/larger-prefill.md).
Text quality, broader context qualification and matched Ninfer performance remain
open. These raw-token trials do not establish performance parity or serving readiness.

Host policy now checks exact model/weights and explicit selected-device admission.
The [live device-policy trial](KNOWLEDGE/findings/selected-device-admission.md)
passes. The standalone model experiment executes all 64 layers. Resident artifact
discovery, ABI loading, tokenization and serving integration remain open.

The internal `.mspec` container now has a content-derived identity, bounded
resident reader and CPU object assembler. Pinned Safetensors import and full
readback now pass on Carrack for 1,635 retained tensors. The compiled tensor
inventory matches, and real-weight embedding/input normalization plus FP8 QKV/Z
and BF16 A/B projections pass on GPU with all three sanitizers. QKV now feeds
causal convolution/SiLU with exact whole-sequence/chunk/token state equivalence.
Resident Q/K normalization and beta/log-decay/decay gates now pass independent
real-weight comparisons and all three sanitizers. The recurrent matrix update
now matches its independent scalar contract exactly, including chunk/state
equivalence. Gated normalization and FP8 output projection now complete the
resident layer-zero attention component chain. Post-attention norm, NVFP4 MLP
projections, SiLU product and the second residual now complete the layer-zero
GPU component chain and pass sanitizers. Independent whole-layer comparison now
passes fixed aggregate and per-token/history/head error budgets for one and 17
tokens. Layer-3 full-attention Q/K/V projections, Q/gate split, per-head Q/K
normalization and partial text RoPE also pass real-weight checks and sanitizers.
Causal attention and persistent BF16 KV now pass real-weight numerical checks,
whole/chunk/token equivalence and sanitizers. Layer-3 sigmoid gating, output
projection, residual/norm and MLP now pass independent whole-layer hidden/KV
comparisons and all sanitizers for one and 17 tokens. FP8 rounding refinement was
required to meet the unchanged per-token error budget; its performance cost is
unmeasured. All 1,620 text tensors now remain resident together in one verified
device arena, with the 64-layer schedule's state allocated at capacity 131,072.
Full readback hashes, zeroed state, resident entry operation and three sanitizers
pass. This is allocation capacity, not tested inference context. The connected
decoder now executes all layers; wider logit qualification and performance tuning
remain open. The final-eight-layer
FP8 MLP execution path now uses persistent weights without host intermediate
readbacks or scalar reference work in the execution path. Layers 56 and 63 pass
independent one/17-token branch comparisons and all three sanitizers.

## Ordered optimization continuation

The user requested continuing one area at a time, including dedicated decode,
larger prefill and MTP. Keep the current arithmetic profile stable while improving
its schedule; speculative target verification must agree with ordinary decode.

1. Dedicated NVFP4 decode: first improvement complete. Exact grouped dots reach
   25.26 short / 22.37 after 128 inputs; independent checks and sanitizers pass.
   See [dedicated decode evidence](KNOWLEDGE/optimizations/dedicated-decode.md).
2. Larger prefill: first improvement complete. Exact 16x8 FP8 tensor-core tiles
   reach 259.38 / 296.58 input tokens/s at 128 / 512, with independent arithmetic,
   whole/token partition and sanitizer evidence. Further tiling remains a measured follow-up.
3. Resident MTP head, draft state, target verification and rollback. Prove accepted
   output agrees with target-only greedy decoding, exercise rejection and rollback,
   then measure accepted tokens per forward and end-to-end throughput.
4. Re-profile and address measured submission/workspace costs in bounded changes.

Each area closes only after host checks, independent numerical checks, relevant
GPU sanitizers, repeated before/after model timings, memory/state validation and
restoration of Ninfer. MTP remains outstanding until a real draft/verify run passes;
retaining its tensors or adding an interface is not completion.

## Objective and boundaries

Build the first stages of issue 1393 in `codex/issue-1393-feasibility`, with bounded
GPT-6-Luna max workers implementing parent-designed pieces. Establish an actual
NInfer baseline before stopping the service for Rust GPU trials. Push this branch,
switch the clean carrack `~/dev/mesh/mesh-llm` checkout without losing its existing
branch, and run the reviewed code there. Preserve raw evidence for prefill, decode,
memory, and context. A microkernel trial cannot close the full-model comparison.

The user authorized starting/stopping ninfer for these trials and the branch
push/pull. Preserve unrelated processes and work. Record exact revisions and
service state before each handoff. Do not publish prototype runtime packages.

## Parent-owned decisions

1. Use a Rust host library `mesh-specialize`, isolated from llama.cpp and stable
   host builds. Its first executable entry is a typed `xtask specialize` command.
   GPU kernels are separate Rust source files compiled to PTX by a pinned nightly
   through a Just recipe. No build script silently installs tools or compiles CUDA.
2. T0 owns CUDA driver loading, allocations, streams/events, launch arguments,
   device kernels, and tuning. Above T0, use owned shape/dtype/tensor descriptions
   and Rust errors. T1 owns model-independent execution/session mechanics; the
   Qwen package owns its fixed schedule. T2 is a later Skippy ABI adapter.
3. Preserve an independent host arithmetic implementation for every tuned op.
   Keep fixture generation independent of packed GPU layout. Host-only tests must
   run without libcuda. No NInfer source, binary, or weight-container import.
4. Initial integration selects one engine before loading the global ABI. Exact
   resident-artifact/device mismatch falls back to llama.cpp at startup. This does
   not claim arbitrary same-process model switching. Production multi-engine
   dispatch remains a separate explicitly tracked requirement.
5. First full-model residency is quantized. Full BF16 reference is CPU/layer-wise
   or on a separate sufficiently large reference machine. For controlled NVFP4
   performance, independently import the same upstream quantized Safetensors
   codes/scales; keep BF16-oracle and quantized-oracle numerical checks distinct.
6. Do not assign broad architectural work to workers. Each task below gets exact
   types/interfaces and acceptance tests in its dispatch prompt. Workers edit only
   owned files; parent owns manifests, command dispatch, recipes, and integration.

## Stages and exit evidence

| Stage | Bounded pieces | Exit evidence |
| --- | --- | --- |
| S0 Baseline | B01 stream observations; B02 HTTP trial runner; parent fixtures/service control | Actual NInfer request results, server timings, TTFT, GPU memory samples, context configuration and tested prompt lengths |
| S1 Rust execution | K01 CUDA driver ownership; K02 independent reference arithmetic; K03 Rust instruction probes; parent build/launch integration | Rust-emitted PTX, successful driver JIT and launches, independent numerical matches, register/resource records |
| S2 Representative kernels | K04 NVFP4 MMA layout; K05 RMSNorm; K06 tiled GEMM; parent workload matrix | Correct nonuniform/signed fixtures, prefill/decode-shaped GPU timings and memory; no model-throughput claims |
| S3 Host ABI | H01 exact identity; H02 driver-only eligibility; H03 `.mspec` discovery; H04 ABI exports; parent startup dispatch | Real serving-path exercise, exact-model/device rejection, old manifests and startup fallback preserved |
| S4 Artifact and model | A01 container; A02 reader; A03 pinned conversion; Q01 embedding/norm; Q02 full attention; Q03 GDN; Q04 fixed schedule; Q05 tokenizer/sampling | Independent per-layer/state/logit evidence and quantized Qwen greedy fixtures |
| S5 Full-model trial | Parent deployment and comparison with fixed baseline corpus | Actual prefill/decode/peak memory/usable context for both engines, quality and limitations recorded |
| S6 Competitive serving | Separate graph replay, KV compression, MTP, rollback and cache tasks | Controlled deployed-profile comparison, including long context and concurrency two |

Stop after S1/S2 for the first carrack checkpoint. Record model prefill/decode as
unavailable if only kernels exist. Resume S3/S4 for the first full-model trial;
do not mark S5 complete on kernel or synthetic ABI evidence. Failure to express
required instructions halts dependent work and requires a design decision.

## First dispatch contracts

### B01: streaming observation parser

Own `tools/xtask/src/specialize/observations.rs` only. Parse SSE `data:` records,
separating role-only chunks from nonempty reasoning/content/tool deltas. Preserve
terminal `usage`, `timings`, finish reason, and raw JSON records. Require `[DONE]`,
usage, finite timing values, and internally consistent token counts. Expose the
parent-specified accumulator and typed result. Test malformed/truncated streams,
role-only first chunks, reasoning-only output, cache accounting, and zero/one-token
timings. No network, GPU access, process launch, Cargo execution, or credential use.

### B02: bounded HTTP trial runner

Own `tools/xtask/src/specialize/baseline.rs` only. Read a versioned JSON plan of
explicit requests and an API-key environment variable name. Use a bounded HTTP
client; stream records through B01; save each result before starting the next.
No retry, process control, remote commands, or arbitrary shell. Limit request count,
request size, output tokens, and per-request timeout. Credentials never enter
records/errors. Persist the request, timing observations, error state, and plan
identity. Parent supplies fixture plan and integrates dependencies/dispatch.

### K01: minimal CUDA driver ownership

Own `src/kernels/cuda/driver.rs` only. Dynamically resolve the parent-listed driver
API symbols. Own context, module, stream, buffers, and events with deterministic
cleanup. Error messages include operation/result code, never silently succeed.
Use selected device and expose compute capability plus memory/resource queries.
No kernel source, high-level scheduler, global symbol singleton, static CUDA link,
package changes, or SSH. Parent checks every FFI signature and lifetime.

### K02: independent arithmetic reference

Own `reference/arithmetic.rs` only, after parent supplies exact input/output types.
Implement scalar FP4 E2M1 decoding, scale decoding, dense dot product, and RMSNorm
with explicit shape/error contracts. Expected values use hand-computed fixtures,
not GPU packing helpers. No CUDA or compiler dependencies.

### K03 and later

Dispatch only after K01/K02 review. Parent specifies exact lane mapping, operand
packing, target instructions, launch dimensions, and expected results. One worker
gets one instruction or operation, not an entire GEMM architecture. Existing good
code stays intact when a later instruction fails.

## Baseline protocol

First capture the installed service profile with its existing MTP4, FP8 KV,
131,072 context ceiling, concurrency capacity two, and 2,048-token prefill chunk.
Run serial, deterministic text requests with explicit greedy settings and bounded
output. Record actual tokenizer prompt counts, rather than equating characters
or repeated words with tokens. Use short, medium, and long prompts, with repeated
samples and a separate warm-cache case. Keep the first run bounded to at most 12
requests, 512 output tokens/request, 180 seconds/request, and an overall budget.

Capture server `prompt_n/cache_n/prompt_ms/predicted_n/predicted_ms` and reported
rates. Decode uses `predicted_n - 1` intervals. Client TTFT begins at request send
and ends at the first nonempty visible reasoning/content/tool delta. These are
different metrics. Record configured context, actual tested context, sampled GPU
memory, startup allocations, GPU UUID, driver, clocks/power, binary hash, artifact
identity/provenance limits, and competing GPU processes. A 131K configuration is
not proof of successful 131K inference. Sampling is not an allocator-exact peak.

After this deployed-profile baseline, collect a matched non-speculative control
with prefix reuse disabled and explicit memory/context allocation when the first
model is ready. Keep control profile changes reversible and labeled. Never compare
that control's rates directly to MTP4 as an isolated speedup across different input.

## Work ledger

| Requirement | State | Evidence |
| --- | --- | --- |
| New isolated branch and assessment | Complete | `docs/design/assessments/issue-1393/` |
| Plan and knowledge base before engine | Committed | `2d9ea2073` |
| B01/B02 bounded baseline tool | Complete | 18 focused tests and successful live run |
| Fresh NInfer baseline | Complete for deployed serial profile | `KNOWLEDGE/findings/ninfer-baseline-20260926.md` |
| Rust instruction gate | NVFP4 and 29 remaining probe cases pass; sanitizer checks clean | `KNOWLEDGE/findings/instruction-qualification.md` and prior NVFP4 evidence |
| Representative GEMM/RMSNorm | 14 cases / 1,489,305 outputs pass; independent cuBLAS reference passes; profiling pending | `KNOWLEDGE/findings/representative-kernels.md` and `cuda-library-reference.md` |
| Upstream push and carrack branch synchronization | Complete through resident attention trial | `f7be962cc`; original carrack branch retained |
| First carrack Rust GPU trial | Complete | 32 cases, 4,096 exact output matches; Ninfer restored |
| A01/A02 container, identity, reader and object assembly | 65 macOS / 68 Linux library tests pass; 17 Linux validator tests pass; low-descriptor checks pass | `KNOWLEDGE/findings/mspec-format.md` |
| A03 pinned upstream import | Real-file import/readback passes; 82 macOS / 85 Linux library tests and 17 Linux validator tests pass | `KNOWLEDGE/findings/checkpoint-intake.md`; 22.52 GB artifact in 58.6 seconds |
| Compiled model tensor inventory | Complete for pinned raw-v1 checkpoint | All 1,635 tensors match compiled metadata; real artifact checked on Carrack |
| Q01 embedding/norm | Partial: first fused embedding/input norm passes real-weight GPU comparison and all sanitizers | `KNOWLEDGE/findings/qwen-entry.md`; full layer/schedule execution pending |
| Q03 GDN inputs | Partial: FP8 QKV/Z and BF16 A/B pass 296,640 real outputs and all sanitizers | `KNOWLEDGE/findings/qwen-projections.md` |
| Q03 causal convolution | Partial: 184,320 real outputs pass; whole/chunk/token states exactly agree; sanitizers clean | `KNOWLEDGE/findings/causal-convolution.md`; gated norm/output pending; recurrence evidence recorded separately |
| Q03 GDN preparation | Partial: 73,728 normalized Q/K values and 2,592 gate values pass; all three sanitizers clean | `KNOWLEDGE/findings/gdn-preparation.md`; complete layer pending; recurrent evidence recorded separately |
| Q03 GDN recurrence | Partial: 110,592 real outputs and FP32 state match scalar exactly; whole/chunk/token paths and sanitizers pass | `KNOWLEDGE/findings/gdn-recurrence.md`; complete layer pending |
| Q03 GDN output | Partial: 110,592 gated-norm and 92,160 projection values pass component bounds; sanitizers clean | `KNOWLEDGE/findings/gdn-output.md`; layer-zero attention chain connected, full-layer/logit parity pending |
| Q01/Q02 post-attention input | Partial: 92,160 residual/norm values and both MLP input quantizations pass; sanitizers clean | `KNOWLEDGE/findings/post-attention.md`; MLP matrix products and full layer remain pending |
| Q01/Q03 layer-zero MLP | Partial: 718,848 real NVFP4 matrix outputs, SiLU product and second residual pass component checks; sanitizers clean | `KNOWLEDGE/findings/qwen-mlp.md`; whole-layer comparison recorded separately |
| Q03 independent whole-layer reference | Partial: one/17-token layer-zero hidden/history/state pass fixed 1% L2 and 0.9999 cosine budgets, including every partition | `KNOWLEDGE/findings/whole-gdn-layer.md`; full-model/logit parity pending |
| Q02 full-attention preparation | Partial: 258,048 layer-3 projection and 129,024 prepared Q/K values pass; all three sanitizers clean | `KNOWLEDGE/findings/attention-preparation.md`; attention/KV cache and full layer still pending |
| Q02 causal attention and KV | Partial: 110,592 real outputs pass; whole/chunk/token FP32/BF16 outputs and KV states exact; sanitizers clean | `KNOWLEDGE/findings/causal-attention.md`; full attention layer/model pending |
| Q02 complete attention layer | Partial: one/17-token full layer and exact initialized K/V pass fixed aggregate/per-token budgets; sanitizers clean | `KNOWLEDGE/findings/full-attention-layer.md`; full-model scheduling/logits and performance remain pending |
| H03 discovery and H04 ABI | Pending | Standalone experimental decoder executes; host discovery and ABI integration are not implemented |
| Persistent text weights and compiled state layout | Allocation/transfer gate passes | `KNOWLEDGE/findings/persistent-residency.md`; all 1,620 hashes and 128 zeroed regions pass, first resident entry exact, sanitizers clean; decoder execution pending |
| Full model performance/context trial | Partial: raw-token 128-input prefill, seven decode intervals, memory checkpoints and execution through 135 positions measured | `KNOWLEDGE/findings/model-timing-20260927.md`; matched corpus, peak memory, text quality and long context pending |
| Connected 64-layer decoder and final vocabulary head | One/two-token independent hidden/logit fixtures bit exact; whole/token state and all three sanitizers pass | `KNOWLEDGE/findings/resident-model.md`; broader numerical/text quality remains open |
| Reference-free resident decoder connection | GDN layer zero and full-attention layer three qualified; full schedule pending | `KNOWLEDGE/findings/resident-decoder.md`; independent whole-block comparisons, exact whole/chunk/token state and all three sanitizers pass |
| Resident final-eight-layer FP8 MLP | Branch execution qualified for layers 56/63, one/17 tokens | `KNOWLEDGE/findings/resident-fp8-mlp.md`; independent scalar comparisons and three sanitizers pass; full schedule pending |
| Direct `.ninfer` source loading | Ordinary commands run the pinned file; legacy/stream equivalence measured | `KNOWLEDGE/findings/ninfer-import-contract.md`; 106-input 306.41/26.48, 512-input 327.62/18.16 tok/s prefill/decode |
| StreamForward and graph replay | Stream opt-in exact; graph exact, memcheck pass, racecheck incomplete, <1% gain | `KNOWLEDGE/optimizations/stream-forward.md` |
| Split decode attention | Executes; quality verdict FAIL; stays opt-in | `KNOWLEDGE/findings/split-decode-quality.md` |
| Decode scorer | Scorer-only GPU smoke passes; full-corpus quality open | `KNOWLEDGE/findings/decode-scorer-qualification.md` |
| Native MTP residency and Q4 head | Packed parents resident; real 131,072-row Q4 head passes bounded inputs and three sanitizers | `findings/native-mtp-residency.md`, `findings/native-q4-operator-qualification.md` |
| Native MTP Q8 FC and projections | Resident FC and 16 projection cases pass normal and sanitizers on bounded inputs | `findings/native-mtp-q8-fc-resident.md`, `findings/native-mtp-q8-projections.md` |
| Target batch verification | 20 N=1..5 normal/sanitizer cells pass; no throughput claim | `KNOWLEDGE/findings/target-batch-decode-qualification.md` |
| FP8 quantize reuse, NVFP4 A16 SwiGLU | Implemented opt-in; GPU qualification not run | `findings/fp8-quantize-reuse.md`, `findings/nvfp4-a16-swiglu.md` |
| Native MTP admission | Closed: whole native MTP forward, acceptance and quality not qualified | `findings/ninfer-source-faithful-port.md` |

Validation belongs to the parent: serial focused Rust tests/check/Clippy, formatting,
repository no-console and crate-coverage checks where affected, and explicit
remote evidence. Update this ledger after each measured gate. Keep failed attempts.

## First full-model profile

The initial full-model timing is now measured. P01 owns a bounded, scoped CUDA
launch recorder in `launch_profile.rs`; P02 owns one profiled decode replay versus
an unprofiled session in `resident_model_profile.rs`. The parent owns driver
hooks, package/CLI wiring, current-source builds, the protected Carrack service
pause, and the next optimization decision. These workers implement the specified
interfaces without changing kernels or expanding the design. The next change
must be selected from measured full-model attribution rather than assumed from
an isolated arithmetic probe.

## Bounded experiment checkpoint, September 27

The requested initial implementation and Carrack trial are complete. The plan,
bounded worker deliverables, Ninfer baseline, pushed branch and clean remote
fast-forwards are recorded. The Rust prototype executes all 64 layers, with
independent one/two-token hidden/logit checks, state equivalence and three clean
sanitizer runs. Actual model prefill, decode, sampled memory and exercised context
are in `KNOWLEDGE/findings/model-timing-20260927.md`. This closes the initial
experiment; it does not close issue 1393 or the later S3 through S6 exit gates.

The final diagnostic at source `55ee5ae56b21a09661d8b79199c73a9f2f999539`
captures 1,476 launches with exact control/profile state and output equivalence.
FP8 projections account for 83.39% of summed event time. The next optimization
target is decode-sized FP8 projection, with the existing arithmetic as its control.
The report and limits are in `KNOWLEDGE/findings/model-profile.md`.

The initial prototype measured 19.2 tokens/s prefill at 128 synthetic raw tokens
and 1.55 tokens/s decode. Persistent weight/state payload is about 20.31 GiB for
135 positions. Allocation samples are not a peak-memory result. Full-model
numerical evidence covers one/two-token fixtures, and execution through 135
positions does not prove usable long-context quality. Matching Ninfer needs a
shared text corpus, tokenizer/chat support, verified weight identity, a matched
non-speculative control and further optimization. ABI integration, graph replay,
FP8 KV, prefix reuse and concurrent serving remain future work. MTP status is
tracked in the current-state section at the end of this plan.

Ninfer was restored after the final profile at 07:45:48 EDT, PID 3197048 and
HTTP 200. ComfyUI PID 448118 remained unchanged. The original Carrack branch is
retained. Do not treat this checkpoint as a production runtime or parity claim.

## Performance iteration requested September 27

Inspect pinned Ninfer source for missed optimizations, then improve measured
bottlenecks without loosening correctness gates. The first bounded change targets
single-row FP8 projections with exact integer accumulation. Parent owns design,
independent fixtures, integration, builds and protected Carrack trials; the worker
owns one kernel. Read-only workers compare Ninfer projection and execution paths.
Record retained changes against the same short and 128-token workload, with
unchanged weights, outputs and context. Preserve old PTX as a control. Performance
parity still requires matched text and non-speculative controls.

## Dedicated decode, larger prefill and MTP continuation

The qualified exact NVFP4 single-row kernel raises median short decode to 25.26
tokens/s (22.37 after 128 inputs). The exact FP8 prefill tile raises median
prefill to 259.38 tokens/s at 128 inputs and 296.58 at 512 inputs. Independent
arithmetic/model checks, state partition checks and all three CUDA sanitizers
pass; see `KNOWLEDGE/optimizations/dedicated-decode.md` and `larger-prefill.md`.

Resident greedy MTP now executes the checkpoint's 15 BF16 tensors, forks target
state for verification and replays accepted inputs after rejection. Independent
head fixtures, exact whole/token partition and exact target-only output/state
comparison pass, including forced rejection. Text benchmarks show substantial
acceptance-dependent gains and regressions; fixed depth four is not a safe
default. Verification-kernel tuning and sanitizer completion are recorded in
`KNOWLEDGE/optimizations/mtp.md`. This does not implement frontend/ABI serving,
adaptive speculation, sampling or EOS policy, or establish parity with Ninfer.


## Resumed-chat workspace checkpoint, 2026-09-27

The former issue-1393-feasibility checkout was removed by app archive cleanup.
Its snapshot `4308f1fcf4a1a30429a835f55660154811858b3d` preserved all pending
tracked and untracked work. Work resumed in the attached managed checkout
`/Users/ndizazzo/.codex/worktrees/ninfer-performance/mesh-llm`, on the same
`codex/issue-1393-feasibility` branch. The primary checkout was untouched.
Ignored PTX/scripts needed for current trials were recovered from Carrack.
F09 and F10 workers have delivered bounded components; whole-model integration
and qualification remain open. F02 natural-language evidence is diagnostic and
neither native profile is promoted.

## Complete-source runtime comparison and next integration sequence

The user requested a deeper codebase investigation rather than further isolated
kernel tuning. The dedicated research worker inspected complete Ninfer source at
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`; parent reviewed the report and key
allocation, ordinary-round, vocabulary-head and NVFP4 dispatch consumers.
See `KNOWLEDGE/findings/ninfer-runtime-deep-dive.md` for pinned references,
measurement limits, numerical/quality gates, and five ranked experiments.

Next implementation is an exact resident MLP workspace/stream experiment using
F07's checked layout/lease. Compare existing execution, persistent storage with
existing waits, and enqueue-only chain with one completion boundary, keeping
arithmetic and diagnostic values unchanged. Exercise real FP8 and NVFP4 MLPs at
rows1/5/128/512, preserve distinct NVFP4 projection input scales, and separately
measure allocation/free counts, driver/host timing, whole-chain event span and
uninstrumented wall time. Shared input quantization is a separate ablation.
Only then extend stable storage and enqueue contracts to whole-model GPU greedy
selection and ordinary graph replay. The exact profile keeps its current gates.

Independent subsequent tracks are small-batch A16/head specialization, larger
pipelined NVFP4 tiles, and chunked GDN/tiled attention with long-context quality.
These are techniques to implement independently in Rust, not imported Ninfer code.
F03's prepared standalone fusion probe remains unqualified and is not selected
as the next performance priority. Existing split-K/compact full-model sanitizer
qualification remains in progress; do not start competing GPU trials or change
Carrack source/binaries until that process is terminal. Matched Ninfer quality,
long context, concurrency and serving evidence remain required for completion.

The split-K/compact sanitizer process subsequently completed successfully; see
`KNOWLEDGE/evidence/iterate-20260927/splitk-compact-check-1`. No benchmark remains
active at this checkpoint. Refresh live process/source state before the next run.

## Workspace and GPU selection checkpoint

The model-owned MLP workspace is opt-in (`MESH_SPECIALIZE_MLP_WORKSPACE=on`).
Paired short natural prompts measured 5.4–5.8% higher ordinary decode throughput
with identical generated tokens, logits and complete model-state hashes.
Model MTP all-accepted and every forced rejection position passed under both
recovery modes; compact recovery passed all three CUDA sanitizers. See F07's
knowledge entry and `model-workspace-1` / `model-workspace-check-1` evidence.
The 128/512-token prefill ablation is running separately at source `3d72188a4`;
do not change Carrack source/binary until its live job is terminal.

A new dedicated `feature_gpu_greedy` worker (GPT-6-Luna, max) implemented only
exact GPU selection kernels and an independent direct-FP32 oracle. Parent
reviewed, registered, and prepared the persistent host selector and ordinary
model integration behind `MESH_SPECIALIZE_GPU_GREEDY=on` (default off).
Full-logit diagnostic forwards retain their contract; model-profile compares
device-only selection with CPU tokens and full state. MTP rejects this flag
until separately integrated. Host266tests, Clippy and Just PTX build pass;
Linux compilation, standalone/device sanitizer qualification and paired model
throughput are next, not complete. This worker owns only GPU greedy selection;
no existing feature worker was assigned to a different feature.

Whole-round graphs, improved arithmetic/shape kernels, long-context algorithms,
matched Ninfer provenance/quality/rates and concurrent serving remain open.
Allocation/driver tracing has not yet been measured; nsys/ncu are not installed
on Carrack's current interactive PATH. Do not infer driver time by subtracting
synchronized kernel-event totals from uninstrumented wall time.

## Next measured bottleneck: resident tiled attention

The512-token workspace profile records22.05ms of causal-attention events in
52.39ms of summed subsequent decode events, and244.85ms during prefill. The
GPU-selection ablation is only0.26–0.52% faster in fixed-order short trials;
this is not a stable throughput win. Prioritize F05 integration after its
current predecessor's model sanitizer job terminates.

Parent inspected `resident_attention_core`: existing and candidate attention
kernels share the five-pointer/six-u32/FP32-scale ABI,256-thread CTA and token-major
BF16 KV layout. Preserve the default FP64 kernel; add an explicit separate
attention arithmetic profile for the FP32 online candidate. First audit identical
real-model prepared Q/K/V and cache values, covering nonzero past and causal tails,
against the exact GPU control and independent logical FP64 oracle using F05's
existing component budgets. Diagnostic outputs must not feed the control model.
Keep same-profile partition checks strict. Then compare teacher-forced logits,
state, natural-language continuations and task quality, followed by matched
prefill/decode timings at increasing contexts. Reject experimental attention in
MTP until recovery/profile identity is qualified. No `.ninfer` parser or imported
compute implementation is involved; existing F05 source remains its worker's
feature and parent owns integration/qualification.

## Online attention model checkpoint

At source238ac8a7d, real-weight layer3 attention audits passed the existing
component budgets at128/512inputs. Candidate-driven Python/prose32-token runs
passed strict same-profile partition and profile/control full-state checks.
Cross-profile prefill KL(exact||online) is0.04787/0.02129; Python tokens agree,
prose wording differs. This does not establish semantic degradation or quality
parity. The short trial was CPU-contended (many unrelated CUDA compiler jobs,
load49.44); timing cannot establish an uncontended win. Evidence is retained in
`attention-model-1`; exact remains default and MTP rejects online attention.

`attention-prefill-1` stopped at the strict128-token partition gate;512 was not
run. Next: localize all-row/layer divergence and audit the first differing
operator on identical inputs. Then candidate whole-model sanitizers,512-token
complete natural answers and multiple-position teacher-forced likelihood and
distribution comparisons on fixed reference text. Two next-token distributions
and opening-word agreement cannot certify quality. Keep time and quality claims
separate; independently inspect later-layer same-input attention if drift needs
localization. Neither native projection nor online attention should inherit a
quality pass from the other. Matched Ninfer provenance/configuration, long-context,
concurrency and serving behavior remain required.

## Partition failure localization

Completed row/stage diagnostics atfbf877158/ff387d8ce locate the online-attention
128-token partition failure to GDNlayer22,row101,NVFP4downprojection. All rows
through layer21 and every layer22 stage through BF16MLPactivation agree. Stage
observation preserves previous whole/token complete state hashes. Next: retain
actual row101 quantized input/scales and compare native NVFP4 MMA, existing
integer decode, and independent CPU oracle on identical weights; record BF16
boundaries, rawerror and input identity. The native/single-row arithmetic split
is the leading hypothesis, not yet a quantified root cause. Do not edit attention
arithmetic or loosen exact gates based on this failure. After localization,
resume candidate quality and performance qualification without making bitwise
identity a substitute for meaningful quality evidence.

Current remote source/binaryff387d8ce; all diagnostic processes are terminal.
Ninfer remainsinactive. Other mesh-llm processes appeared during trials and were
preserved; refresh process/service state. Stage1's exact memory-release gate
failed due to concurrent allocation; numerical stage checks passed. Stage2
passed memory release but retains the expected strict partition failure.

## Deep-dive follow-up workers

User explicitly requested GPT-6-Astra at low reasoning (correcting an initial
Luna request) for this dispatch. Three dedicated workers delivered their bounded work:

- `feature_a16_head_schedule`: item3, new sliced-K BF16-input/FP8-weight head
  kernel and independent reference; owns only fp8_a16_head device/reference
  files and optimizations/a16-head-schedule.md.
- `feature_nvfp4_pipeline`: item4, new32x32 multiwarp K64 staged NVFP4 kernel
  and independent reference; owns only nvfp4_prefill_tiled device/reference
  files and optimizations/nvfp4-prefill-pipeline.md. No baseline arithmetic edit.
- `feature_ordinary_graph_plan`: item2, source-backed whole-model eager-stream
  and graph integration design; owns only findings/ordinary-decode-graph-plan.md.

Each assignment is bounded and keeps existing feature workers separate. Workers
may not run Cargo, GPU jobs, SSH, Git mutations, services or further delegation.
Parent retains design review, integration, registration/assembly inventory,
serial builds, GPU/reference/sanitizer/model qualification, Git and the ongoing
layer22 NVFP4 numerical audit. Candidate delivery is not performance parity.

Parent registered both separate candidates and independent references, and added
all new PTX sites to the assembly inventory. 276 host tests, host Clippy, and
Just PTX compilation pass. GPU execution, actual resource use, sanitizers,
resident dispatch and performance qualification remain pending. The graph plan
is source-reviewed design only. No existing arithmetic profile was promoted.


A16 head follow-up: 37 independent GPU cases pass normal execution plus memcheck,
racecheck and synccheck. Explicit head-only A16 GEMV/MMA model profiles retain
decoder arithmetic and MTP rejection. Two prompt model comparisons pass strict
within-profile checks and equal same-input state, but the new schedule does not
improve ordinary decode. It remains experimental. Evidence is `a16-head-model-1`.
Dedicated NVFP4 worker has delivered a bounded GPU harness, not yet registered or
compiled; parent will qualify it next. Goal and meaningful quality gates remain open.


NVFP4 pipeline at e68e93d95: original and fixed-producer variants pass19 GPU
cases and all three sanitizers. Model128/512 comparisons pass strict same-profile
and cross-schedule captured logits/state/token checks. Original tiled regresses
prefill; fixed producers recover baseline with only0.28%/0.33% measured differences,
not a useful established speedup. Evidence: `nvfp4-tiled-model-1`. Default remains
unchanged. A separate32x128 CTA candidate with four output fragments per warp is
assigned to the same dedicated NVFP4 worker, still Astra low. It reuses A fragments
and preserves K64 arithmetic. Parent owns all integration and qualification.
Current512 diagnostic ranks FP8, attention and GDN above NVFP4; runtime parity
requires progressing these costs and quality, not merely passing tile tests.

## September 28: direct Ninfer artifact loading

The user explicitly selected native `.ninfer` input, superseding the previous
runtime-parser exclusion and the briefly approved offline-conversion prerequisite.
The offline bundle and unfinished optional assembler are preserved, not deployed.
The runtime remains independently implemented Rust with CUDA driver loading only.

The v3 single-file reader and model-source adapter now compile; 353 macOS library
tests, Linux-target Clippy, native Clippy, Just PTX build and no-console check pass.
Ordinary commands accept the exact pinned Ninfer file through checked canonical
views, encoded FP8 embedding gather, and FP32 GDN parameters. This is a new source
identity, not a reassignment of the raw `.mspec` control identity. Packed MTP and
proposal data remain in the source, with execution explicitly rejected for now.

Qualification order: compare independent Rust whole/object hashes to official
reader evidence; run parameter oracles and sanitizers; run same-source legacy/
stream state and logit checks; score fixed text and measure matched workloads.
All arithmetic differences listed in `KNOWLEDGE/findings/ninfer-import-contract.md`
remain explicit. No direct-file model correctness or throughput is established
by the host checks alone. Shared-GPU claims and earlier invalid host-overhead/
converter-provenance inferences were corrected in the knowledge entries.

## Worktree recovery, September 28

Archive cleanup removed the managed ninfer-performance checkout during this session.
Snapshot 81e7f0eb20cb6c08e20aa30bb571842f2b3e2a47 preserved all 12 pending files
on parent 9b627a3d61629132625b9a7c985f70d6beffa351. They were restored byte-for-byte
as uncommitted changes at /Users/ndizazzo/dev/worktrees/ninfer-direct-runtime,
on the same codex/issue-1393-feasibility branch. The primary checkout was not
modified. Carrack retained the tested commit and raw qualification evidence.
Use this new local path; the managed ninfer-performance path is now stale.

## Current state and next steps, October 1

This section replaces the root `HANDOFF_NINFER_*.md` notes, which were removed.
Local checkout: `/Users/ndizazzo/dev/worktrees/ninfer-direct-runtime`, branch
`codex/issue-1393-feasibility`. Work is uncommitted until the parent commits it.

Gap: the Rust direct-file eager stream measures 306.41/26.48 tokens/s
prefill/decode at 106 inputs and 327.62/18.16 at 512. Ninfer MTP0 with BF16 KV
measures 3,378/76.3 and 8,563/76.3; MTP4 with FP8 KV decodes at 141.2/213.5. The
user's target is roughly 10x decode. Graph replay, GPU greedy selection and the
MLP workspace each gave at most a few percent, so the next gains must come from
native MTP and shape-specific decode kernels.

Next, in order:

1. Native MTP forward on real weights using the qualified Q8 FC/projection,
   Q4 head and residency pieces, with target batch verification. Batched
   verification arithmetic is row-dependent; qualify against ordinary decode.
   Proposal vocabulary is 131,072 rows versus the 248,320-token target head.
2. GPU-qualify FP8 quantize reuse and NVFP4 A16 SwiGLU, then measure paired decode.
3. Full-corpus quality with the decode scorer against `findings/quality-gates.md`
   (NLL ≤0.5%, domain ≤1%, top1 ≥98%, mean KL ≤0.02, p99.9 KL ≤1, determinism;
   KL is a top-64 lower bound).
4. Long context, concurrency and ABI/serving integration remain after that.

Retained negative results: graph replay is not a material win; GPU event sums
cannot be subtracted from wall time; native FP8 prefill and online attention fail
top-1 (~94%); split decode fails quality. Paused: offline `ninfer_bundle`
(preserved, not deployed) and `nvfp4_prefill_large.rs` (unqualified).

Operations:

- Carrack `carrack.patio51.com`, checkout `/home/ndizazzo/dev/mesh/mesh-llm`;
  run as `ssh -tt carrack.patio51.com '/bin/zsh -ilc ...'`, build with
  `just specialize-tools-build`. `/usr/bin/time` is absent there.
- GPU0 RTX 5090 (SM120, driver 615.71.09,
  `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`). GPU1 RTX 3080 is not ours.
- The user authorized stopping user units `ninfer-qwen38.service` and
  `battlecity-comfy.service` for exclusive trials. Restore the initial state
  afterward and never start a unit that was initially inactive. Ninfer health:
  `http://127.0.0.1:1235/health`.
- Artifact `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`, 23,719,715,844
  bytes, SHA-256 matches the published manifest (unsloth-derived, not from the
  NVIDIA converter).
- Local gates, serial, through Just only:
  `MACOSX_DEPLOYMENT_TARGET=26.0 just with-lld cargo test -p mesh-specialize --all-features`,
  `... cargo clippy -p mesh-specialize -p xtask --all-targets --all-features -- -D warnings`,
  Linux-target Clippy for `mesh-specialize` only (`--target x86_64-unknown-linux-gnu`;
  xtask fails there on aws-lc-sys), `just specialize-ptx`, and
  `MACOSX_DEPLOYMENT_TARGET=26.0 just no-console-print`.
- At most two active workers. The commit hook rejects agent attribution trailers.
