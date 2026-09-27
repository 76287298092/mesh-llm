# Specialized Qwen runtime implementation plan

Status: connected 64-layer GPU decoder and first performance iteration qualified.
The retained two-token hidden/logit/state reference matches exactly and all three
CUDA sanitizers pass. Matched three-sample medians improved short-prefix decode
from 1.55 to 20.07 tokens/s, 128-prefix decode from 1.55 to 18.15, and 128-input
prefill from 19.18 to 136.83. Both short/128-prefix profiles pass equality and
memory-release checks. See [performance results and remaining gaps](KNOWLEDGE/optimizations/decode-projections.md).
Text quality, longer-context qualification and matched Ninfer performance remain
open. This pass does not establish performance parity or serving readiness.

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
2. Larger prefill tiles and weight reuse, with independent reference and partition
   agreement before measuring. Preserve failed candidates and select by model timing.
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

The current prototype measures 19.2 tokens/s prefill at 128 synthetic raw tokens
and 1.55 tokens/s decode. Persistent weight/state payload is about 20.31 GiB for
135 positions. Allocation samples are not a peak-memory result. Full-model
numerical evidence covers one/two-token fixtures, and execution through 135
positions does not prove usable long-context quality. Matching Ninfer needs a
shared text corpus, tokenizer/chat support, verified weight identity, a matched
non-speculative control and further optimization. ABI integration, graph replay,
FP8 KV, MTP, prefix reuse and concurrent serving remain future work.

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
