# Specialized Qwen runtime implementation plan

Status: deployed-profile baseline, required instruction probes and first GEMM/RMSNorm
GPU trials and independent cuBLAS comparison complete. Profiling and specialized
model inference remain open.

Host policy now checks exact model/weights and explicit selected-device admission.
The [live device-policy trial](KNOWLEDGE/findings/selected-device-admission.md)
passes. Resident artifact discovery, ABI loading and full model execution are
still required before this can serve a request.

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
whole/chunk/token equivalence and sanitizers. Output gate/projection, complete
attention layers, full-model/logit parity and full model execution remain open.

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
| Upstream push and carrack branch synchronization | Complete through causal attention/KV trial | `9333e0918`; original carrack branch retained |
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
| H03 discovery and H04 ABI | Pending | No executable specialized model yet |
| Full model performance/context trial | Pending | None |

Validation belongs to the parent: serial focused Rust tests/check/Clippy, formatting,
repository no-console and crate-coverage checks where affected, and explicit
remote evidence. Update this ledger after each measured gate. Keep failed attempts.
