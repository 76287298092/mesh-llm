# Issue 1393 feasibility assessment

Assessed September 26, 2026, America/Toronto. MeshLLM revision
`4b48a298c347cea818e13d339ff069cc6172cd09`.

Subsequent implementation evidence is in the
[runtime work ledger](../../../../crates/mesh-specialize/PLAN.md).
The authorized experiment has now measured the deployed Ninfer baseline, qualified
the required Rust instruction probes, and connected all 64 decoder layers on
carrack. The first one-token hidden/logit comparison matches the independent CPU
reference bit-for-bit after correcting SiLU rounding. See the
[full-model evidence](../../../../crates/mesh-specialize/KNOWLEDGE/findings/resident-model.md)
for the current qualification boundary. The original assessment
below records the initial snapshot; its service-state and implementation-state
observations are historical. End-to-end performance parity remains unproven.

**Recommendation: proceed with a bounded Rust kernel and host-integration experiment.
Matching ninfer is technically plausible, but it is not demonstrated and the issue
needs corrections before full implementation.** There is no existing
`mesh-specialize` implementation in this checkout to enable or tune. This is a new
inference engine, quantization pipeline, and model execution implementation.

Rust is not a demonstrated performance barrier. It can target the relevant GPU
instructions. The harder question is whether our kernels, memory layout, graph
replay, recurrent-state handling, and speculative verification can do comparable
work at comparable cost. A successful GEMM experiment would justify continuing,
not establish end-to-end parity.

## What carrack actually runs

Read-only SSH inspection found `ninfer-qwen38.service` **inactive**, with a clean
stop recorded September 25 at 19:00:40 EDT. No ninfer process or port 1235 listener
was found. I did not restart it or submit inference requests.

| Item | Observed value |
| --- | --- |
| Target GPU | RTX 5090, 32,607 MiB reported by NVIDIA |
| Other GPU | RTX 3080, 10,240 MiB, outside the issue's single-GPU target |
| Driver | 615.71.09 |
| Source checkout | `/home/ndizazzo/dev/oss/ninfer`, HEAD `9e163eee4b8acec21ab0ac765107b6a3f287b217` |
| Source status | Modified `src/serve/serve_options.h`; untracked `.omo/` and converter directory |
| Binary | `build/apps/ninfer-serve`, modification time September 19 |
| Model file | `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`, 23,719,715,844 bytes |
| Serving profile | GPU 0, NVFP4, FP8 KV, context limit 131,072, concurrency limit 2 |
| Speculation | MTP, four draft tokens, optimized draft output head |
| Prefill | 2,048-token chunks |
| Startup allocation | 19.7 GiB weights, 9.28 GiB runtime, 1.14 GiB free |
| Shared KV capacity | 250,944 tokens, distinct from the per-request context limit |
| Prefix state | Two active and two cached device states; eight host states and 8 GiB host KV |

The source HEAD is **not verified binary provenance**. Neither the binary nor the
model file was rebuilt or checksummed. The model container was not parsed.

The September 25 service window contains 126 completed requests with reported
decode rates of **131.4–228.3 tok/s**, and a median per-request rate of
**173.65 tok/s**. These are historical, uncontrolled application observations,
not a fresh benchmark, an aggregate throughput result, or proof of C=2 saturation.
Prompts, cache hits, lengths, and MTP acceptance vary. For example, request 114
reported 104,319 prompt tokens, 16,384 output tokens, 20.6-second TTFT, and
138.0 tok/s decode. Request 117 reused 99.8% of its prefix and reported
184 ms TTFT, illustrating why warm-cache and cold-cache results must stay separate.

The complete metadata and extracted request metrics are in
[carrack-observation.json](carrack-observation.json). No prompt contents or API key
are retained.

## Changes needed in the issue

### The BF16 milestone cannot fit

The full BF16 text model cannot be resident on the 5090 while preserving the
no-offload and no-multi-GPU rules. The MLP weights alone require
`64 × 3 × 5120 × 17408 × 2 = 31.875 GiB`. The untied embedding and output matrices
add 4.736 GiB. That exceeds the card before attention/GDN weights, KV, or scratch.
Short context does not fix this. The geometry comes from the
[upstream Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json).

Keep BF16 per-layer oracle injection on carrack. Run the full independent reference
on CPU or a designated larger-memory reference machine, then make the first
fully resident 5090 execution quantized. Reference-only layer streaming is another
option, but must be explicitly separated from the deployed runtime's contract.

The same geometry implies 64 KiB of BF16 full-attention KV per token, and about
144 MiB of FP32 GDN matrices per complete sequence state. These are calculated
lower-level storage estimates, not measured total allocations. Rollback copies,
convolution state, graph buffers, and workspace add to them.

### Exact token equality needs a better weight contract

The issue assumes both engines independently convert the same BF16 checkpoint.
The official ninfer NVFP4 profile imports mixed packed NVFP4/FP8 values from
`unsloth/Qwen3.8-27B-NVFP4`. Its first 56 MLPs use NVFP4; later MLPs and other large
text projections use FP8. It is not uniformly four-bit. See the
[model card at carrack's source revision](https://github.com/Neroued/ninfer/blob/9e163eee4b8acec21ab0ac765107b6a3f287b217/model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md).
The local artifact's exact recipe remains unverified.

For a controlled runtime comparison, independently read the same upstream
quantized Safetensors and preserve their logical codes/scales in `.mspec`.
This requires changing the issue's BF16-only conversion requirement, but does not
require reading `.ninfer` or importing ninfer code. If independent BF16
quantization remains mandatory, compare quality and numerical error separately;
different quantizers can legitimately produce different greedy tokens.

Use independent arithmetic references, teacher-forced per-layer/state/logit
checks, deterministic replay, and fixed greedy fixtures for a declared numerical
profile. Track top-1 margins at divergences. A free-running token mismatch alone
cannot distinguish a kernel defect from different weights, KV precision, or
floating-point reduction order.

### Host integration is larger than `serves`

| Current code | Consequence |
| --- | --- |
| `crates/skippy-ffi/src/dynamic.rs:22,57–69` stores one global `OnceLock<Symbols>` | A second runtime load returns `AlreadyLoaded`. Model-by-model specialization and same-process llama.cpp fallback need instance-aware dispatch. |
| `crates/mesh-llm-host-runtime/src/lib.rs:230–237` selects a runtime during initialization | Concrete resident model identity must become available before selection. |
| `crates/mesh-llm-native-runtime/src/manifest.rs:26–46` has no model constraint | Add exact artifact/recipe identity, resolver input, rejection reasons, and old-manifest tests. |
| `crates/model-artifact/src/lib.rs:263–270` accepts GGUF and SafeTensors | `.mspec` needs format discovery, metadata, and a loader route. Passing it to current GGUF/SafeTensors handling is insufficient. |
| `crates/mesh-llm-native-runtime/src/resolver.rs:542–644` checks CUDA architecture/toolkit | Add selected-device VRAM admission, minimum-driver enforcement, and explicit driver-only dependency semantics. Current CUDA policy assumes cudart/cuBLAS/cuBLASLt. |

A narrow prototype can select one engine before loading anything and retain
llama.cpp fallback at startup when no specialization matches. That is an explicit
scope reduction from arbitrary same-process model switching. Otherwise carry
runtime instances through model/session handles and their optional symbol caches.
Neither approach should disguise the engine as backend `Other` to bypass checks.

The issue's symbol inventory is stale. The current loader requires **125 function
symbols plus `skippy_abi_version`**, for 126 exports total. Those 125 comprise
86 Skippy, 33 mtmd, and six llama/ggml functions. A separate foreign stub library
therefore needs **39 exports**, not 26. ABI **0.1.64** is matched exactly in
`crates/skippy-ffi/src/abi.rs:146–149`. Unsupported Skippy calls also need explicit,
signature-correct failure behavior. Loading inert stubs proves linkage only;
include a synthetic functioning model/session/tokenization path through serving.

Structured runtime events and optional capability hooks already exist at
`crates/skippy-ffi/src/dynamic.rs:390–415` and
`crates/skippy-runtime/src/runtime_events.rs:236–245`. Reuse them and audit remaining
dependencies rather than starting that deferred item from zero.

## Rust kernel feasibility

The documented route is Rust `no_std` device code, nightly PTX-kernel and inline
assembly support, emitted PTX, then final compilation through the dynamically
loaded CUDA driver. This can satisfy the no-C/C++ and no-nvcc runtime requirement.
See [Rust's NVPTX target](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html),
[experimental assembly registers](https://doc.rust-lang.org/beta/unstable-book/language-features/asm-experimental-arch.html),
and [CUDA's driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html).

Carrack's installed rustc 1.98.1 uses LLVM 22.1.8 and enumerates `sm_120a` and
`ptx87`. This is target-description evidence only. No Rust device kernel was
compiled, loaded, or timed in this assessment.

The NVFP4 block-scaled `mma.sync` instruction requires the architecture-specific
`sm_120a` target with PTX 8.7, or an appropriate documented family target.
Plain `sm_120` is insufficient for this instruction variant. `setmaxnreg` is
supported on `sm_120a`, but requires valid warpgroup execution and register-budget
setup. SM100 `tcgen05` assumptions cannot be transferred merely because both GPUs
are called Blackwell. See NVIDIA's
[MMA instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
and [register-budget instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).

The feasibility experiment must distinguish compiler success, correct PTX target,
driver acceptance, launch success, numerical correctness, and performance. Test
nonuniform scales and signed inputs, actual shared-memory transfers/waits, and
decode-sized and prefill-sized GEMMs. PTX registers are virtual; measure final
registers, spills, occupancy, bandwidth, and GPU execution time. Profiling remains
unverified: `ncu` and `nsys` were absent from the remote PATH and checked
`/opt/cuda/bin` locations. Pin toolchain, driver, device clocks, and reproduction
commands in the proposed knowledge base.

## What matching performance entails

The existing deployment already relies on graph capture, mixed quantization,
compressed KV, speculative decoding, and recurrent prefix-state reuse. A short
context, batch-one, non-speculative engine would prove correctness but would not
match this operating profile.

Upstream's separate RTX 5090 records report 71.2 tok/s for NVFP4 without
speculation at a 7,680-token prompt. Its MTP3 NVFP4 corpus reports 126.1 tok/s for
stories, 194.3 for code, and 219.8 for structured output. These use different
workloads and are not an isolated MTP speedup. They also differ from carrack's
MTP4/FP8-KV configuration. NVFP4 improves prefill in those records without winning
every batch-one decode comparison. See the
[versioned performance report](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/docs/performance/qwen3.8-27b.md).

Match the actual workload in separate gates:

1. Compare both engines with speculation off, identical logical weights,
   numerical settings, tokenized prompts, context, output budget, and concurrency.
2. Add graph replay and tune quantized matrix operations, attention/GDN, fusion,
   and KV layout against independent correctness checks.
3. Implement MTP4, draft-head behavior, efficient verification, and full recurrent
   rollback to match carrack. A rejected suffix must restore GDN and convolution
   state as well as the KV frontier and draft state.
4. Match cold/warm prefix reuse, long contexts, and C=1/C=2 service behavior.
   Complete continuation state must correspond to the precise prefix frontier.
   See ninfer's [state reuse description](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/docs/maintainer/resource-scheduling-and-context-cache.md).

For each gate, record raw requests/results, exact binary and artifact hashes,
recipe identity, GPU UUID, driver/toolchain, clocks/power, cache state, and other
GPU workloads. Measure committed output tok/s, client TTFT, total request time,
prefill throughput, VRAM, and quality. Separate per-request latency from aggregate
concurrent throughput. Alternate engines on the same GPU and retain all repeated
samples. Agree the acceptable parity tolerance before running comparisons.

The issue's 2x llama.cpp floor also needs a measured same-host baseline. Nothing
collected here establishes that ratio. Current ninfer master has newer serving
and speculation features; pin the deployed control rather than chasing a moving
README result.

## Proposed decision gates

| Gate | Evidence required to continue |
| --- | --- |
| Correct the spec | Quantized first resident model, explicit weight/numerical identity, startup-only versus multi-engine dispatch, refreshed ABI inventory |
| Prove Rust execution | Required instructions execute correctly; representative GEMM/RMSNorm has usable register allocation, timings, and profiling |
| Prove host integration | Exact resident-artifact/device selection, real serving-path exercise, explicit unsupported calls, wrong-model rejection and fallback |
| Prove the model | Independent per-layer and recurrent-state oracle, chunked versus incremental prefill parity, quantized end-to-end fixtures |
| Prove competitiveness | Controlled no-speculation comparison, then deployed MTP4/cache/context/concurrency comparison and quality checks |

I would fund the first two technical experiments before committing to the full
engine. The model implementation can follow if they pass. Performance parity is a
credible research target, but an unconditional parity commitment is not supported
by the evidence. Keep the prototype local until the issue's production trust,
capability negotiation, and failure-containment work is complete.

## Evidence and validation limits

- [Issue snapshot](issue-snapshot.json), open and last updated September 9, with no comments at capture.
- [Carrack observation](carrack-observation.json), service metadata and historical request metrics.
- Source inspection and arithmetic verified the global loader, symbol count,
  ABI version, unsupported artifact format, and BF16 residency blocker.
- No implementation code changed. No builds, GPU kernels, inference benchmarks,
  service changes, CI runs, commits, pushes, or GitHub comments were performed.
- Documentation checks cover JSON validity, local evidence links, and whitespace.
  They do not establish runtime correctness or performance.
