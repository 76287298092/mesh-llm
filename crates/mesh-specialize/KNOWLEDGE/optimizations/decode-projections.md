# Decode and prefill performance iteration

The September 27 performance pass retains parallel exact FP8 projections,
warp-parallel BF16 gates, packed NVFP4 loads, four-row FP8 prefill reuse and an
attention reduction with fewer synchronization barriers. These changes improve
our own runtime materially. They do not establish Ninfer performance parity.

## Matched before and after

Three fresh-session samples per case, identical raw token IDs, eight fixed output
tokens and one untimed warmup. The table uses sample medians. The before control
is the preserved pre-optimization executable/PTX, rerun during this pass.

| Measurement | Before tokens/s | After tokens/s | Ratio |
| --- | ---: | ---: | ---: |
| Decode after two inputs | 1.547 | 20.071 | 12.97x |
| Decode after 128 inputs | 1.553 | 18.146 | 11.69x |
| Prefill, 128 inputs | 19.184 | 136.830 | 7.13x |

Prefill includes final logits and the first greedy output; decode includes seven
subsequent complete forwards, logit download and CPU greedy selection. Weight
loading, JIT warmup, session allocation and diagnostic reference work are outside
the timed intervals. Every sample retains the same eight output IDs. The 128-input
case exercises 135 positions, not a long-context quality qualification.

## What changed and why

The original short-prefix profile attributed 83.39% of its 633.56 ms summed
kernel-event time to FP8 projections and another 11.26% to BF16 gates. The costly
FP8 implementation used MMA followed by FP64 error bounds and serial refinement.
Finite E4M3 values are exact signed integers divided by 512; at supported widths,
their product sum fits below 2^51. Warp-parallel i64 dots preserve the independent
FP64-dot result without the serial fallback. The conversion and scale order are
unchanged. Prefill now reuses a decoded weight across four token rows; decode
keeps one row. Both paths have zero reported local storage, with 36/40 registers.

The small BF16 A/B gates use parallel FP64 dots. Their reduction order differs
from the sequential reference, so fixture agreement is not a universal bit-equality
claim. NVFP4 reads aligned full four-byte data/scale words with the existing
bounds-checked byte fallback for tails. Its MMA arithmetic and order are unchanged.

Attention keeps the original FP64 reduction tree and exponential polynomial,
reduces the dot with first-warp shuffles, and computes the online-softmax scalar
updates once before broadcasting them. The convergent reduction stays inside
one inline PTX block so LLVM cannot thread a later lane-zero branch through its
barriers. This is a synchronization change, not reduced arithmetic precision.

| Iteration | Short decode | 128-prefix decode | 128-input prefill | Decision |
| --- | ---: | ---: | ---: | --- |
| Exact FP8 decode | 8.029 | 7.243 | 19.290 | Retained |
| Parallel BF16 gates | 17.931 | 14.391 | 19.658 | Retained |
| Exact FP8 prefill and NVFP4 word loads | 19.991 | 15.685 | 112.562 | Retained |
| Four-warp NVFP4 block | 20.072 | 15.662 | 112.519 | Rejected |
| Four-row FP8 and out-of-line attention | 19.982 | 18.037 | 132.771 | Replaced after memory check |
| Four-row FP8 and inline attention | 20.071 | 18.146 | 136.830 | Retained |

The retained short-prefix profile sums to 40.741 ms across 1,476 launches,
versus 633.564 ms before. NVFP4 accounts for 18.627 ms, exact FP8 for 9.927 ms,
BF16 gates for 2.705 ms and FP8 quantization for 2.605 ms. After 128 inputs the
sum is 46.227 ms, including 5.634 ms of causal attention. Both profiles pass
control-versus-instrumented output/state equality and memory release. Event
instrumentation changes scheduling; these totals are not model throughput and
the difference from wall time does not isolate allocator or launch overhead.

## Qualification and failed candidates

The retained code passes the two-token independent full-model reference with zero
BF16 hidden-state differences in all 64 layers, exact final logits and exact
whole-sequence versus token-at-a-time state. Both FP8 variants pass all 129,178
independent FP32/BF16 output checks, including every finite code pair, row/K/output
tails, maximum supported K and cancellation. BF16, NVFP4 and attention component
fixtures also pass. Memcheck and synccheck report zero errors; racecheck reports
zero hazards, errors or warnings. This does not establish natural-text quality or
agreement with Ninfer's different arithmetic profile.

Host tests and Clippy pass on macOS and Carrack. CUDA assembly and the repository
console-output policy check pass. The retained benchmarks observe return to their
pre-model free-memory baseline after release. Payload residency is unchanged:
20.160 GiB weights and 20.312 GiB weights plus state at capacity135. Those are
payload/checkpoint measurements, not transient memory peaks.

The four-warp NVFP4 experiment passed arithmetic but did not improve timings; its
entrypoint was removed. The first inlined attention reduction timed out after
240 seconds during component checks. PTX showed a barrier duplicated across
lane-zero control paths. The out-of-line workaround passed all arithmetic and
sanitizers but failed the short profile's global free-memory check, leaving
1,117,061,120 fewer free bytes after model release. Its short benchmark also
missed the release baseline, while its 128-input benchmark returned to baseline.

The initial stack-allocation explanation was not established: offline SASS reports
the same 152-byte attention frame for the old and revised kernels, and that
64-register offline build is distinct from the unrestricted runtime JIT. The final
inline block removes the helper call and passes both benchmark and both profile release checks.
Keep the failed report rather than attributing its memory delta solely to stack
size. Failed compilation evidence also records the Rust named-assembly-label lint;
PTX's block-scoped labels use a local documented lint allowance and assemble cleanly.

## What remains between this runtime and Ninfer

The [pinned Ninfer source comparison](../findings/ninfer-performance-comparison.md)
identifies concrete remaining work:

- Dedicated single-token FP8/NVFP4 GEMV with BF16 activations. Ninfer selects these
  paths by shape and token count. Our current activation quantization and rounding
  boundaries differ, so this needs a separate reference and quality qualification.
- Pipelined prefill matrix tiles with shared-memory reuse. Our four-row integer
  kernel remains far smaller than Ninfer's 64-by-128-by-128 FP8 tile and does not
  use FP8 tensor cores. Larger tiled kernels are the main remaining prefill project.
- Fused projections and activations, reusable workspace, and CUDA graph replay.
  These can remove repeated quantization, allocations and launches; the current
  event profile does not isolate host-side savings.
- MTP, FP8 KV, tokenization/chat serving and long-context qualification. These
  remain separate runtime capabilities, not claimed by this performance pass.

Ninfer's measured deployed profile was 164.2-201.0 decode tokens/s and
5,956-10,416 cold-prefill tokens/s on different requests, with MTP4 and FP8 KV.
Our synthetic prompts, output length and timing boundaries are not matched to that
baseline. The gap remains substantial; do not present the ratios above as an
engine comparison with Ninfer. A fixed text corpus and a non-speculative Ninfer
control are needed before assigning the remaining gap to individual components.

## Reproduction and provenance

Final source `30c1e3e6b94129bede413211b2c79301734b0b4f`; PTX SHA256
`fc44eab0434edbdad7407fbd795ede1d9d79595e896cd433291c7304f20d6e88`.
Carrack RTX5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver615.71.09,
CUDA13.4.92 and Rust1.98.1; Rust PTX compiler nightly-2026-09-25. No power/clock
settings were changed. The model, independent-reference hash, all trial source
commits, executable/PTX hashes, exact wrappers, build logs, JSON samples and
service/GPU snapshots are in [the evidence directory](../evidence/perf-20260927/README.md).
Raw evidence remains under `target/specialize/perf-20260927/` on both hosts.
Ninfer source was inspected at `9e163eee4b8acec21ab0ac765107b6a3f287b217`; deployed
binary provenance was not established. No Ninfer source entered the Rust engine
or its independent reference.

The final trial restored Ninfer at 09:53:32 EDT, PID3244818, with HTTP200.
ComfyUI PID448118 remained running at 498 MiB throughout the recorded handoffs.
