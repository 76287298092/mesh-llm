# First connected-model timing, 2026-09-27

Historical pre-optimization evidence. See the [subsequent measured performance
iteration](../optimizations/decode-projections.md) for current results.

The Rust-only 64-layer prototype runs on Carrack's RTX 5090, but this untuned
implementation is far below Ninfer's measured speed. This is the first model
measurement, not a matched performance or language-quality comparison.

| Raw-token case | Prompt tokens | Fixed output tokens | Prefill seconds | Prefill tokens/s | Decode tokens/s | Processed positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved short prefix | 2 | 8 | 0.649290122 | 3.0803 | 1.5479 | 9 |
| Varied synthetic prefix | 128 | 8 | 6.660594758 | 19.2175 | 1.5518 | 135 |

Each case is one fresh-session sample following an untimed one-token warmup.
Prefill includes final logits and selection of the first output token. Decode is
seven subsequent complete model forwards, including final-logit download and CPU
greedy selection. Weight loading, JIT warmup, session allocation, memory queries,
CPU reference work and diagnostic observers are outside these timings. The whole
harness takes 26.160490424 / 32.159249045 seconds including loading; do not use
those durations as inference time. Fixed-length generation deliberately ignores
EOS. Prompt IDs and generated IDs are saved in each report. The larger fixture
starts with 248044 and uses `300 + (i * 7919) % 100000` for `i=1..127`; it is not
natural text or the Ninfer baseline corpus. Short output repeats ID 271; the
varied case alternates 98094 and 6013. These outputs do not establish useful text
quality, tokenizer correctness or chat-template support.

## Memory and context

Weights occupy 21,646,588,928 payload bytes (20.1600 GiB). State payload is
154,533,888 bytes for capacity 9 and 162,791,424 bytes for capacity 135. Combined
persistent payload is 20.3039 / 20.3116 GiB. CUDA checkpoint samples show maximum
increases of 20.3047 / 20.3125 GiB from the pre-model baseline. Maximum sampled
CUDA-reported device use is 21.7124 / 21.7202 GiB, including other device users and CUDA
resources. These checkpoints miss buffers allocated and freed inside a forward;
they are not transient peaks. Both runs return memory to their pre-load baseline.

The larger run processes 135 positions: 128 prompt tokens plus seven generated
inputs. Its eighth generated output has not been fed back. This verifies bounded
execution and cursor growth, not numerical quality at that context. Independent
full-model hidden/logit qualification currently covers one/two-token fixtures;
the two-token fixture and three sanitizers pass bit-for-bit. Earlier state
allocation at capacity 131,072 establishes only that allocation fits. Long-context
inference remains unqualified.

## Ninfer comparison boundary

The [deployed-profile Ninfer baseline](ninfer-baseline-20260926.md) measured
5,956–10,416 prefill tokens/s and 164.2–201.0 decode tokens/s in its listed cold
cases, using 724–42,837 input tokens and 512 outputs. It uses MTP4, FP8 KV,
2,048-token prefill chunks and a different artifact/container. The Rust prototype
uses BF16 KV, no MTP, no graph replay and accurate but untuned kernels, including
FP64 attention accumulation. The input/output counts, content, context capacity,
allocation policy and timing endpoints are not matched. Do not derive an engine
speedup or memory saving from these figures. The direct conclusion is that the
current prototype does not approach the deployed Ninfer performance target.

The next engineering decisions require profiling the full forward, improving the
matrix/attention kernels and allocation/launch path, then repeating a fixed text
corpus with tokenizer support and a matched non-speculative Ninfer control.
MTP, graph replay, FP8 KV, long context and concurrent serving remain separate
work. The current evidence demonstrates Rust execution feasibility, not that
competitive performance has been achieved or is guaranteed.

## Reproduction and provenance

Source `d99306f87cee4f669590e6b2ae259c96d1f150ab`; release xtask SHA256
`60653b0574f4e1217fa47d9e1f8d40b6dc1cc1e47b9df0c64307cf60940ab512`; PTX SHA256
`fa04eb2e19c22bcd47fc657c9adb6d8e079349719d31f7bbb213fe85a8a70ab6`.
Artifact model `qwen3.8-27b:text:nvfp4-fp8:upstream-raw-v1`, weights identity
`sha256:f49713878a072f8c9043060dc0e2f3b28421301e49471bee0c13c7570e59e81e`.
RTX 5090 UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver 615.71.09,
CUDA 13.4.92, Linux Rust 1.98.1. No clocks, power settings or other GPU workloads
were changed. The local Rust PTX compiler is pinned to nightly-2026-09-25.

```sh
just specialize-tools-build
target/release/xtask specialize qwen-model-bench --artifact ARTIFACT --tokens 248044,271 --output-tokens 8 --repetitions 1 --ptx PTX --device 0 --output NEW_FILE
```

Run only after admission checks and the authorized bounded Ninfer stop, using the
saved `run-model-bench.sh` restoration wrapper. The short run restored Ninfer at
07:22:57 EDT, PID 3176422; the larger run at 07:24:04 EDT, PID 3177329. Both returned
HTTP 200. ComfyUI PID 448118 remained at 498 MiB. Compact reports and service/GPU
evidence are in [the experiment directory](../evidence/qwen-model-20260927/README.md),
under `bench-two-eight/` and `bench-128-eight/`. Full raw evidence remains in
`target/specialize/qwen-model-20260927/` on both hosts.
