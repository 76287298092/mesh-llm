# Ninfer deployed-profile baseline, 2026-09-26

Measured on carrack at 22:51 EDT, harness commit `dc1e24fe966b903e6e9d7a882d5d67e4e1d3def5`.
All nine requests completed in 35.25 seconds. The fixture text is synthetic.

| Cold case | Input tokens | Output tokens | Prefill tokens/s | Decode tokens/s | Client first output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short 1 | 725 | 512 | 6,033 | 198.2 | 121 ms |
| Short 2 | 725 | 512 | 5,956 | 164.2 | 124 ms |
| Medium 1 | 7,186 | 512 | 10,110 | 192.3 | 719 ms |
| Medium 2 | 7,186 | 512 | 10,416 | 201.0 | 698 ms |
| Long 1 | 42,837 | 512 | 8,028 | 187.8 | 5,354 ms |
| Long 2 | 42,837 | 512 | 8,012 | 189.0 | 5,367 ms |
| Short final | 724 | 512 | 7,065 | 168.5 | 108 ms |

All these cold cases reported zero reused input tokens. The separate warm repeat
reused 42,830 of 42,837 tokens, computed seven, and reached first output in 48.9 ms;
decode was 190.3 tokens/s. Do not pool its prefill rate with the cold cases.
Warmup used 276 input/64 output tokens and is excluded from the comparison table.

The existing service profile is Qwen3.8-27B NVFP4, MTP four draft tokens with
draft LM head, FP8 KV, context ceiling 131,072, KV capacity auto, prefill chunk
2,048, and concurrency capacity two. Tests were serial, greedy, thinking disabled.
Decode rate uses 511 intervals for 512 output tokens. This validates at least
42,837 input plus 512 generated tokens; it does not qualify 131K context or
concurrency two. There is no matched non-speculative control yet.

GPU: RTX5090, UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, driver 615.71.09,
32,607 MiB total. All 352 samples at approximately 100 ms intervals reported
31,004 MiB total device usage. Ninfer process usage before/after was 30,046 MiB;
ComfyUI stayed at 498 MiB. Peak sampled SM clock was 2,872 MHz and power 599.84 W.
These are sampled device/process figures, not allocator-exact peaks. Clock ramp
and content-dependent speculative acceptance contribute to variation. No clocks
or service settings were changed.

Binary SHA256: `140ec5e660fb613ec2f57d2a99c4a274a0363f2cdca7e2652ca7a0337bff49ad`.
Weight SHA256: `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
Weights: `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`; only bytes were hashed,
not parsed. Source revision/binary correspondence remains unverified.

Compact raw metrics: [JSON](../evidence/ninfer-baseline-20260926.json).
Full plan, requests, timestamped SSE JSON, device CSV and memory samples are in
`target/specialize/baseline-20260926/` in both this worktree and carrack's checkout.
Local archive `target/specialize/baseline-20260926.tar.gz` has SHA256
`4cefb6c93a2b32bc147fe42a4d694cefca98a27fe360262299a984a6f45c5cb0`.
The untracked raw evidence must be retained when retiring this worktree.

Reproduce with `just specialize-tools-build`, generate a fresh plan with
`xtask specialize baseline-plan`, then use `just specialize-baseline` with
`NINFER_API_KEY` sourced from the existing private environment file. Never put
the value in a command argument or evidence file.
