# Staged FP64 at 2K and 8K context

Status: measured on 2026-09-29 as an opt-in ordinary-decode schedule. This
trial establishes matching greedy token IDs against the exact runtime control
at 2,048 and 8,192 prompt tokens. It does not establish full-logit or state
equivalence at either length. The chunked harness cannot run its one-shot
partition check above 512 prompt tokens.

## Reproduction and identity

The trial used Carrack GPU 0 (RTX 5090, SM120, UUID
`GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`) with the clean checkout at
`1f520b1aeb1b0101946ad916e0838d5852890bd6`. The `.ninfer` artifact SHA-256
was `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`;
the `xtask` SHA-256 was
`548f9a0a4b687491c3faa11836042086cc1dbe14e6bbc22a94f80d4e442a6ae6`;
the PTX SHA-256 was
`47a6cb946ab572809c1af7117c6f3070043a91384fdce7da0174a06a801ae762`.
Raw reports and process/service logs are in
`KNOWLEDGE/evidence/reassess-20260928/staged-long-context-1/`.

The ignored trial script is
`target/specialize/reassess-20260927/run-staged-long-context.sh`. It selects
`wiki-2k` and `pg19-8k` token-ID arrays from the pinned `matched-prompts.json`
fixture, uses 512-row prefill chunks, and runs `qwen-chunked-bench` with 32
fixed output tokens, two repetitions per run, and forward/reverse mode order.
Each mode gets a fresh session per repetition; its warmup is excluded. The
script admits only a clean checkout and the pinned PTX, runs GPU-exclusive,
and restores the two initially active user services through its EXIT trap.
The log ended `ALL-DONE`; Ninfer health returned HTTP 200 after restoration.

## Whole-model measurements

The table reports the median of four decode rates in tokens/s. Decode rates
count the 31 forwards after the final prefill selection. The chunked bench
includes per-token host wall work and uses the same text-only native source,
GPU, prompt, and output length for both schedules.

| Prompt tokens | Exact control | FP8 vector16 + paired FP64 A/B + NVFP4 PRMT + staged attention |
| ---: | ---: | ---: |
| 2,048 | 8.62 | 16.46 |
| 8,192 | 2.66 | 5.63 |

The four repetitions of each mode completed at both lengths. All eight
generated 32-token sequences per prompt were identical across both modes and
repetitions. Prefill rates were about 222 tokens/s at 2K and 98.5 tokens/s at
8K; the combined decode schedules did not measurably change prefill. This is
evidence of deterministic greedy continuation for these inputs, not an
independent arithmetic oracle or a model-quality score.

Every report marks `partition_check.status=not_run` because the one-shot
StreamForward diagnostic stops at 512 rows. No long-context full-logit or
state-hash comparison was run, and this result does not qualify graph decode
or native MTP. The 2K/8K performance is still far below the short-context
Ninfer reference, which was measured under different context lengths and
must not be treated as a matched long-context comparator. Scaling the staged
coefficient reduction and removing redundant FP8 activation quantization
remain separate candidates requiring whole-model GPU qualification.
