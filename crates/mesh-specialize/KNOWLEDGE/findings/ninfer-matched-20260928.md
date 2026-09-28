# Matched Ninfer reference, September 28

Status: measured on Carrack, 2026-09-28 (exact timestamps in the evidence). This replaces the
[September 26 baseline](ninfer-baseline-20260926.md) as the comparison target: the
serve executable hash differs from that run. Its file timestamp was September
27 at 21:27; that timestamp does not establish build provenance.

## Identity

- Observed Ninfer source checkout `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d` with one local edit
  (`serve_options.h` default `max_concurrency` 1→2, overridden by the CLI) and an
  untracked local converter `tools/convert/qwen3_8_27b/`. `ninfer-serve` SHA256 and
  the (older, 2026-09-19) `ninfer-perplexity` SHA256 are in `hashes.txt`. The
  serve and perplexity binaries' source revisions are unverified.
- Model `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`, 23,719,715,844 bytes,
  SHA-256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`,
  identical to the published artifact manifest and September 26 model. The
  manifest declares an unsloth quantized source. Our `.mspec` pins another
  revision of that repository; logical tensor equality remains unresolved.
  Local NVIDIA converter scripts do not prove this artifact used them.
- Prompts: `target/specialize/reassess-20260927/matched-prompts.json`, rendered with
  tokenizers 0.22.2 and jinja 3.1.6, thinking disabled. Follow-up extraction with
  Ninfer's own read-only `tools.artifact.reader` confirms that its tokenizer and
  template assets render all five prompts identically and produce identical
  token IDs for those prompts and all four perplexity slices. Stored hashes:
  `evidence/reassess-20260928/ninfer-identity/token-assets-comparison.json`.
  This is asset-level equality for these inputs, not a capture of the server's
  internal token buffer. Server-reported prompt counts also agree.

## Configuration

Separate `ninfer-serve` on port 18235 (deployed service stopped and restored),
`--max-context 40960 --kv-capacity 40960 --max-concurrency 1 --prefill-chunk 2048
--no-prefix-reuse --greedy --no-thinking`, CUDA graphs enabled by default. Requests:
streaming chat, `temperature 0`, `max_tokens 256`, one warmup, then each prompt in
forward and reverse order. ComfyUI remained resident (about 500 MiB); sampling does not prove complete
inactivity. Label: shared GPU. Every request produced 256 tokens.

## Results

Decode = `(predicted_n − 1) / predicted_ms`. Prefill rate is `prompt_n / (prompt_ms/1000)`. The column labeled TTFT
below is the server-reported `prompt_ms`, not measured client-visible TTFT.
Client `first_visible_ms` is retained separately in request records. Two repetitions per cell agreed within 0.5%;
medians shown.

| Prompt tokens | MTP0 BF16 KV prefill tok/s | Server prompt ms | MTP0 BF16 KV decode | MTP0 FP8 KV decode | MTP4 FP8 KV decode | MTP4 accepted/drafted |
|---:|---:|---:|---:|---:|---:|---:|
| 106 | 3,378 | 31.4 | 76.3 | 76.8 | 141.2 | 143/445 |
| 512 | 8,563 | 59.8 | 76.3 | 76.8 | 213.5 | 181/292 |
| 2,048 | 11,537 | 177.5 | 75.8 | 76.5 | 231.2 | 187/270 |
| 8,192 | 11,170 | 733.4 | 73.7 | 75.5 | 172.4 | 165/357 |
| 32,767 | 8,781 | 3,731.7 | 69.1 | 72.9 | 216.9 | 187/267 |

MTP4 speed is prompt dependent through acceptance and is not a runtime constant.

## Slice perplexity

`ninfer-perplexity`, corpus `ninfer-ppl-1m-v1-slice12k` (four 12,288-token streams,
see [quality gates](quality-gates.md)), 12,287 scored tokens per stream.

| KV, context/stride | chinese | english long | english ref | code | overall NLL | PPL |
|---|---:|---:|---:|---:|---:|---:|
| BF16, 512/256 | 1.944369 | 1.935350 | 1.894918 | 0.708471 | 1.620777 | 5.0570 |
| FP8, 512/256 | 1.947506 | 1.936098 | 1.894284 | 0.707754 | 1.621410 | 5.0602 |
| BF16, 4096/2048 | 1.843737 | 1.817683 | 1.719042 | 0.478297 | 1.464690 | 4.3262 |

## Evidence and limits

Evidence: `evidence/reassess-20260928/ninfer-matched-2/` (plans, per-request
records, request logs, summaries, perplexity reports, hashes, unit states).
`ninfer-matched-1` failed before any request because the runner caps
`timeout_seconds` at 180. Limits: greedy only, one request at a time, 256 outputs,
no concurrency or serving-load measurement, ComfyUI resident but idle, one host.

Rule: compare our runtime against these MTP0 BF16 KV rows using the verified fixture IDs,
and against MTP4 only with the same prompts and a reported acceptance rate.
