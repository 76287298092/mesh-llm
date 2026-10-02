# Skippy CLI

`skippy` runs a model locally and exposes OpenAI-compatible and Anthropic
Messages APIs on the same address. It also downloads models, manages native
runtimes, and runs explicit split stages. The CLI owns argument parsing and
terminal output; `skippy-serving` owns the serving loops.

Build with `just skippy` to package a local native runtime and build the CLI.
The executable automatically discovers the verified `native-runtimes/` directory
beside it, including `target/debug/native-runtimes` from that build. Use
`just skippy-cli-build` only when a compatible runtime is already available.
If no local runtime matches, serving tries a compatible release runtime;
source builds can have a newer Skippy ABI than the published release. Use
`--runtime-bundle /path/to/bundle` to select another local bundle explicitly.

## Run one model on one machine

```sh
skippy models recommended
skippy serve --model Qwen3-0.6B-Q4_K_M
```

`--model` also accepts a Hugging Face repository reference such as
`unsloth/Qwen3-8B-GGUF:Q4_K_M`, or an existing local model path. Remote GGUF
files and complete SafeTensors checkpoints are resolved to an immutable revision
and cached before loading. Direct SafeTensors serving requires a model family
supported by Skippy's native checkpoint loader; Qwen3.5 checkpoints are not yet
supported, so use a GGUF variant such as
`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` for that family. Skippy
reuses installed exact refs from the shared Hugging Face cache and reports the
model ID and API address after `GET /v1/models` succeeds.

For a SafeTensors family supported by the native checkpoint loader, `--quant`
uses Mesh's on-load quantization recipes (`preserve` is the default). Low-bit
recipes that need an importance matrix also accept
`--checkpoint-imatrix /path/to/model.imatrix`. Quantization does not add support
for an otherwise unsupported checkpoint architecture; use a GGUF variant for
Qwen3.5 today.

```sh
skippy serve --model /models/supported-checkpoint --quant Q4_K_M
```

For multimodal GGUFs, Skippy looks for a matching installed `mmproj` sidecar.
Catalog downloads include the catalog's projector asset; select an explicit
local projector with `--mmproj /path/to/mmproj.gguf`.

To start the server and immediately chat with the model in the same terminal:

```sh
skippy serve --model Qwen3-0.6B-Q4_K_M --prompt
```

The API remains available while the prompt is open. Enter `:quit` to end the
prompt and shut down this combined serving session. `:reset` clears chat
history. To connect a prompt to a server that is already running, use
`skippy prompt --endpoint http://127.0.0.1:9337/v1`.

Both API styles use the same model ID and port. In another terminal:

```sh
curl http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B-Q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

```sh
curl http://127.0.0.1:9337/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"Qwen3-0.6B-Q4_K_M","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

## Find and download models

```sh
skippy models recommended
skippy models search qwen --limit 10
skippy models search qwen --mlx --sort downloads
skippy models show unsloth/Qwen3-8B-GGUF:Q4_K_M
skippy models download unsloth/Qwen3-8B-GGUF:Q4_K_M
skippy models installed
skippy models updates --check
skippy models updates unsloth/Qwen3-8B-GGUF
skippy models cleanup --unused-since 30d
```

`recommended` reads the same remote `meshllm/catalog` as Mesh. `search` queries Hugging
Face for GGUF repositories by default; `--mlx` selects MLX repositories, and
`--catalog` limits results to the curated catalog. `show` resolves an exact artifact and displays its
revision and file set. `download` uses a catalog layer package when available;
`--direct` downloads the exact Hub artifact instead, and `--draft` also fetches
the recommended speculative draft. Direct downloads verify selected files and
include a catalog projector when present. Serving the same reference reuses the
Hub cache. The reported
SHA-256 describes the bytes obtained but is not an external authenticity claim.

Skippy and Mesh use the same Hugging Face Hub cache, honoring `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, and `XDG_CACHE_HOME` in that order. Hub
endpoint and token configuration follows Hugging Face settings. `skippy models
delete org/repo:Q4_K_M` previews the selected installed GGUF files; add
`--yes` to remove them. It never deletes a remote repository.

`installed` includes both GGUF files and cached SafeTensors checkpoints.
`cleanup` is a dry run by default and only targets managed model files recorded
by the CLI; use `--yes` after reviewing its preview. `delete` also previews
derived stage files associated with the selected model before removal.
`updates --check` compares cached repository refs to upstream revisions without
downloading; `updates <repo>` or `updates --all` refreshes cached files and
`config.json` using the same policy as Mesh.

Skippy owns the layer-package workflow used by both local and distributed
serving. Package creation is a dry run unless `--confirm` is supplied:

```sh
skippy models package unsloth/Qwen3-8B-GGUF:Q4_K_M
skippy models package unsloth/Qwen3-8B-GGUF:Q4_K_M --confirm --follow
skippy models certify meshllm/Qwen3-8B-Q4_K_M-layers --package-only
skippy models prune
skippy models prune --yes
```

`models package` also accepts Mesh's job-management switches (`--status`,
`--logs`, `--cancel`, `--list`, and `--update-script`). Certification can use
`--api-base` for runtime smoke checks and `--report-out` for an auditable JSON
report. Stage-cache pruning is dry-run by default and preserves pinned stages.

## Run a split on one machine

Prepare one config per stage. The ordered `--worker` addresses are the internal
stage listeners; the public inference API remains on port `9337` by default.

```sh
skippy plan-split --model-path /models/model.gguf --model-id local-model \
  --worker 127.0.0.1:9400 --worker 127.0.0.1:9401 --output-dir new-plan
skippy serve --config new-plan/stage-1.json --stage-transport binary --worker-only
```

Start stage 1 in its own terminal, then start stage 0 in another:

```sh
skippy serve --config new-plan/stage-0.json --stage-transport binary
skippy prompt --endpoint http://127.0.0.1:9337/v1 --model local-model
```

`--prompt` can be added to the stage-0 `serve` command. A downstream stage has
no public API, so `--prompt` and `--worker-only` cannot be combined. The binary
stage transport is the normal split path. Internal HTTP stages use
`--stage-transport http --worker-only` and run on their own listener; that
internal `/v1/messages` route is separate from the public Anthropic route.

`plan-split` admits every native stage before writing configs. Its output
directory must not exist. The generated stage files, not the diagnostic
`admissions.json`, are the serving inputs. Regenerate the plan when changing
the topology rather than editing stage files by hand. The direct-GGUF planner
defaults to one lane, a 512-token context, and CPU execution
(`--n-gpu-layers 0`).

## Run stages on different machines

Use addresses reachable from the other machines in the plan:

```sh
skippy plan-split --model-path /models/model.gguf --model-id local-model \
  --worker 10.0.0.10:9400 --worker 10.0.0.11:9401 --output-dir remote-plan
```

Place each generated `stage-N.json` on its worker and make the exact verified
model files available at the paths recorded in that config. Start the final
stage first and stage 0 last. To shut down cleanly, stop stage 0 first and
wait for it to exit before stopping downstream workers.

## Inspect runtime and machine state

```sh
skippy doctor
skippy runtime list
skippy runtime list --available
skippy runtime install
skippy runtime install metal
skippy runtime install --manifest /path/to/runtime-catalog.json
skippy runtime remove <native-runtime-id>
skippy runtime prune
```

Runtime storage resolves from `--runtime-cache`, then
`MESH_LLM_NATIVE_RUNTIME_CACHE_DIR`, then the same platform cache directory as
Mesh. `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` adds bundle roots. `runtime install`
selects the recommended compatible runtime by default; `--manifest` selects an
explicit release catalog. `runtime prune` keeps the active and previous release
unless `--active-only` is passed.

## Output for terminals and automation

Interactive terminals show concise status, download and native model-load progress, and a ready
summary. Use `--output human` to request that presentation explicitly.
Commands that return one result use JSON when stdout is redirected; use
`--output json` to request it explicitly. A long-running `serve` command uses
JSONL when redirected, or when `--output jsonl` is given:

```sh
skippy models installed --output json | jq .
skippy serve --model /models/model.gguf --output jsonl | jq -c .
```

Each JSONL line has `schema_version`, `sequence`, `type`, and `data`. Wait for
the `ready` event before sending API requests. Progress, diagnostics, and
errors are events in JSONL mode; failures also return a nonzero exit status.
Terminal control characters are never written to JSON or JSONL output.

`example-config` emits one JSON stage-config document without loading a
runtime. SIGINT and SIGTERM request graceful service shutdown; draining
in-flight requests depends on the serving backend.
