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
`unsloth/Qwen3-8B-GGUF:Q4_K_M`, or an existing local GGUF path. Remote model
files are resolved to an immutable revision and cached before loading. Skippy
reports the model ID and API address after `GET /v1/models` succeeds.

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
skippy models show unsloth/Qwen3-8B-GGUF:Q4_K_M
skippy models download unsloth/Qwen3-8B-GGUF:Q4_K_M
skippy models installed
```

`recommended` is a short standalone starter list. `search` queries Hugging
Face for GGUF repositories. `show` resolves an exact artifact and displays its
revision and file set. `download` verifies the selected files and prints the
primary local path; serving the same reference reuses the cache. The optional
`--sha256` and `--size-bytes` pins on `download` apply to the primary file and
are checked on cache hits too. Without an independent expected digest, the
reported SHA-256 describes the bytes obtained but is not an external
authenticity claim.

The model cache resolves from `models --cache-dir`, then
`SKIPPY_MODEL_CACHE_DIR`, then the platform cache directory under
`skippy/models`. Hub endpoint and token configuration follows Hugging Face
settings. `skippy models remove org/repo --dry-run` previews removal of all
local revisions of one repository; omit `--dry-run` to remove them. It never
deletes a remote repository.

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
skippy runtime install --manifest /path/to/runtime-catalog.json
skippy runtime import /path/to/verified-bundle --dry-run
skippy runtime import /path/to/verified-bundle
```

Runtime storage resolves from `--runtime-cache`, then
`SKIPPY_NATIVE_RUNTIME_CACHE_DIR`, then the platform cache directory under
`skippy/native-runtimes`. `SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR` adds explicit
bundle roots. A manual `runtime install` requires exactly one `--manifest` or
`--manifest-url`. `runtime import` copies a verified bundle and leaves its
source untouched. Mesh environment variables do not select standalone
runtime storage.

## Output for terminals and automation

Interactive terminals show concise status, download progress, and a ready
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
