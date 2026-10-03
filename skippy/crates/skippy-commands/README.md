# skippy-commands

Standalone Skippy command execution and output formatting. Argument parsing lives in `skippy-cli`; this crate executes the parsed commands against the shared Skippy API and renders their JSON output through a single console facility.

`models` resolves Hub references, downloads verified artifacts with size/SHA-256 verification and manages the local model cache. `runtime` lists, installs from explicit catalogs, imports and migrates native runtime caches. `split` plans and admits direct GGUF splits against the same release-bound certification roster Mesh uses, publishing stage configs and admission descriptors only after every stage is admitted. `console` installs the standalone diagnostics sink and writes JSON documents.

`prompt` connects to an existing stage-0 OpenAI endpoint and provides an
interactive chat or raw-completion client. It does not load a native runtime or
manage stage processes.

Responses stream as they arrive. After each response, a single-line emoji footer
shows generation speed, time to first token (TTFT), total elapsed time, input and
output token counts, and cached input tokens with their reuse percentage. The
footer has no indentation, goes to stderr, and is dimmed on terminals unless
`NO_COLOR` is set.
TTFT and total time are measured by the client, including connection and server
wait time. Generation speed uses server timings when supplied; otherwise it is
estimated from the output token count and the time between the first and last
generated text events, including reasoning events. Unavailable metrics appear
as `—`; incomplete streams are marked `Interrupted`.

Commands are expressed as plain typed actions (`ModelAction`, `RuntimeAction`, `PlanSplitCommand`), deliberately decoupled from Clap. The crate has no dependency on `skippy-serving`, adds no serving options types, and reads only the documented `SKIPPY_*` environment variables through `skippy-config` path policy.
