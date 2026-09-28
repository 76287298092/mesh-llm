# Bounded chunked benchmark

Status: implemented, unqualified. This is measurement tooling, not an optimization
promotion. No Cargo, GPU, SSH, or Git commands were run by this worker. No timings,
quality result, usable-context claim, or speedup is established.

## Entry and scope

The separate command accepts exactly these ordered arguments:

```text
xtask specialize qwen-chunked-bench --artifact PATH --tokens-file PATH --chunk-size N --output-tokens N --repetitions N --ptx PATH --device N --output NEW_FILE
```

`--tokens-file` is a JSON array of unsigned 32-bit token IDs, limited to 1 MiB.
Prompt length is 1..32768, chunk size 1..512, output count 2..512, and repetitions
1..5. IDs must fit the package vocabulary. Device ordinal must be nonnegative.
Capacity is exactly `prompt_len + output_tokens - 1`, checked against the package
configuration. Accepting a size is not evidence that inference at that size has
been qualified. Existing benchmark limits, default execution, and exact arithmetic
controls are unchanged.

Library entry: `kernels::qwen_chunked_benchmark(path, ptx, device, request, report)`
with `ChunkedBenchRequest { tokens, chunk_size, output_tokens, repetitions }`.
The report is a mutable JSON object, preserving completed repetitions if a later
step fails. CUDA execution requires Linux, an SM120 device, and SM120a PTX.

The command directly selects ordinary StreamForward, independently of
`MESH_SPECIALIZE_EXECUTION`. It uses the executor's supported-profile validator:
exact projection profiles, default exact attention or explicit `split-decode`,
workspace off, split-K off, and NVFP4 audit disabled. It does not change those
controls. `MESH_SPECIALIZE_GPU_GREEDY` only controls legacy execution; this path
always performs StreamForward's GPU selection. Raw environment settings and
resolved profiles are recorded separately. Under `split-decode`, rows 1..8 use
the separate attention candidate, including small prefill chunks/tails; larger
chunks keep exact attention. Cross-partition equality is still a strict check.

## Execution and measurement

- The pure plan divides the prompt into contiguous half-open row ranges. Each
  chunk calls `StreamForward.forward` sequentially against the same session.
  Cursor advances use checked addition and must match the plan.
- Constructor `max_rows` is `min(chunk_size, prompt_len)`. Weights and the model
  arena stay resident across the warmup and all timed repetitions.
- Each chunk computes its final-row vocabulary head and greedy selection. This
  is wasted work on non-final chunks and is included in measured prefill time.
  Only the final chunk's selection becomes generated token zero.
- Ordinary decode executes exactly `output_tokens - 1` forwards. Output is fixed
  length, with no EOS stopping, prefix reuse, speculative decode, or concurrency.
- One excluded warmup executes the full same prompt with the same partition and
  two output tokens. Every warmup/repetition starts from a new zeroed session.
  Session allocation/initialization and diagnostics are outside measured time.
- Each repetition records chunk start/end rows, row count, past before/after,
  relative start/end timestamps, and elapsed time; prefill sum and wall time;
  per-token decode intervals and count, decode sum/wall time, generated IDs,
  final cursor, and sequence wall time. Prefill rate uses wall time; decode rate
  uses the sum of the `outputs - 1` forward intervals.

Admission checks canonical weight/state layouts, the actual arena planner,
checked RoPE size, and a 512 MiB margin before resident allocations. With
`split-decode`, it also includes
`attention_v2_plan::persistent_workspace_bytes(max_rows, capacity)` for each
live executor. The initial budget covers the larger of the timed working set or
the diagnostic pair of streams/sessions, with shared weights counted once.
Admission is checked before/after module load, before diagnostics, and before
allocating each fresh timed/warmup session. This is a free-memory snapshot, not a
reservation or allocator-exact peak; unrelated allocation races remain possible.

## Strict partition diagnostic

For prompt lengths up to 512, two fresh sessions compare chunked prefill against
one-shot StreamForward under the same selected profile, followed by two decode
forwards. Both decode paths consume the one-shot selected token even after a
divergence, so their input sequences remain identical. Diagnostics use a separate
`prompt_len + 2` capacity, including when timed output count is two. They never
reuse timed session state or change the timed capacity.

Each of the three checkpoints compares BF16 logits exactly, selected tokens,
cursors, and SHA256 hashes of every initialized state region. KV hashes cover the
initialized token-major prefix, not unused capacity. Convolution histories and
recurrent matrices are hashed in full. Byte counts and both sets of hashes are
recorded; state downloads and hashing are excluded from throughput timings.

`partition_check.passed` remains a separate strict boolean. A numerical mismatch
records `false` and is explicitly printed by the command, but does not discard
measurements or fail benchmark execution. There is no tolerance-based replacement
for this strict check. A diagnostic execution error does fail the run and retains
its error/partial report. For longer prompts, the check is `not_run` with
`passed: null`, never a default pass.

## Evidence and failures

Output uses exclusive creation and refuses any existing path. Once an ordered
request reserves its new file, input/validation/execution errors are saved with
`completed: false`, phase, and error. An initial `started` report is synced before
execution; normal return saves completed repetitions and available diagnostic
steps. A killed process can leave only that initial report; in-flight intervals
are not a durable journal. Malformed command syntax and inability to create/write
the output cannot produce a new completed report.

Provenance includes exact input IDs, SHA256 of concatenated little-endian u32 IDs,
raw token-file SHA256, PTX SHA256/path, executable SHA256/path, package version,
artifact content-derived model/weight identity, source checkpoint revision,
recipe hash, device information, and raw profile environment. The source-tree
revision is explicitly not captured; parent qualification must record it beside
the executable identity. Artifact integrity is checked by VerifiedArtifact and
again while uploading weight objects. No `.ninfer` parser or new artifact profile
is part of this command.

Validation authored: pure plan tests at lengths 1/511/512/513/8192 with chunks
1/128/512; invalid chunk 0/513, prompt/output bounds, capacity equality, and cursor
overflows; command flag-order and token-hash encoding tests. Rustfmt was run with
edition 2024. Compilation, unit-test execution, Clippy, GPU/sanitizer checks, and
all runtime measurements remain parent-owned and not run here.

Suggested parent qualification covers short prompts with chunks 128/512 and a
separate 8192-token run. Record exact source revision, PTX/executable identities,
device/driver/toolchain, competing load, and retained output paths. No observed
before/after results exist for this new command yet.

Durable rule: a completed benchmark and a passed strict partition diagnostic are
different facts. Neither establishes text quality or long-context readiness.
