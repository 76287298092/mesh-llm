# Bounded baseline harness

Status: host tests pass; live measurements pending. Implementation starts from
`2d9ea2073`, on macOS arm64 with Rust 1.98.1. GPU/driver/clocks are not applicable
to these parser tests.

`xtask specialize baseline-plan --output NEW_FILE` generates nine synthetic
requests. `just specialize-baseline PLAN NEW_DIRECTORY` sends them serially to
the loopback server. The saved plan contains the API-key environment variable
name, never its value. Every successful request preserves raw SSE JSON events,
client arrival times, server usage/timings, and the request body. Failures persist
a safe error and halt the run. No retries or redirects are allowed.

Validation: `just with-lld cargo test -p xtask specialize` passed 18 tests;
`just with-lld cargo clippy -p xtask --all-targets -- -D warnings` passed. Tests
cover SSE framing through the actual accumulator, truncation, bounds, role-only
chunks, timing monotonicity, cache accounting, and malformed records. Local test
linking reports the installed Homebrew Rust macOS deployment-target mismatch;
the test binary nevertheless ran successfully.

Review caught a framing/accumulator integration defect: consuming `[DONE]` in
the framer without forwarding it makes every completed observation appear
truncated. Preserve the integration test. Actual tokenizer counts and measured
cache reuse, not fixture row counts, determine the reported context evidence.
