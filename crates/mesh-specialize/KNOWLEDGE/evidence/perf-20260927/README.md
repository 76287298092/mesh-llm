# Performance iteration evidence, 2026-09-27

These are bounded raw-token trials on Carrack's RTX 5090. The final findings and
comparison limits are in [the optimization report](../../optimizations/decode-projections.md).

Each trial directory records its source commit, executable/PTX hashes, GPU and
service state, component/full-model check, timing samples and kernel profile.
`attention-round/` is the failed 240-second timeout, including its empty result
file. `warp4-round/` is the numerically passing but rejected geometry experiment.
Neither failed nor rejected evidence establishes a retained speedup.

The interim `reuse4-round/` contains memcheck, racecheck and synccheck reports,
a failed short profile and a fresh before control using the preserved old
binary and PTX. Its profile passes output/state checks but fails the global CUDA
free-memory check; the long profile was therefore not run. `inline-round/` uses
the revised reduction. `run-round-stack.sh` records the interim wrapper, and
`run-round.sh` records the final wrapper. It requires the named
clean source revision, verifies PTX hashes, bounds each command to 240 seconds and
8 GiB host memory, and restores the Ninfer user service through an exit trap.
Do not replay its historical stop/start action without current authorization and
checking other users of the host. Use a new output directory for each run.

The before executable was built from `55ee5ae56`, which adds profile command
wiring to the original decoder. Its SHA256 is
`67f9f807c1a544ff7cf2fb3d5b4b2c4dd5d8f4395b0c70644178a42fdb8099fc`.
Before PTX SHA256 is
`fa04eb2e19c22bcd47fc657c9adb6d8e079349719d31f7bbb213fe85a8a70ab6`.
The final source is `30c1e3e6b94129bede413211b2c79301734b0b4f`;
its PTX SHA256 is
`fc44eab0434edbdad7407fbd795ede1d9d79595e896cd433291c7304f20d6e88`.
The interim source was `eff80d4854dbeb71fef0eca2e5aa2c7c60b6e766`, with PTX
`aac9b70b8202e685e851e76b27b434dfb1d9fbb06e7daa9ef1c47f14b1c09724`.
Exact executable hashes are in each trial's `trial-hashes.txt`.

The model is the independently imported raw `.mspec` Qwen3.8-27B artifact with
weights identity
`sha256:f49713878a072f8c9043060dc0e2f3b28421301e49471bee0c13c7570e59e81e`.
The independent two-token reference SHA256 is
`6e5fd184d8b93d04c3716ec10a12605a3b51065580d9dc711812c8e2c1525da1`.
The [initial model timing](../../findings/model-timing-20260927.md) records model
provenance and the exact timing endpoints. Prompts and generated IDs are in every
benchmark report. They are synthetic IDs, not a natural-language quality corpus.

JSON is compacted without changing values. Service-journal lines matching
`api[_-]?key`, `authorization` or `bearer`, case insensitive, are omitted completely.
Raw journals remain under ignored `target/specialize/perf-20260927/` on both
hosts. PTX, executables, model files and the inspected Ninfer source archive are
also outside Git. Build/test logs and `summary.json` accompany the final reports.
The summary script checks expected output IDs and computes medians from all
three samples. No timing confidence interval or statistical significance is claimed.
