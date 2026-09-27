# One full-decode profile, 2026-09-27

Source `55ee5ae56b21a09661d8b79199c73a9f2f999539`.
See [the finding](../../findings/model-profile.md) for interpretation and limits.

- `decode-two/report.json` records instrumented/control output and state equivalence,
  memory release, wall times, all 1,476 launch timings grouped by launch shape.
- `kernel-totals.json` aggregates those groups by kernel name. It is derived from
  the report, not a separate trial.
- `run-profile.sh` records the protected service-stop/restart procedure.
- The initial Linux test log preserves the validator module inclusion failure.
  Retry, Clippy and release build logs preserve the successful checks.
- macOS logs preserve library tests, Clippy and no-console validation.
- Trial identity, GPU observations and before/after service state accompany the
  report. The service journal omits 1 authentication-related line.

Raw originals remain at `target/specialize/qwen-profile-20260927/` on the local
worktree and Carrack. This committed mirror normalizes trailing blank lines in
text files. No binary, weight or PTX copy is committed here. The PTX is unchanged
from the independently qualified full-model trial.
