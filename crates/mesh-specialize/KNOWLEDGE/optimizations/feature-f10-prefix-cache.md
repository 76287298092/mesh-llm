# F10 bounded prefix checkpoint policy

Status: generic cache policy is registered and host ownership/identity tests pass.
Resident runtime integration and GPU suffix-equivalence qualification remain pending.
This change does not promote prefix reuse as a runtime default and does not claim
any performance gain.

The policy owns an opaque checkpoint payload under a caller-supplied identity:
weights content ID, arithmetic profile, KV format, and model/context geometry ID.
It stores the exact consumed `Vec<u32>` token prefix and the caller-accounted
payload bytes. The `max_bytes` limit applies only to supplied `accounted_bytes`;
it does not include token-vector, identity-string, entry, or container overhead.
Real integration must include the full retained footprint in its accounting.
Lookup requires exact identity equality and chooses the longest
cached complete prefix of the requested token sequence. A hit returns an
immutable borrow and prefix length, so a caller must fork checkpoint state before
appending suffix tokens; the cache retains its original payload. Entries use a
bounded least-recently-used order and checked byte accounting. The order is kept
as LRU-to-MRU indices, avoiding a wrapping timestamp.

## Checkpoint admission contract

Before inserting a real session payload, the caller must establish that it is a
committed, usable checkpoint. The cursor and all device state must include every
token in the recorded prefix. Final hidden state and any pending next-token
information must correspond to the final token in that prefix. The payload must
retain the full session state required to continue: all attention KV, GDN
recurrent state, and convolution history. A failed or poisoned session is never
admissible; this generic cache cannot inspect `T`, so the parent integration must
enforce this rule before calling `insert`.

The cache proves ownership, identity isolation, prefix selection, and bounded
eviction only. It does not prove that a forked session processing the uncached
suffix is equivalent to an ordinary full prefill. Real session suffix
equivalence remains a required independent qualification gate. These tests do
not emulate CUDA state or claim equivalence.

## Evidence and limits

- Source base: `16eb9a43345c74d904b683ad0e6b4d3e5331bf01`; implementation commit is pending.
- Model: not applicable to the generic cache policy. No model session was inserted.
- GPU, driver, clocks, CUDA tools: not applicable; no GPU or service action was run.
- Reproduction: focused unit tests are included in `src/engine/prefix_cache.rs`; they were not run in this worker because the parent owns module registration and Cargo validation.
- Expected: identity mismatch, incomplete token match, or invalid/oversized insertion cannot produce a hit; entry and byte limits hold after every accepted insertion.
- Observed: source implementation and mock-payload tests cover identity-field mismatch, exact and longest token prefixes, LRU/byte eviction, replacement accounting, and payload drop ownership. Test execution and session suffix equivalence are pending parent validation.
- Performance and Ninfer comparison: not measured. Ninfer's observed warm-prefix reuse is not evidence of this implementation's speed or parity.

Do not derive a weights identity from a filename. The caller must provide a
content-bound weights ID and exact arithmetic, KV, and geometry identifiers.
Do not insert a session after a partial failure, and do not treat a cache hit as
proof that the caller's suffix continuation is correct.


Parent registered the cache module. Its ownership/identity/bounds tests pass as
part of the 259-test macOS suite; host and Linux Clippy pass. No resident model
prefix checkpoint is cached yet, and GPU suffix equivalence remains untested.
