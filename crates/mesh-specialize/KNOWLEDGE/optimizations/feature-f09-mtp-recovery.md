# F09: compact GDN recurrence replay

Status: the primitive is device-qualified and integrated behind an experimental
compact-recovery selector. Full-model normal checks pass all four forced rejection
positions and all-accepted commits with exact target tokens and state. Full-model
memory, race, and synchronization checks also pass. Repeated text timings are
in progress.
Full-forward recovery remains the default. Earlier pending statements below
record the original worker delivery; parent results follow.

## Contract

`kernels/nvptx/gdn_replay.rs` adds two candidate entrypoints:

- `gdn_recurrent_record` preserves the first twelve `gdn_recurrent` arguments
  in the same order and adds a final `replay_delta: *mut f32`. It executes the
  existing recurrence with its explicit `mul.rn.f32`, `add.rn.f32`, and
  `sub.rn.f32` sequence and unchanged BF16 input/output boundaries. It stores
  each computed delta at `[row, value_head, value_column]` while it updates
  state and writes both existing output forms. Keys and decay remain in
  caller-owned buffers.
- `gdn_replay_state(k, decay, delta, state, rows, key_heads, value_heads,
  width)` applies records in increasing row order. For every state cell it
  performs rounded `state * decay`, rounded `key * delta`, then rounded add, in
  that order. It does not use FMA or reassociate these operations.

Both kernels use one CTA per value head and one thread per value column, with
the same value-head to key-head mapping as `gdn_recurrent`. Their initial MTP
row bound is 1..=5. Head counts and width retain the existing recurrence
constraints. The host reference rejects invalid shapes, extents, decay/beta
domains, non-finite inputs, and arithmetic overflow. `replay_prefix` also
models a zero-row accepted prefix as a no-op from the base state.

## Independent reference and tests

`reference/gdn_replay.rs::record` independently evaluates the original scalar
recurrence and produces output, unrounded output, final state, and deltas.
`replay_prefix` implements only the recurrence replay identity. The authored
CPU test compares its state for every prefix from zero through all five rows
against fresh runs of `gdn_recurrent_reference::run` from the same nonzero base
state. The fixture includes signed Q/K/V values and nonuniform decays, and also
compares the complete recorded output and state with the original recurrence.
Invalid row bounds, extents, and non-finite/domain-invalid values have rejection
cases. The parent owns executing these tests; they have not run in this bounded
source task.

## Proposed assembly inventory entries

These are candidate inventory rows for parent registration, not qualification:

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/gdn_replay.rs:gdn_recurrent_record` | CTA/thread coordinates; explicit rounded FP32 multiply/add/subtract; BF16 decode and RNE encode; original ordered recurrence plus row/head/column delta stores | SM120a | Independent ordered recurrence and BF16 boundaries in `reference/gdn_replay.rs::record`, checked against `reference/gdn_recurrent.rs` | Source authored; module registration, PTX compile, launch, numerical checks, and sanitizers pending |
| `kernels/nvptx/gdn_replay.rs:gdn_replay_state` | CTA/thread coordinates; explicit rounded FP32 decay multiply, key/update multiply, and add in row order | SM120a | `reference/gdn_replay.rs::replay_prefix`, with accepted-prefix state independently recomputed by the original recurrence | Source authored; module registration, PTX compile, launch, numerical checks, and sanitizers pending |

## Integration boundary

The caller must retain the untouched base GDN state, the exact verification
keys and decays for the rows it may accept, the delta records, and the accepted
KV rows. Replay is valid only from the same base state and for a prefix of the
records produced by that verification pass. The MTP transaction must also
preserve the matching causal-convolution prefix history and hidden/token
boundary; this primitive does not own or repair those values. Parent integration
must still prove forced rejection at each draft position, all-accepted commit,
target-only greedy token equality, and whole target-state equality.

No Cargo command, NVPTX compilation, GPU launch, sanitizer, model trial, timing,
or live Ninfer comparison was run for this bounded task. Until parent-owned
integration and qualification pass, there is no model-equivalence or
performance result and no resident dispatch change.


## Parent qualification, 2026-09-27

Source `f5870dab84177bf0c93d864fc82a26ed6702be89`, PTX SHA256
`089e027984eb7937fef47f5de3c01e92f08108d832eda1937ffae68eddbfd782`.
259 macOS crate tests, host Clippy, PTX compilation, no-console-print, and Linux
Clippy/tools build pass. Fixture expansion initially exposed an immutable iterator
compile error and two cognitive-complexity failures in test assertions; those
were corrected without relaxing checks or suppressing lints.

The parent `feature-gdn-replay-check` device probe passes at widths 1, 2, and 128,
with two key heads and four value heads. Full recorded output, raw FP32 output,
deltas, and state agree bitwise with the independent record oracle. Each prefix
from zero through five rows agrees bitwise with a fresh original recurrence
from the initial state. Signed decimal-rounded inputs and beta/decay endpoints
are included. Memcheck and synccheck report zero errors; racecheck reports zero
hazards, errors, and warnings. Recording uses 35 registers, replay 24, with no
local or shared bytes. Evidence: `../evidence/iterate-20260927/features-f09-1/`.
Ninfer was inactive before and after; ComfyUI remained resident.

This qualifies the synthetic primitive only. Whole-model forced rejection at
every draft position, all-accepted commits, KV/convolution/hidden/token boundary
equality, and measured MTP performance remain open.

### Next integration design

Record the exact normalized keys, decays, deltas, and pre-convolution projected
QKV during verification. Retain an untouched base session until the accepted
prefix is known. For rejection, copy only accepted KV rows from verification,
replay GDN deltas into base recurrent state, and reconstruct convolution history
from the base history and accepted projected QKV rows. Advance the cursor through
an abort-poisoning transaction only after all layer recovery succeeds. Preserve
verified hidden rows and the existing draft teacher-forcing boundary. Keep the
full-forward replay path as an explicit control during qualification. This first
integration can retain verification forking; removing that copy needs separate
ownership and failure-recovery proof.


Parent model integration is now authored behind `MESH_SPECIALIZE_MTP_RECOVERY=compact`.
Absent or `full-forward` retains the previous control. Invalid values fail closed.
Verification records exact GDN operands/deltas and raw projected QKV. On rejection,
a cursor transaction applies compact recurrence/history recovery and copies only
accepted attention KV rows into the untouched base session. All-accepted rounds
retain the existing verification-session commit. Reports label the recovery mode.
Source integration is not yet qualified; Linux build and real-model comparisons
are required before any performance or correctness claim.

The qualification harness now forces the first round to accept all configured drafts, then separately forces rejection at every draft position with a correct preceding prefix. Each run checks exact final target tokens and full state hash against ordinary greedy decoding and reports the observed first-round acceptance count. Execution of this expanded model harness remains pending.

A host admission guard now enforces the record kernel contract before launch: at most five rows, at most 64 key heads, and a power-of-two head width. This does not change arithmetic for the qualified Qwen geometry.

The report retains `forced_rejection` for the first-position case and adds `forced_rejections` for complete position coverage, preserving existing result readers.


## Full-model compact recovery qualification

Source `74916ccd7ea593da898acfe2ffa2a79f616e66be`, unchanged F09 PTX
`089e027984eb7937fef47f5de3c01e92f08108d832eda1937ffae68eddbfd782`.
Evidence: `../evidence/iterate-20260927/compact-recovery-1/`.
Normal full-forward and compact runs use eight output tokens at depth four.
Each forces an all-accepted first round and rejection after exactly 0, 1, 2,
and 3 accepted draft tokens. Every run agrees with ordinary greedy target
output and the complete target state hash. Sanitized compact runs use six
output tokens at depth four and exercise the same four rejection positions and
all-accepted boundary; all pass. Memcheck and synccheck report zero errors;
racecheck reports zero hazards/errors/warnings. These are raw-token recovery
checks, not semantic-quality evidence or a matched Ninfer comparison.

On the one unsanitized short sample, full-forward recovery took 55.91 ms total
versus 3.14 ms for compact recovery; decode was 23.40 versus 28.41 tokens/s.
This is one shared-GPU sample, not a stable performance estimate. Repeated
natural-language trials remain necessary. Verification still forks the full
state. Ninfer remained inactive and ComfyUI stayed resident.

Host checks at this integration: 259 macOS library tests; 329 Linux library
and 26 Linux validation tests; Clippy on both hosts, PTX compilation and
no-console-print pass. Saved copied test logs normalize trailing blank lines;
original logs remain in the ignored working output directory. Subsequent source
changes only add the recording geometry guard and preserve the legacy JSON
field; they do not alter the valid tested Qwen arithmetic or recovery path.
