# F09: compact GDN recurrence replay

Status: bounded source and CPU reference are authored. Parent integration,
registration, compilation, device checks, and MTP qualification remain pending.
This entry makes no model-equivalence or performance claim.

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
