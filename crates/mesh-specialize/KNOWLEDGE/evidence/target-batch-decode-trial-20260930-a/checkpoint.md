# Target batch checkpoint

The full N=1..5 report passes normal execution and synccheck. Memcheck and
racecheck exit 124 at the original timeout and fail the qualification gate.
Their partial reports and raw logs remain unchanged. Neither timeout is a pass.
Both authorized services were restored to active, with Ninfer HTTP health restored.

The original executable SHA256 is
`96f6542e5693fd9c8d22decec9fb9fae43b89a91c3c78a400fdb7c1187607126`.
After adding checked `--rows N` selection, the rebuilt executable SHA256 is
`6846136ff113b66924091b94ad2b74bb2fdce80ea075c5482050afc211eb626d`.
It has not yet run the selected-row GPU qualification at this checkpoint.
The artifact, PTX, and fixture identities remain those in `hashes.txt`.

Retained source archives:

- `target-batch-before-selected-20260930.tar.gz`, SHA256
  `68a4bcdd712b7c08cfbc61648e03a78824b74d3833466b3d94aee6ee032fe69b`,
  contains the five files immediately before selected-row integration.
- `target-batch-selected-sources-20260930.tar.gz`, SHA256
  `d7e2ea541a3d258c44746ac7a74eada406dc248606c44fdc48182036dbc7a76b`,
  contains the four selected-row implementation files. AppleDouble metadata
  entries are not Rust source.
- `target-batch-selected-source-hashes-20260930.txt`, SHA256
  `83674ab9ab11ee082d38c3b310c92f63bad5294c398ca19ba446339f8439b7bd`,
  records the current remote Rust source files under the specialized runtime,
  its references, and the specialize CLI. This is a pending exported snapshot,
  not a clean committed revision.

Selected-row filter QA accepts five transformed positive fixtures derived from
the retained normal report and rejects 45 corrupted fixtures. Those transformations
are test inputs only, not new GPU evidence. The next trial must cover every
N=1..5 under normal, memcheck, racecheck, and synccheck. Each report must compare
all 64 layer outputs, full hidden and vocabulary words, all 128 state regions,
the cursor, and the three-token ordinary continuation exactly.

Native MTP admission, complete native MTP execution, whole-model quality,
and throughput parity remain open. No numerical gate was relaxed.
