# Mesh-Native Distributed SSD Inference

## Status

Design proposal. This is the target architecture for combining Mesh's
multi-node staged inference with node-local, SSD-backed model weights. It is
not an implementation or a claim that the current SSD telemetry plugin,
Flash-MoE adapter, or a successful DeepSeek-V4.1 run already provides this
behavior.

## Goal

Run one model request across multiple Mesh nodes while keeping each node's
weights on its own local storage and bounding the amount of model weight data
resident in physical memory. Mesh contributes aggregate *execution capacity*
through explicit stage placement; it does not create shared physical RAM or a
remote virtual-memory pool.

The initial acceptance target is the user's seven-file
`DeepSeek-V4.1-Flash-Q2_K` GGUF set: approximately 264.5 GB (246 GiB), 40
layers, and 384 experts with six selected per token. Its seven GGUF files are
container shards, not Mesh layer-stage packages.

## Current boundaries

- The `mesh-ssd-node` plugin reports node-local resources and model inventory.
  It does not load weights, pool capacity, plan stages, or affect the
  scheduler.
- The Flash-MoE plugin registers one external OpenAI-compatible backend. That
  backend owns a complete single-node inference request; the adapter does not
  compose it with Skippy stages or dispatch experts between peers.
- Skippy already has the right execution boundary for distributed layer
  inference: contiguous layer stages on peers exchange activations and
  required state. Published layer packages can let a peer materialize only its
  assigned stage artifacts.
- Existing mmap support is a local backend option, not a memory limit. A
  Windows experiment mapped the V4.1 files but drove available RAM close to
  zero and did not reach a successful serving or inference result. No full
  model load should be used as a validation step until bounded residency has
  been implemented and measured.
- The Mesh-pinned llama.cpp runtime does not yet have validated DeepSeek-V4.1
  support for this artifact's architecture and compression metadata. Do not
  rename `deepseek41` to `deepseek4`, bypass model checks in production, or
  treat GGUF metadata parsing as proof of inference compatibility.
- The exact target has since passed Mesh tensor-inventory inspection,
  byte-preserving package verification, and package-only two-range artifact
  integrity checks. Those checks do not invoke the native model loader or
  execute a stage. See the target-specific
  [DeepSeek-V4.1 compatibility plan](DEEPSEEK41_COMPATIBILITY.md).
- The existing experimental MoE-aware assignments/session-sticky routing are
  not evidence that one generation is executing experts across nodes. They
  must not be presented as distributed expert parallelism.

## Target execution model

Use two orthogonal placement levels. The first is required; the second is a
later optimization and must not be simulated by model-replica routing.

### 1. Layer stages across nodes

1. Resolve one immutable model identity, including the ordered source
   filenames, sizes, hashes, tokenizer, and architecture metadata.
2. Convert the source GGUF set into a versioned Mesh layer package. Assign
   every tensor to its owning layer/stage, preserve shared tensors as explicit
   shared artifacts, and record exact per-file offsets, lengths, and digests.
3. Plan contiguous layer ranges using *per-stage* compute-memory budgets,
   context/KV requirements, available local storage, and observed peer
   bandwidth/latency. Do not reject a distributed plan merely because the sum
   of all model weights exceeds one node's RAM.
4. Each peer downloads or reuses only its immutable package artifacts, maps
   its own stage files, and runs that stage. Stage 0 owns input/tokenizer work;
   the final stage owns logits/sampling unless the runtime contract explicitly
   places these elsewhere.
5. Forward activations and required recurrent/KV sidebands through the
   existing authenticated Skippy stage transport. Keep the existing
   stage-ready barrier: stage 0 is routable only after every required stage is
   admitted and ready.

This path shares the model's layers, compute, and storage across nodes for one
request. It does not require expert weights to cross the network on every
token.

### 2. SSD-backed weights within each stage

Each node stores the files for its stage on its own local SSD/NVMe and keeps
only the metadata, runtime state, KV cache, and actively needed weight pages
resident. The loader must:

- open and validate metadata and tensor ranges without eagerly reading all
  weight payloads;
- map or read stage-owned tensors on demand using a backend-supported path;
- make cold reads, page faults, and any prefetch explicit and observable;
- enforce a configurable, measured resident-weight budget with backpressure
  or request rejection before the OS is put under memory pressure;
- account separately for runtime/graph buffers, KV cache, activations, pinned
  pages, and weight-cache residency;
- never promise that ordinary `mmap` by itself limits working set or
  guarantees low-RAM operation.

Memory admission is per node and per stage, not total-model-size versus
physical-RAM. The planner must still reject a stage when its measured minimum
working set, runtime overhead, and requested KV/context allocation exceed that
node's safe budget.

Use the existing model-package and native stage-admission machinery as the
integrity boundary. Do not introduce a second unverified weight distribution
path in the telemetry plugin.

### 3. Optional expert parallelism within a MoE stage

After layer staging and bounded local SSD loading work, add an explicit
expert-parallel topology for models whose experts can be dispatched
independently:

1. One stage computes the router/top-k decision and assigns each selected
   expert to an owning expert worker.
2. It sends only selected token vectors, sequence/request identifiers, and
   routing weights to those workers over an authenticated, versioned Mesh
   stream.
3. Workers load their assigned expert tensors from their local SSD within a
   bounded cache, compute the expert outputs, and return them.
4. The owning stage combines outputs in the model-defined order and continues
   the same generation.

This requires graph/runtime support, a wire protocol, scheduling and failure
semantics, and model-family validation. Measure network bytes and per-token
latency; remote expert execution is not automatically faster than local SSD
access. Do not ship this mode until a real multi-node benchmark proves its
benefit. Ordinary session-sticky routing to a node that has a model copy is a
separate replica-routing behavior.

## Protocol and resource contract

Keep the plugin API advisory. Resource facts that affect placement must be
collected and interpreted by the Mesh host/runtime, using the existing peer
identity and authenticated control plane. A plugin may supply a node-local
inventory, but cannot self-assign work or silently change the host's stage
plan.

Advertise bounded, versioned stage capabilities rather than a misleading
cluster-wide memory sum:

- free/total physical RAM and a configurable safe inference budget;
- storage capacity, filesystem/volume identity, free bytes, and whether the
  volume is eligible for the configured local model path (do not infer SSD
  versus HDD from fixed/removable drive type);
- stage model/package digests already present and local artifact sizes;
- CPU/GPU backend capabilities and allocatable compute memory;
- supported architecture, quantization, KV/cache modes, and native runtime
  build/ABI;
- optional measured local-storage throughput and peer link latency/bandwidth,
  with timestamp and expiry.

Planning must reserve memory for runtime and KV before allocating a stage
weight-cache budget. Resource snapshots are hints, not a reservation: perform
an atomic local admission check when a stage is loaded, publish readiness only
after the native loader confirms the exact admitted tensor closure, and
withdraw/replan on resource loss. Never advertise the sum of peer RAM as
usable RAM available to one process.

## Model compatibility and artifact preparation

Before V4.1 can use this path:

1. Implement and validate the target-specific native runtime path described
   in the [DeepSeek-V4.1 compatibility plan](DEEPSEEK41_COMPATIBILITY.md):
   metadata, tensor layout, graph, state/KV, and stage-boundary semantics.
   Preserve V4.1's own compression ratios and inter-layer dependencies.
2. Build a converter/package materializer that turns the seven original GGUF
   files into content-addressed layer artifacts without loading all weight
   payloads into RAM. Prove every source tensor appears exactly once or is
   explicitly declared shared; record source-to-package tensor identity and
   hashes.
3. Ensure stage boundaries carry every V4.1 dependency (including any
   compressed-attention and recurrent side state). A layer-number range alone
   is not sufficient evidence.
4. Add synthetic graph-planning and stage-contract tests first, then certify
   the immutable real artifact with small, bounded, non-generative loader
   tests before attempting inference.
5. Update the split-family certification roster only after stage correctness,
   state mobility, and context/KV behavior have passed the family's
   certification gates.

The existing Flash-MoE adapter may remain an optional single-node backend.
Making it a distributed stage backend would require an explicit stage ABI and
artifact contract; neither is assumed by this design.

## Delivery phases

### Phase A — safe, measurable planning

- Extend node capability snapshots with safe memory budget, stage-artifact
  inventory, and native backend/family capabilities.
- Keep snapshots advisory; test expiry, stale data, peer departure, and local
  admission races.
- Add planner tests proving the sum of multiple node budgets can admit a
  multi-stage plan while every individual stage remains within its owner's
  budget.
- Keep existing behavior unchanged for models without the new capability.

### Phase B — package-backed layer stages

- Materialize and verify stage-owned artifacts from a multi-file GGUF source
  without assembling a full in-memory model.
- Reuse Skippy package identity, integrity, admission, readiness, and
  activation transport.
- Prove on two machines that each peer only reads its assigned stage weights
  and that one request crosses the stage boundary successfully.

### Phase C — bounded local SSD weight access

- Implement demand-backed tensor loading and a real resident-weight budget in
  the native runtime; instrument reads, page faults, evictions, cache size,
  load time, and memory high-water mark.
- Add cancellation, backpressure, and clear admission errors when SSD
  throughput or the memory budget cannot sustain a request.
- Test cold and warm reads on the target OS. Do not use only a sparse mapping,
  virtual address reservation, or `mmap=true` as passing evidence.

### Phase D — DeepSeek-V4.1

- Implement and certify the architecture and the source-to-layer package
  transformation for the exact V4.1 Q2_K artifact.
- Start with metadata/tensor-closure validation, then one-stage bounded load,
  then a tiny-context inference smoke, then two-node staged inference.
- Run on a canary with a hard stop threshold for free RAM and process working
  set. Never repeat the earlier uncontrolled full-model load.

### Phase E — optional cross-node expert workers

- Add expert dispatch only after the target family and SSD stage path are
  correct.
- Prove dispatch/output parity, request isolation, cancellation, peer loss,
  cache pressure behavior, and network overhead before enabling it by default.

## Acceptance criteria

The goal is complete only when all applicable criteria pass on the target
artifact and hardware:

1. **Integrity:** packaged tensor inventory exactly matches source metadata;
   every artifact is content-addressed and verified before load.
2. **No hidden full copy:** package conversion, stage startup, and normal
   inference do not create a complete in-RAM copy of the source model.
3. **Per-node bounds:** measured peak resident memory stays below the configured
   safe budget on every node, including KV, runtime overhead, and weight
   caching; violating the budget rejects or backpressures the request rather
   than exhausting system memory.
4. **SSD evidence:** cold-start logs/counters show stage-owned weight bytes
   read from local storage on demand; report cold/warm startup, read
   throughput, page/cache residency, and first-token latency.
5. **One distributed request:** two or more nodes execute one request across
   their assigned layer stages and return a correct completion; this is not
   satisfied by choosing a different replica for each session.
6. **Failure safety:** unavailable peers, insufficient storage, incompatible
   runtime/architecture, stale telemetry, checksum mismatch, and exhausted
   budgets fail explicitly without publishing a false-ready stage.
7. **V4.1 certification:** the exact `deepseek41` artifact passes architecture,
   graph, state/KV, stage-boundary, and bounded inference tests. Metadata
   parsing or a successful `--dry-run` alone is insufficient.
8. **Performance disclosure:** publish hardware, topology, context, cache
   settings, cold/warm latency, throughput, disk/network traffic, and memory
   high-water mark. No speedup is implied by using more nodes or SSD.

## Non-goals

- Combining RAM into a cross-machine shared-memory address space.
- Treating aggregate free RAM or SSD capacity as allocatable by one process.
- Replacing Skippy scheduling with plugin-controlled placement.
- Claiming that the current `mesh-ssd-node` or Flash-MoE adapter implements
  distributed SSD inference.
- Starting the full 246-GiB model on a 16-GiB Windows host as a test before
  bounded residency is implemented.

## Measured stage footprint

Analytic accounting through the exported Skuppy ABI (`skippy_model_info_*`, metadata only, no
weight payload read) over the 46-artifact V4.1 package:

| quantity | value |
|---|---|
| package total | 246.344 GiB, 1046 tensors |
| one normal layer | 4.977 GiB = **58.7 MiB dense** + **4.581 GiB fused experts** (25 tensors) |
| engram layers 1 and 14 | 37.29 GB each = **32.37 GB dense** + 4.581 GiB experts |
| one expert, one layer | **12.2 MiB** |
| the six active experts, one layer | **73.3 MiB** |
| embedding / output / final norm | 217.2 MB / 543.0 MB / 20 KB |

Consequences for placement:

1. Steady state per owned layer is about **56 MB of resident dense weights plus 73.3 MiB of
   streamed experts**.
2. A single-stage 40-layer graph must fault in `73.3 MiB x 40 = 2.86 GiB` of expert pages inside
   one compute, because Windows cannot release a mapped page mid-graph. This is the observed
   source of the ~4.6 GiB working-set plateau. Layer staging divides that term by the number of
   stages, which is what makes small nodes viable at all.
3. Mapping the whole 264 GB costs on the order of **0.5 GiB of page tables alone**, and that cost
   scales with *mapped* bytes rather than with owned layers. A stage that maps only its own
   artifacts pays proportionally less, which is a second, independent argument for layer staging.
4. The engram layers stay affordable only because `engram_embd` is `TENSOR_READ_LAZY` and is
   gathered row-wise; 32 GB of dense table is never resident. This must be re-verified on ARM,
   whose page cache is smaller.

Derived steady-state capacity, assuming a reserved runtime/KV budget and using the numbers above:

| node | usable after reserves | layers |
|---|---|---|
| 2 GB class | ~800 MB | ~5-6 |
| 4 GB class | ~2.5 GB | ~18-19 |

Nominal RAM is never the budget: on these devices the OS and platform take a substantial share
first, which is what `mesh-llm-system::capacity::AdvertisedMemory` already models.

## Stage ranges need paired block boundaries, and the port was not emitting them

`skippy_stage_planner_realize_v1` rejected **every** range tried — including the full `[0,40)` —
with:

```
plan: execution profile decode rejected: state inference requires an exact graph
      and paired block boundaries
```

**Retraction.** An earlier revision of this section called that message a possible artifact of a
catalog built with the wrong offsets. It is not, and the catalog was never the cause. The message
comes from `skippy_infer_state_effects` (`src/skippy/state_effects.cpp`), which requires a
non-empty, even-sized block-boundary vector, and `llama-graph.cpp` only produces one through
`begin_block` / `end_block` calls made by each model's own graph builder. All 139 other builders in
`src/models` make that pair; the V4.1 port did not. `res->block_boundaries` was therefore empty and
every range was rejected before any tensor locator was consulted. Patch 0061 adds the calls, and the
range that failed now plans.

So a contiguous layer range is not inherently inadmissible and there is no boundary to hunt for: a
range works once its builder records block boundaries the way every other architecture does. Fixing
this also exposed two further omissions in the same port — a missing `request_input` identity on the
engram rows leaf, and destination-only geometry notes on the engram's normalization reshapes. All
three are recorded in patch 0061.

## Package tensor binding

The planner validates every catalog entry against the physical GGUF descriptor
(`src/skippy/package_tensor_binding.cpp`): the tensor type and dimensions must match, with
trailing ones permitted. The catalog the runtime feeds it is built from the **package manifest
plus its metadata carrier** (`skippy.package.tensor_part` / `tensor_offset` / `tensor_size`), where
each tensor's `split_no` is its **package artifact index** and `data_offset` is the offset *within
that artifact*.

Two out-of-process attempts bracket the requirement:

- a catalog built from the **metadata carrier** passes binding (its type and dimensions are the
  physical ones) but realises no range, because its offsets are package-relative rather than
  resolved locators;
- a catalog built from the **original seven container shards** fails binding on the first sorted
  tensor, because the raw-shard layout is a different coordinate system from the package layout.

Any out-of-process measurement harness must therefore resolve through the package carrier
(`crates/skippy-model/src/package_carrier.rs`), which is where the manifest and the carrier are
combined into locators. Reconstructing that merge by hand is what both attempts above got wrong.

## Where expert-level splitting would attach

Recorded while the mechanism is still understood, so a later decision does not have to re-derive
it. The current chain is:

- `llama_model_loader_stage_selection { int32_t layer_start, layer_end; unordered_set<string>
  resident_tensor_names; }` — stage identity is a contiguous layer range plus a whole-tensor
  allow-list. There is no expert-range or row-range concept anywhere in the runtime.
- Stage identity in the protocol is likewise `layer_start` / `layer_end` plus `resident_tensor_ids`
  (`StageAdmissionDescriptor`, `StageTopologyStageDescriptor`, `LayerRange`).
- The `ExpertGroup` / `ExpertProjection` machinery in `crates/skippy-model/src/gguf_writer.rs` is
  converter-side **fusion** (per-expert source tensors into one fused GGUF tensor), not splitting.
- The deleted "legacy expert split serving" belonged to the removed llama.cpp `rpc-server` path.

Expert-level splitting would therefore attach at `stage_selection` (an expert-range field alongside
`resident_tensor_names`), in the dispatch wire protocol, and in the state semantics — not in the
package format, provided per-expert row ranges are recorded now (see the preservation measures in
the edge-cluster plan).

## Measured: layers [0,1) plan, open, and prefill

First end-to-end measurement of a real V4.1 stage through the exported ABI. Host: 15.69 GB Windows,
model on a local SSD, `ctx = 512`.

| step | result |
|---|---|
| catalog | 1046 tensors, 246.344 GiB, built from the seven container shards |
| plan | accepted; 4 profiles — `decode`, `prefill`, and an `embeddings:` variant of each |
| selected profile | `decode`, `imports = 0`, `exports = 2` (source stage, not terminal) |
| resident closure | 26 tensors, **4.838 GiB** |
| stage graph | **667 nodes** — the layer slice, not the 33472-node whole-model graph |
| compute buffer | 4.22 MiB |
| RSS after open | +115.2 MiB |
| RSS after session create | +0.0 MiB |
| **peak RSS across a real prefill** | **0.340 GiB** against a 4.838 GiB map |

Three things follow.

1. **The graph really is sliced.** `sched_reserve` takes its graph from the installed stage program,
   and with one installed the builder emits only the staged layers. Layer staging bounds the compute
   graph, not merely residency.
2. **The quantity that decides whether a node can host a stage is the working set, not the closure.**
   MoE sparsity means one token touches the embedding, one layer of dense weights, and six experts —
   about 0.35 GiB of a 4.838 GiB map, the same order the analytic accounting above predicts. That is
   much better news for 2 GB-class devices than a closure-sized budget implies.
3. **It is a per-step figure, not a steady-state bound.** Pages touched by one token stay resident,
   later tokens activate different experts, and the resident set therefore grows toward the closure
   unless it is evicted. This is the job of the residency trim, and it is why the number above is a
   floor rather than a capacity claim.

`skippy_prefill_chunk` returned OK having produced `0` output bytes. Whether a source stage delivers
its activation payload through that parameter or only through
`skippy_session_copy_output_activation_frame` is **not established**: the stage computed, but it is
not yet shown to hand its export to a successor stage.

Also unresolved: the activation-frontier requirement. `skippy_finish_model_open` insists that
`is_source_stage(config) == (layer_start == 0)` and `is_terminal_stage(config) == (layer_end ==
layer_count)`, and those predicates are defined as "import count is zero" and "export count is
zero". A default-constructed config has both counts zero, so it always claims to be terminal; a
range that stops short of the last layer has to carry the plan's exported frontier explicitly.

## An un-sliced reserve is what makes a failed stage program fatal

`MESH_LLAMA_WS_BUDGET_MB` (`llama_residency_trim_if_over_budget`, patch 0059) bounds residency only
inside `llama_context::graph_compute`. Nothing before compute is covered, and in particular
`sched_reserve` reserves a graph selected through `active_stage_program()`:

- **program installed** — the builder emits the staged range, so the reserve is small and cheap:
  `[0,1)` reserved 667 nodes and a 4.22 MiB compute buffer in 296 ms.
- **no installed program** — the builder emits every layer instead, and `graph_reserve` allocates
  real buffers for all of it, because `no_alloc` is false outside the `resolve_fused_ops` probe
  (`llama-context.cpp`).

The guard that already sat at the top of `sched_reserve` throws only when a program *is* installed
and its computation generation moved. A **failed installation therefore walks straight into the
whole-model reserve**, and on a large model that allocation attempt is what ends the process: the
real diagnostic — here a geometry-recipe error — never gets reported.

This was measured the hard way. `[0,2)` fails its stage program install, and every attempt froze the
host rather than returning that error.

**A guard here was tried and rejected.** Refusing the reserve whenever a stage plan has no program
looks like the obvious fix and is wrong: the first reserve of a stage-plan runtime happens while the
context is being created, and the program is extracted *afterwards*, so a null program is the normal
state at that point, not an error. That version rejected every real range and every run of the
synthetic tiny model. Patch 0062 records the rejection instead of the guard, so the attempt is not
repeated.

Where the un-sliced reserve actually costs a host its life is therefore still open: it is not "a
program is missing", and it is probably not reached at all once a stage program installs, because
the install is what selects the sliced graph. Confirming it needs an instrumented reserve on a host
that can survive losing.

Operational consequence for the ARM target: a stage node's reserve cost is bounded **only while its
stage program installs**. A node that cannot install one falls back to a whole-model allocation it
can never satisfy. The fix belongs on the install path, not on the reserve.

## The verification vehicle: a synthetic tiny model

Every structural failure described above costs about five minutes and a whole-model graph reserve on
the real artifact, and froze the development host repeatedly. None of it needs the real dimensions:
the geometry recipes, block boundaries, request-input identities and proof rules depend on which
reshapes and which leaves the graph contains, not on how wide the tensors are.

`tools/v41-mktiny.cpp` writes a synthetic `deepseek41` GGUF at toy dimensions -- 57 tensors, 61 keys,
**393 KB** -- preserving every relation the loader validates (`hc_dim = hc_mult * n_embd`,
`hc_mix_dim = (2 + hc_mult) * hc_mult`, the expert and engram tensor expressions) and putting an
engram layer at index 1 with `compress_ratios = {0, 0}`, which is what the real model has for layers
0 and 1. It loads through the real loader, builds the real graph and runs the real stage-program
extraction, in seconds.

It reproduces the real failures at a different scale, which is the only reason they were fixable:

| real model | tiny model |
|---|---|
| `5120x4 -> 20480`, `blk.1.engram_k.weight` | `64x4 -> 256`, `blk.1.engram_k.weight` |
| ~5 minutes per attempt, host at risk | ~40 seconds, no measurable cost |

It is also a complete working vehicle, not only a reproducer. A source stage over layer 0 alone runs
the whole path end to end in about a second:

```
PLAN-OK   residents=26
OPEN-OK   rss_open=+5.9 MiB  session=OK
PREFILL OK  tokens=1  peak_rss=0.081 GiB
```

Every change to the staging path can therefore be validated through plan, install, open, session and
compute before it ever touches the real artifact.

Working against it moved the engram install through three distinct failures in one session: a gate
weight reshape that needed a static geometry note (patch 0063), a request input with no owned recipe
that needed a stage-program input kind (patch 0064), and now a RESHAPE whose recorded geometry
disagrees with the tensor inside a CONCAT chain along the hc axis.

Any further port work on this model should go through the tiny model first. Reach for the real
artifact only to confirm.

## The split contract is valid at the plan level

`tools/v41-chain.cpp` realizes several ranges over one model and runs
`skippy_stage_plan_validate_chain_v1` on them. That call works on realized plans, so it answers the
first half of the multi-node question without needing a stage program to install, a whole model in
memory, or a second machine.

Splitting the tiny model at the layer-1 boundary:

```
RANGE [0,1) -> plan [0,1) of 2, profiles=4 residents=26   imports=0 exports=2 inputs=4 states=2
RANGE [1,2) -> plan [1,2) of 2, profiles=4 residents=33   imports=2 exports=0 inputs=5 states=2
CHAIN OK
```

and the frontier matches by identity, not merely by count:

| `[0,1)` exports | `[1,2)` imports |
|---|---|
| `skippy-port:v1:08dbc688…` binding=`hc_mixes-0` | `skippy-port:v1:08dbc688…` binding=`hc_mixes-0` |
| `skippy-port:v1:3c6cf5ec…` binding=`l_last-0` | `skippy-port:v1:3c6cf5ec…` binding=`l_last-0` |

So contiguous layer ranges tile cleanly and the activation frontier a stage must hand over is
exactly the one its successor must accept. **The split contract is not the risk it looked like.**

What this does *not* establish is the runtime half: that a stage's export is actually delivered to
its successor. That needs both programs installed and is still gated by the geometry work above.

### The producer half of the runtime handoff is now proven

`skippy_prefill_chunk` returns `produced = 0` even for a source stage that declares two exports,
which looked like a stage failing to emit anything. It is not. The export is not delivered through
that parameter at all; it is retrieved with `skippy_session_copy_output_activation_frame`, and on
the tiny model it comes back complete:

```
FRAME OK  parts=2 payload_declared=1120 payload_copied=1120 tokens=1 seqs=1 layers=[0,1)
  part 0 rank=2 token_axis=1 dims=24x1x1x1 payload=96   id=08dbc688976d004b…52c2c5dcf
  part 1 rank=3 token_axis=2 dims=64x4x1x1 payload=1024 id=3c6cf5eceb79a6a6…77c246de6
```

Those two part identities are **byte for byte** the activation export identities the plan declared
for the same range:

| frame part | plan export |
|---|---|
| `08dbc688…52c2c5dcf` | `skippy-port:v1:08dbc688…52c2c5dcf`, binding `hc_mixes-0` |
| `3c6cf5ec…77c246de6` | `skippy-port:v1:3c6cf5ec…77c246de6`, binding `l_last-0` |

Shapes agree too: 24 is `hc_mix_dim`, and `64x4` is `(n_embd, hc, n_tokens)` with one token.

So a stage emits exactly the frontier its plan promised, identified the same way, with the token axis
declared per part. A successor can therefore verify it received precisely what its own plan requires
-- which is the property a multi-node handoff depends on. What remains unproven is the receiving
side, and that is gated by the layer-1 install blocker rather than by anything about the protocol.

Two details worth carrying forward. The engram input belongs to the second stage, so the current
install blocker gates `[1,2)` and not `[0,1)` -- a source stage over layer 0 alone has no engram and
should install exactly as the real model's `[0,1)` did. And the second stage carries five request
inputs against the first stage's four: the extra one is the engram rows leaf, which is the input kind
patch 0064 added.

## Resolved: the RESHAPE at layer 1's entry was a self-inflicted annotation bug

Installing a program for any range that reached into layer 1 failed in the proof replay
(`stage_program_replay.cpp`, `GGML_OP_RESHAPE`), which compares the operation's recorded target
dimensions against the tensor it is actually replayed on:

```
stage program RESHAPE changes element count for l_last-0 (reshaped)
(op/674/RESHAPE source=op/658/CONCAT name=l_last-0):
source=512 dims=64x4x2x1 target=64x4x4x1
```

Reading the message against its format string: the recorded target is `(64, 4, 4)` -- 1024 elements --
while the source tensor is `(64, 4, 2)`, 512 elements. The recorded geometry claimed a third axis of 4
where the graph has 2.

**The root cause, and it was introduced by patch 0061.** The engram's `grouped_norm` has two callers
whose sources have different ranks:

```cpp
grouped_norm(key, ...)   // key is (hc_dim, nt)              -- flat, must be split into copies
grouped_norm(x,   ...)   // x is the layer input (n_embd, hc, nt) -- already split
```

0061 annotated the first reshape inside it as `dsv4_note_source_dimension(t, 2, 0, 1)`, meaning "dim 2
comes from the source's axis 1". That is correct for the flat key, whose axis 1 is the token axis. For
the layer input, axis 1 is `hc`, so the recipe recorded `hc` = 4 where the graph has the token axis.
**`hc` being 4 is why the two numbers read like "hc against nt"** -- the leading hypothesis recorded
below was right, and the culprit was the patch that fixed the three other annotations.

Patch 0068 annotates by source rank: the token-axis note when the source is already split, the
source-relative note when it is flat. Verified on the synthetic tiny model, where this had blocked
every range above layer 0:

```
[0,2)  PLAN-OK  OPEN-OK  PREFILL OK    (the range containing the engram layer)
[1,2)  PLAN-OK  OPEN-OK                (pure import stage)
[0,1)  unchanged
chain [0,1) + [1,2)  CHAIN OK
```

What the earlier isolation work established still stands and was worth having: the failure was
intrinsic to layer 1's entry rather than to the frontier, the split point or the first stage's
exports, so it was never a split-contract problem. Realizing `[1,2)` alone reproduced it identically
with the source being the imported frontier leaf. That is what pointed at the layer rather than at
the handoff.

**The consequence is worth more than the fix.** `[1,2)` now reaches the point of asserting its own
contract:

```
PREFILL FAILED  err=non-first runtime slices require activation input
```

That is the *consuming* half of the runtime handoff becoming reachable for the first time. It fails
only because the harness feeds it no frontier, so feeding it the frame `[0,1)` produces completes the
last unproven piece of the split contract.

## Reproducing any of this

The patch queue `0001..0068` in `third_party/llama.cpp/patches` reproduces the working tree exactly
(`git am` onto the port baseline yields tree `4ed867be`). The measurement harness is out of tree, at
`build/v41-port/tools`, with `SAFETY.md` beside it; read that first.

Two hard-won operational notes belong here too, because both cost time to diagnose:

- A run that dies inside a single long agent tool call is not necessarily a crash. This harness kills
  a tool call's job runner on a timer, and the job object takes the child process with it.
- Post-hoc free-memory readings prove nothing about safety. Runs that froze this host reported a
  healthy 9.6 GiB free immediately afterwards. The only sound rule is to size the host for the
  reserve, not for the closure, and to grow a range by one layer at a time.
