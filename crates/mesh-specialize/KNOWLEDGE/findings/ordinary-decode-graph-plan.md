# Ordinary one-row decode preparation and graph replay

Status: source-backed implementation proposal, 2026-09-27. Parent approval and
implementation are pending. No build, GPU execution, tracing, sanitizer run, SSH,
service change, or performance measurement was performed for this document.

The first implementation should be a prepared one-row projection API with checked
borrowed device views and a single explicit stream lease. Exercise it on the
existing MLP chain before extending it through a GDN block. Keep every kernel,
quantization boundary, diagnostic write, and input scale unchanged. The useful
first result is exact eager execution with stable storage and no operator-local
waits. Calling the existing model loop inside `begin_capture` cannot work.

This assignment is whole-model preparation. It does not replace the existing F08
driver graph implementation or reimplement GPU greedy selection. MTP capture is
excluded until ordinary replay has passed its own gates.

## Source identity and current controls

The inspected workspace is `/Users/ndizazzo/.codex/worktrees/ninfer-performance/mesh-llm`.
Its HEAD was `4a30a84074784395a3b3c4d8df74e1fb2f16d906`; the CUDA source directory
had no reported working-tree changes at inspection. References below beginning
`M:` are relative to `crates/mesh-specialize` at this pin. Use the
[mesh source root](https://github.com/Mesh-LLM/mesh-llm/tree/4a30a84074784395a3b3c4d8df74e1fb2f16d906/crates/mesh-specialize).
Ninfer checkout `target/specialize/ninfer-deep-dive` was verified at
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`; `N:` references use that pin.
The source pin does not identify the historical installed Ninfer executable.

A follow-up provenance check used read-only `git show` for all 28 referenced Rust
source files. Twenty-seven still matched the pinned blobs exactly. The working
`resident_nvfp4.rs` had acquired parent-owned `prefix` storage and a post-sync
`nvfp4_projection_audit::compare` call. Its cited spans were rechecked against
the pinned blob and contain none of those additions. The parent also changed
`resident_model_profile.rs`, which this document does not cite as pinned source.
Several range endpoints that extended beyond a file were tightened to its actual
end. These parent diagnostic changes are excluded from the baseline source claim.

The [runtime deep dive](ninfer-runtime-deep-dive.md) is the starting analysis,
but two assumptions there are now superseded. `model_workspace.rs:8-56` supplies
an opt-in model-shared MLP arena, gated to exact FP8 and split-K off.
`resident_model.rs:142-153,184-225,383-397` now has opt-in GPU selection for ordinary
`forward_selected`. Diagnostic forwards still download full logits. The current
[knowledge index](../README.md) records device/resident and sanitizer qualification
for GPU selection; the opening of its original worker report predates integration.
Neither feature makes the complete model capture-ready.

Preserve the baseline as an explicit control: same artifact/tensor hashes,
config/capacity, initial complete state, token sequence, one-row shape, exact FP8,
split-K off, exact attention, existing one-row NVFP4 integer decode, same epsilon,
RoPE tables and BF16 boundaries. Record all environment profile settings and PTX
hashes. Compare workspace off/on separately from selection CPU/GPU, then compare
prepared eager with prepared graph. Do not combine a new A16 head, native FP8,
online attention, quantization sharing, or diagnostic-write removal with this change.

The current [plan](../../PLAN.md) records an unresolved multi-row versus one-row
NVFP4 down-projection discrepancy after online-attention inputs. That does not
justify changing graph arithmetic or relaxing checks. Require exact output/state
agreement between **identical one-row schedules and inputs**. Prior short fixtures
do not establish universal prefill/decode partition equality. Compare against the
existing shape-specific eager baseline and report any pre-existing partition
failure independently. Parent owns that arithmetic investigation.

## What the current call chain prevents

This inventory covers the ordinary non-recorded model path. Every listed owning
scratch `Buffer` eventually invokes `cuMemFree_v2`, including diagnostic values
explicitly dropped before the caller returns. Replacing allocation while leaving
those drops in capture is insufficient.

| Owner and exact source span, all under `M:src/kernels/cuda/` | Present work that must leave the captured body | Prepared interface/change |
| --- | --- | --- |
| `resident_model.rs:285-408` | Embedding returns owning buffers; each block replaces and frees hidden; observers and partition-stage audits can download; head/selection completes before cursor commit | Separate `prepare_ordinary` and `enqueue_ordinary`; reject observers, recording, all-row logits and stage auditing in this lane. Retain existing diagnostic forward |
| `resident_embedding.rs:52-108` | Host token serialization; alloc/upload IDs; residual, normalized and FP32 output allocation; function lookup/default launch/context wait | Stable ID ingress plus caller-owned residual/normalized/raw views. Preserve otherwise unused normalization and raw writes initially |
| `resident_norm.rs:55-115,119-175,182-216` | Row-ID serialization/allocation/upload in `run`; residual/normalized/raw allocation; final residual-add output allocation; every launch waits | Preupload row ID zero; `enqueue_norm`, `enqueue_residual_norm`, `enqueue_residual_add` use explicit outputs and prepared functions |
| `resident_fp8.rs:74-181` | Profile/shape dispatch, quantize/linear lookup; codes/scales/BF16/raw allocation; two default-stream launches and wait/error drain | Frozen exact one-row projection descriptor and quantize/linear enqueue. Retain independent scales for every projection |
| `resident_nvfp4.rs:81-155,226-239` | Packed/scales/effective/output/raw allocations; quantize/linear lookup and default launches; sync/error drain; one-row/multi-row arithmetic dispatch | Frozen one-row descriptor with distinct `input_scale` and factor per projection. Caller supplies all five regions |
| `resident_bf16.rs:43-96` | A/B output/raw allocation, lookup/default launch/wait | BF16 enqueue into explicit output/raw views |
| `resident_conv.rs:40-99` | Next history, convolution output, FP32 convolution and SiLU allocations; default launch/wait; synchronous state D2D copy | Preserve separate next-history region; append ordered copy kernel or checked async D2D on the same stream after convolution |
| `resident_gdn_core.rs:256-430,439-484` | Q/K, beta/g/decay, recurrent BF16/raw, gated BF16/normalized/weighted/SiLU/raw allocation; per-stage lookup/default launch/sync; optional recorded delta allocation | Prepared QK/gates/recurrent/gated-norm functions with views. `record=false` is fixed; recurrent state pointer remains stable |
| `resident_attention.rs:149-245,258-265` | CPU RoPE generation per layer/step and cos/sin allocations/uploads; all ordinary projection/norm blockers | Generate identical host RoPE data before enqueue; upload to stable per-geometry input regions outside capture initially |
| `resident_attention_prepare.rs:67-151` | Values/normalized/raw/gate allocation, function lookup/default launch/wait for both Q and K; even K's unused gate region is allocated | Prepared Q/K descriptors and disjoint outputs, preserving the existing writes |
| `resident_attention_core.rs:46-112,124-255` | Attention output/raw allocation; append and attention function lookup/default launch/wait; `past` is passed by host value | Device-position entry variants of append/attention; keep original arithmetic body, exact masking, capacity checks and launch geometry |
| `resident_attention_gate.rs:11-48,81-115` | Output/sigmoid/activated/raw allocations, default launch/wait | Explicit output and all diagnostic views |
| `resident_mlp.rs:63-98`, `resident_mlp_workspace.rs:178-248`, `mlp_workspace_projection.rs:39-150` | Shared cache may allocate/replace on shape change; projection and activation function lookups/default launches; chain completes its lease; `copy_region` allocates and synchronously copies down output | Freeze row-one layout before execution. Reuse bindings but accept prepared functions, stream and a caller-owned lease. Return a down-output view used by residual-add before arena reuse |
| `resident_activation.rs:12-59` | Ordinary fallback allocates values/SiLU/activated/raw and waits | Same explicit regions as the MLP activation enqueue |
| `resident_head.rs:38-77` | Allocates/copies last hidden row, then norm and projection | Checked one-row hidden view; norm/head outputs in persistent regions; head weights retain current FP8 profile |
| `resident_greedy.rs:21-112` | Persistent scratch already exists, but `inspect` resolves both functions, default-launches, completes workspace and downloads 16 bytes | Split existing selector into prepare, enqueue, completed readback/validation. Preserve kernels and result ABI |
| `resident_state.rs:15-27,36-49,64-73` | Initialization/fork allocates and copies; convolution `copy_from` uses synchronous D2D | Initialization/fork remain outside prepared execution; checked state views and ordered convolution copy replace only per-step copy |

Alternative FP8 A16 and split-K branches allocate their own outputs/partials and
wait (`resident_fp8_a16.rs:9-57`, `resident_fp8.rs:200-247`). Admission excludes
them in version one; a later profile must provide its own layout, functions, key
and numerical qualification. This avoids hiding an unsupported branch behind an
environment setting. Host Vec allocation for launch arguments is not itself a
CUDA capture prohibition, but remove per-launch Vec/string construction from the
prepared fast path to isolate submission cost.

## Driver restrictions and ownership contract

The existing [driver capture implementation](https://github.com/Mesh-LLM/mesh-llm/blob/4a30a84074784395a3b3c4d8df74e1fb2f16d906/crates/mesh-specialize/src/kernels/cuda/driver_graph.rs#L60-L147)
mirrors thread-local capture restrictions and rejects nested capture.
`driver.rs:419-425,448-451,485-558,691-709,766-810,839-875` rejects context
synchronization, allocation, synchronous copies, function resolution, default
launches and profiling operations during capture. `Function::launch_on_stream`
already bypasses default-stream profiling and checks stream/module context.
Use it; do not make the guards permissive.

`Buffer::drop` at `driver.rs:565-577` directly frees without a capture guard.
`WorkspaceStep::complete` and error drop at `resident_workspace.rs:123-148` call
context synchronization. Neither is a valid capture-body scope. A short-lived
`DeviceRegion` whose raw address is saved into a graph is not lifetime proof.
The owner must prevent arena replacement for the complete graph lifetime.

Proposed concrete interfaces, names provisional for parent review:

- `DeviceView<'owner, 'ctx>` contains context identity, checked pointer, byte extent,
  alignment and read/write role, borrowed from buffer, state, weight or scratch
  owner. Constructors validate overflow and subrange bounds. No owning-buffer
  façade or raw-pointer-only public API. Current `DeviceRegion` is workspace-only;
  generalize the checked view contract in a narrowly named device-view module.
- `PreparedProjection<'module, 'weights, 'ctx>` borrows module functions and weight
  owner, and fixes dtype/profile, dimensions, grid/block, scales and scratch view
  requirements. `enqueue(&Stream, input, outputs)` performs no lookup/allocation/
  wait/readback. Equivalent descriptors belong in existing norm, conv, GDN,
  attention and selector modules, not a duplicate kernel implementation.
- `OrdinaryWorkspace` owns one finalized disjoint layout for rows=1, plus stable
  ingress/egress. A model-specific `OrdinaryPlan` owns immutable descriptors and
  borrows weights/module. A session-bound `PreparedOrdinary` binds stable state
  arena and workspace addresses. Graphs must not be cached merely by model shape.
- `OrdinaryStep` exclusively borrows session state, workspace and stream. Normal
  eager completion drains that stream once. On failure it marks session/workspace
  unusable and drains before releasing owners. A separate capture preparation
  scope ends/aborts capture before any drain; it does not open or commit a host
  sequence transaction. Do not run the existing workspace-drop synchronizer inside
  an active capture guard.

Pre-resolve every `Function`, validate all extents and state region names, and
freeze launch arguments before capture. Stack scalar/argument arrays need only
survive each `cuLaunchKernel` call, as documented by the existing driver. Device
addresses and module code survive capture, graph instantiation, all replays and
completion. Dynamic host scalar changes after capture do not update graph nodes.
Keep persistent device input addresses in nodes instead.

Destroy in this order after an explicit successful drain: executable graphs,
graph definitions, bound workspace/state, module and weights, then context.
A drain failure is not proof that owners are safe to recycle; fail the execution
context and retain resources until controlled context teardown. Log both original
and cleanup error. Stream destruction and graph destruction alone are not a
substitute for proving completion. Keep one host thread and one in-flight use per
prepared session in version one. No shared mutable scratch across sessions.

## Checked workspace and liveness plan

Use the existing 256-byte aligned `WorkspaceLayout::new` and checked region
indexing, `M:src/engine/workspace.rs:19-56` and
`M:src/kernels/cuda/resident_workspace.rs:94-120`. Initial plan assigns **distinct
regions to every named output below**, including diagnostics. This is deliberately
conservative. Source liveness below permits future reuse investigation, not
permission to alias kernel inputs/outputs. The first implementation should generate
the layout mechanically from descriptors and reject duplicate names, overflow,
misalignment, undercapacity and overlap before touching CUDA.

Notation: H hidden width, I MLP width, V vocabulary, C context capacity; GDN has
K key heads, G value heads, D head width and Q=(2K+G)D convolution channels;
attention has A query heads, B KV heads and E head width; R rotary dimension.
All rows equal one. Products, ceiling divisions and aligned sums require checked
arithmetic before conversion to u32/u64. Sizes are bytes.

| Named regions | Required sizes | First producer → last consumer; owner |
| --- | --- | --- |
| token / position / row-zero ID | 4 / 4 / 4 | Host ingress → embedding / KV append+attention / norm; persistent inputs |
| cos / sin for each distinct RoPE geometry | R each | Host exact table generator → both Q/K preparation; persistent ingress |
| hidden.in, hidden.out | 2H each | Embedding or previous block → post residual; final block → head. Distinct ping-pong regions, reused only after preceding block has consumed old input |
| embedding.normalized / raw | 2H / 4H | Embedding → diagnostic checkpoint only; retained writes |
| norm.residual / normalized / raw | 2H / 2H / 4H | Norm → QKV/Z/A/B or Q/K/V; residual/raw diagnostics. Norm uses constant row ID zero |
| each FP8 projection k→n: codes / scales / values / raw | k / 4 / 2n / 4n | Quantizer → projection; values → actual downstream consumers; raw → diagnostic checkpoint |
| each NVFP4 projection k→n: packed / scales / effective / values / raw | k/2 / k/16 / 4k/16 / 2n / 4n | Quantizer → projection; scale divisibility validated. Effective/raw writes retained |
| each BF16 projection k→n: values / raw | 2n / 4n | Projection → gates or other consumer; diagnostics retained |
| GDN qkv / z / a / b projection values | 2Q / 2GD / 2G / 2G | Projections → conv / gated norm / gates / gates; independent full projection scratch above also required |
| conv.next / values / conv.raw / silu.raw | 6Q / 2Q / 4Q / 4Q | Conv reads old 3-row history → state-copy after conv; values → QK prep and recurrent V; diagnostics retained |
| GDN q / k | 4KD each | QK prep → recurrence |
| beta / g / decay | 2G / 4G / 4G | Gates → recurrence for beta/decay; g diagnostics |
| recurrent.values / raw | 2GD / 4GD | Recurrence → gated norm; state written in place, no recorded delta in ordinary mode |
| gated.values / normalized / weighted / silu / raw | 2GD / 4GD / 2GD / 4GD / 4GD | Gated norm reads recurrent and z → out projection; diagnostics retained |
| attention q-linear / k-linear / v-linear values | 4AE / 2BE / 2BE | Q includes gate half; Q/K → preparation, V → append; full projection scratch required |
| prepared Q values / normalized / raw / gate | 2AE / 2AE / 4AE / 2AE | Preparation → attention and sigmoid gate; diagnostics retained |
| prepared K values / normalized / raw / gate | 2BE / 2BE / 4BE / 2BE | Preparation → KV append; unused gate remains distinct initially |
| attention.values / raw | 2AE / 4AE | Reads Q and initialized KV prefix → attention gate |
| attention gate.values / sigmoid / activated / raw | 2AE / 4AE / 2AE / 4AE | Gate reads attention and Q gate → out projection |
| post.residual / normalized / raw | 2H / 2H / 4H | Residual-norm reads hidden.in and branch → final residual / MLP / diagnostic |
| MLP gate/up/down scratch | Projection formulas for H→I, H→I, I→H | Gate/up values both survive through activation; down values survive final residual-add |
| MLP activation.values / silu / activated / raw | 2I / 4I / 2I / 4I | Activation → down projection; diagnostics retained |
| final norm scratch and head scratch | Norm formulas at H; FP8 H→V formulas | Final hidden → logits → selector. No last-row owning copy for one row |
| selector partials / result | 16 ceil(V/1024) / 16 | Tile reduction → finish → completed host status read; stable persistent regions |
| GDN state per layer: history / recurrent | 6Q / 4GD² | Persistent across steps; never scratch-alias or overwrite while read |
| attention state per layer: K / V | 2CBE each | Persistent append-only logical prefix; unused suffix must remain invisible |

The formulas are checked against current host allocation/extents definitions,
not against a new layout implementation or GPU trace. Specific sources are
`resident_embedding.rs:150-171`, `resident_gdn_core.rs:164-196,256-430`,
`resident_attention_prepare.rs:174-210`, `resident_attention_core.rs:151-190`,
`resident_attention_gate.rs:24-27`, and `mlp_workspace_projection.rs:22-38`.
The current shared MLP superset uses codes=k, scales=max(4,k/16),
effective=max(4,4k/16), values=2n, raw=4n per projection, plus 12I activation
bytes before alignment, `engine/workspace.rs:101-143`. Preserve its 19 disjoint
regions initially. Retained FP8 effective padding is not arithmetic data.

For version one, allocate one GDN scratch set and one attention scratch set sized
to the maxima across configured blocks, a common post/MLP set and a separate head
set. Do not overlay these sets. Sequential blocks may reuse the same named set
only after all its prior consumers have been enqueued on the same stream. No
recovery record, observer or external consumer may retain those views. Validate
all layer shapes against maxima. A scalar count of total workspace bytes is the
sum of each aligned region, not the sum of simultaneous live values. Calculate
and publish this exact plan before memory admission; no guessed fixed MB budget.

Convolution particularly needs two distinct history locations during its kernel.
Keep the existing next-history output followed by an ordered 6Q-byte state copy.
Do not point next-history at old history on the assumption that each thread owns
one channel. That requires a separate kernel-level race/liveness proof. Likewise,
no in-place norm, attention gate, or residual alias is admitted by this proposal.

## Dynamic input and attention topology

The current attention host grid is already independent of position:
`resident_attention_core.rs:163-189` gives append ceil(BE/256) CTAs and attention
A CTAs for one row. `past` appears in kernel parameters, not grid dimensions.
Exact and online profiles select different functions but share that host shape.
The exact device edit sites are `M:kernels/nvptx/causal_attention.rs:209-244`
for append and `277-334` for attention, including its `past + row + 1` loop
bound. The optional online counterpart is `attention_online.rs:388-432`;
it remains excluded from initial qualification. No device RoPE ABI change is
needed while cos/sin ingress stays host-generated.
A graph captured with the present ABI would keep writing the captured cache
position forever. Updating a Rust `past` variable cannot fix it.

Add narrowly scoped device-position entry variants for KV append and causal
attention. They read the same stable u32 position input; preserve the original
inner arithmetic/masking. Retain the scalar ABI for baseline tests. Check position
on the host before launch, and cover invalid device values with qualification
fault injection. An optional device bounds/status guard must make every stateful
consumer skip invalid work; a guard alone cannot protect downstream kernels that
ignore its status. Do not claim fault containment until that full contract exists.

For initial replay, compute RoPE with the existing host generator and upload the
same BF16 table into stable ingress before each replay. This removes allocation
and avoids inventing a new device trigonometry arithmetic profile. Synchronous
uploads outside capture are acceptable for the first experiment provided previous
stream work has completed. Measure them. Later alternatives are precomputed
capacity-bound tables with a device position gather, or pinned async ingress;
each needs independent byte equality and ownership tests.

Start with one graph per bound session and one validated interval `[0,C-1]` for
the current exact one-row attention topology, because its grid/kernel/scratch
contract does not change with position. A smaller tested interval may be admitted
first and must reject or safely fall back outside it **before** any state mutation.
Never create one graph per token. An eventual split/chunk attention schedule must
return a `TopologySignature` for every position interval: functions, grid/block,
shared bytes, scratch extents, split policy, masks and maximum visible keys. Merge
intervals only when signatures agree and kernels read actual position dynamically.

Ninfer's [ordinary profile planner](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/program/planning/graph_profiles.cpp#L56-L60)
uses measured topology thresholds. Its values 127/511/2047/4095/8197/16389/32767
are not our dispatch boundaries. Its
[ordinary body](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/program/decode.cpp#L24-L78)
uses stable ingress/egress, stream-ordered execution and GPU sampling. Its
[shared prepared-body helper](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/program/graph_execution.h#L11-L28)
and [workspace formulas](https://github.com/Neroued/ninfer/blob/e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d/src/models/qwen3_5/execution/workspace.h#L48-L122)
show the relevant separation. Adopt that separation in independent Rust code;
do not import Ninfer code or assume its arithmetic/performance transfers.

Bound the graph cache to one executable per admitted topology interval per session,
with an explicit maximum interval count and memory budget. Key the plan by device
UUID/context generation, module/PTX hash and build options, weight artifact identity,
all arithmetic profiles and split policies, model/layer geometry, KV dtype/layout,
rows=1/batch=1, capacity, interval and topology signature. Bind it additionally to
weight/state/workspace/ingress/egress base addresses, extents and owner generations.
Addresses alone are not identities after allocator reuse. Fork, reset with replacement,
resize, reload or profile change destroys/rebuilds the binding after drain. No graph
reuse across a new session just because its shape matches.

## Transaction, errors and output semantics

Keep the existing `Cursor` contract, `M:src/engine/session.rs:54-113`. Preflight
validates token<V, rows=1, context/module/owner identities, graph key, interval,
capacity and poison state before beginning the cursor transaction. Ingress upload
must finish before any consumer. Then begin one cursor transaction spanning the
whole enqueued model, final head, selector and successful completed readback.
Commit exactly past+1 only after valid finite output. Never advance on enqueue,
graph launch success alone, or selector token bytes without status validation.

The 16-byte selector result remains `[token,status,first_nonfinite,bf16_bits]`.
Preserve lowest-index ties, signed-zero behavior, all-negative values and rejection
of any NaN/Inf even when some finite winner exists. Validate status, token bounds,
nonfinite index and finite BF16 bits exactly as `resident_greedy.rs:97-112` does.
The graph must overwrite every partial and result on each run. Full logits/state
readback happens after completion in qualification, not within capture.

On any failure after transaction start, its drop poisons the cursor. Since GDN
and KV may already have changed, there is no retry on that session. Poison the
workspace and graph binding as well, drain the stream, preserve both errors and
require fresh/reconstructed state. Nonfinite selection follows this rule too;
successful kernel completion does not make partially advanced semantic state
reusable. A preflight rejection before transaction start leaves the session usable.

Capture preparation must not mutate the live cursor. Capture only records calls;
any eager warmup actually changes state and must use a disposable independently
initialized session or a verified state snapshot restored outside capture. On
capture failure, end/abort capture before lease cleanup and any synchronization.
Instantiate failure discards the graph without advancing live session state.
Never run an eager fallback after an uncertain partially executed replay. Fallback
is only allowed after a preflight miss with no submitted work.

## Ranked implementation and qualification sequence

1. Add checked device views, prepared exact FP8/NVFP4/BF16 descriptors, and a
   stream-aware exclusive execution lease. Adapt existing MLP bindings to enqueue
   into caller-owned views using pre-resolved functions on `launch_on_stream`.
   Preserve default APIs as controls. Validate exact one-row FP8 and NVFP4 MLP
   values, raw values, activation diagnostics and distinct projection scales.
   Inject failure after quantize, gate, activation and down; prove drain and poison.
   Test context mismatch, undersized and overlapping views, wrong dtype/alignment,
   overflow, repeated use and stable addresses. Run memcheck/racecheck/synccheck.
   This is the strongest first bounded coding assignment.
2. Prepare embedding/norm/residual, then one complete GDN layer including ordered
   convolution history copy. Compare every retained intermediate and complete
   history/recurrent state with the current one-row eager control, starting from
   zero and nonzero state. Repeat independent tokens and injected failures after
   convolution/copy/recurrence. No scratch reuse before its last consumer. Keep
   the exact same scope for sanitizer execution.
3. Prepare one attention layer and head, then all blocks. Initially eager only,
   with host scalar past and stable host-generated RoPE ingress. Compare exact
   BF16 hidden/logits and complete state at every step against one-row controls
   from identical prefills. Test fresh sessions, alternating independently owned
   sessions, capacity-1 and capacity rejection, and workspace re-creation. Retain
   diagnostic snapshots as bounded qualification output. No graph speed claim.
4. Split existing greedy host wrapper into enqueue and completed validation,
   preserving kernels. Complete entire ordinary eager round once on the explicit
   stream. Compare GPU result and complete state with CPU full-logit greedy;
   inject malformed status, NaN/Inf, ties, tail maxima and post-head failures.
   Establish traced zero steady-state CUDA alloc/free/function lookups and no
   operator-local synchronization. Host readback remains 16 bytes in timing lane.
5. Add device-position variants with unchanged arithmetic; keep ordinary eager
   execution as control. Test every admitted interval edge, position 0/1/C-1,
   changing input tokens with fixed addresses, and fresh-session stale-KV traps.
   Compare same-position RoPE bytes and all cache writes, not only sampled IDs.
   Reject unsupported profiles before transaction. Qualify invalid ingress without
   pretending host validation protects arbitrary device corruption.
6. Capture the very same prepared enqueue body using existing F08 wrappers.
   Verify capture leaves state/cursor untouched; instantiate and replay against
   independently initialized eager sessions. Test repeated replays, interval miss,
   destruction/rebinding, module/weight/state lifetime rejection, capture abort,
   instantiate failure, enqueue failure and replay completion failure. Exercise all
   three sanitizers on repeated multi-layer/model replay and cleanup. Publish exact
   outputs/state and poison/cursor assertions alongside every performance sample.

For each step, parent controls Just/Cargo slots and GPU use. These are proposed
checks, none run by this worker. Use existing independent operator references and
same-input checkpoints, not graph versus graph self-comparison. Use fixed prompts
and teacher-forced tokens across enough positions to expose growing-context errors;
free-running token agreement alone can hide state corruption.

Measure separate ablations with paired repeated uninstrumented runs: allocating
control; persistent storage with old waits; persistent one-stream eager; identical
prepared graph replay. Match GPU-selection mode and ingress/egress in the final
pair. Separately collect driver API alloc/free/lookup/copy/sync/launch counts and
bytes, CPU submission duration, stream span and graph preparation/instantiation
cost. Instrumented counts and event spans belong in diagnostic runs; default-stream
per-kernel profiling is incompatible with capture and cannot time the new stream.
Add explicit stream events outside capture or an external trace for the diagnostic
span. Report warmup, first replay, steady state, sample dispersion, clocks,
contention, GPU/toolchain/PTX/source identities and total memory.

No new traced counts exist here. The deep dive's historical 1,476 exact one-row
launches is an earlier measured profile, not a verified count of this prepared
path. The allocation table is source inventory, not driver timing. Never infer
host overhead by subtracting instrumented event sums from uninstrumented wall time,
and do not promise a percentage gain before the matched eager/graph experiment.

## Decisions retained for parent

Approve the first bounded view/stream/projection assignment before implementation.
Choose initial tested capacity/position interval and explicit graph/workspace memory
budget from the admitted model configuration. Decide whether ordered convolution
copy uses an independent Rust copy kernel or a narrowly added async D2D driver API;
the former keeps the initial graph body kernel-only. Start ingress/egress outside
capture unless there is a measured reason to add pinned async host transfers and
their longer host-pointer lifetimes. Choose ownership composition that makes bound
state and scratch impossible to replace while graphs exist; raw graph handles alone
do not encode those Rust borrows.

MTP remains outside this plan. Keep future recorded-state outputs and multi-row
shape descriptors separate so later draft/verify graphs can borrow the same checked
views. Acceptance, rollback, recovery and frontend committed-token boundaries need
their own host transaction and graph key. Do not capture current host acceptance
branches or route recorded execution through ordinary scratch reuse.

Coverage limits: inspected ordinary Rust orchestration, operator allocations,
state/workspace/driver lifetimes, selectors and Ninfer ordinary capture/profile/
workspace sources. No complete device instruction audit, live driver trace,
allocation count, code generation, numerical trial, memory-admission calculation,
long-context qualification, concurrency qualification or matched Ninfer benchmark
was performed. The document proposes independent Rust changes only. Parent's
NVFP4 arithmetic investigation and broader quality/provenance gates remain open.
