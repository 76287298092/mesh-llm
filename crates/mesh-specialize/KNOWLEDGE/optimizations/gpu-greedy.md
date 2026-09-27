# Exact BF16 GPU greedy reduction

Status: two standalone Rust NVPTX kernels and an independent CPU oracle are
authored. The host/model path is not integrated. The CPU source tests are
present but were not run by this worker. PTX compilation, GPU execution,
sanitizers, and performance are unmeasured; do not select this path for model
serving until the parent qualification is complete.

## Kernel ABI and launch contract

The device source exports two unmangled symbols:

```text
greedy_bf16_tiles(logits: *const u16, partials: *mut u32, vocabulary: u32)
greedy_bf16_finish(partials: *const u32, result: *mut u32, tile_count: u32)
```

Launch `greedy_bf16_tiles` with grid x `ceil(vocabulary / 1024)` and block x
128. Each CTA handles one contiguous logical tile, with each thread examining
eight indices spaced 128 apart. The supported host extent is 1 through 262,144
BF16 logits, so the tile count is 1 through 256. Launch `greedy_bf16_finish`
with grid x 1 and block x 128.

Each tile writes four `u32` words: winning global token index, the original
winning BF16 bits widened to `u32`, the lowest nonfinite input index in that
tile or `u32::MAX`, and reserved zero. A tile with no finite input writes
`u32::MAX` and zero for its winning pair. The final four words are winning
index, status (`0` finite, `1` at least one nonfinite), lowest nonfinite input
index or `u32::MAX`, and original winning BF16 bits widened to `u32`. When no
finite value exists, the winning pair is `u32::MAX` and zero. The host rejects
any nonzero status and must validate vocabulary, exact tile count, buffer sizes,
alignment, lifetime, and nonaliasing before launch.

The integer order transform canonicalizes both signed zeros, complements all
bits for negative finite values, and flips the sign bit for nonnegative finite
values. It packs the rank, inverted 18-bit index, and original 16 BF16 bits into
one unsigned 64-bit reduction key. Higher keys therefore represent larger
finite logits; equal values select the lowest token index. Finite BF16 values
are exact FP32 values, so this order agrees with direct FP32 comparison without
performing device floating-point arithmetic. Exponent-all-ones values are
excluded from winner selection and reduced separately with unsigned minimum to
report the lowest NaN or infinity index. The final raw BF16 bits remain carried
in the packed key, including the original sign of a selected zero.

## Independent CPU oracle and authored fixtures

`reference/greedy_bf16.rs` exposes `Selection { token, bits }`,
`GreedyError`, and `greedy(&[u16])`. It converts each value with
`f32::from_bits((bits as u32) << 16)`, rejects the first nonfinite value, and
uses a direct `>` comparison while retaining the first tie. It does not reuse
the device's integer key or rank transform.

Source tests cover every one of the 65,536 BF16 bit patterns as a single-value
selection/rejection, signed-zero ties and negative values, ties across a tile
boundary, a partial final tile, a maximum 262,144-token vocabulary, multiple
nonfinites across tiles, and empty/oversized extents. These tests are source
fixtures only; this worker did not run Cargo or a device build.

Proposed parent GPU qualification fixtures: exhaustive finite BF16 codes in
ascending and descending order; all-negative and all-positive vectors; both
signed-zero orders; positive and negative subnormals adjacent to zero; equal
maxima straddling indices 1023/1024 and the first/last vocabulary positions;
lengths 1, 127, 128, 129, 1023, 1024, 1025, 262143, and 262144; a maximum at
the final tail element; one and multiple NaNs/infinities with the lowest index
on a later tile; and an all-nonfinite vector. Compare winner index/bits and
status/index against this independent oracle plus the existing sampling result
and direct first-nonfinite scan. Run normal execution, memcheck, racecheck, and
synccheck before any model integration or speed claim.

## Complete per-assembly-site inventory for parent integration

The following enumerates every `asm!` site in
`kernels/nvptx/greedy_bf16.rs`. Add these operations to the shared
`KNOWLEDGE/asm-inventory.md` during parent integration; none is qualified by
this source-only implementation.

| Source site | PTX operation | Required target | Independent reference | Status |
| --- | --- | --- | --- | --- |
| `thread_and_tile` | Read `%tid.x` and `%ctaid.x` with `mov.u32` | NVPTX | Tile/thread ownership in the reduction oracle fixtures | Not compiled or qualified |
| `thread_index` | Read `%tid.x` with `mov.u32` | NVPTX | Finish-thread striding over tile records | Not compiled or qualified |
| `shuffle_down_u64` | Two full-mask `shfl.sync.down.b32` operations to exchange a packed 64-bit candidate key | NVPTX warp shuffle | Direct FP32 host comparison and first-index tie semantics | Not compiled or qualified |
| `shuffle_down_u32` | Full-mask `shfl.sync.down.b32` for minimum nonfinite index | NVPTX warp shuffle | Direct scan for lowest nonfinite input index | Not compiled or qualified |
| `shared_partials_base` | Declare 64-byte aligned CTA shared storage and obtain its address | NVPTX shared memory | Four warp-result records produced by each CTA | Not compiled or qualified |
| `publish_warp_partial_and_sync` | Predicate-store each warp's 64-bit candidate key and 32-bit nonfinite minimum, then CTA `bar.sync 0` in the same opaque assembly site | NVPTX shared memory and CTA barrier | Integer key from finite BF16 rank/token/source bits and direct first-nonfinite index scan | Not compiled or qualified |
| `load_shared_key` | Load one 64-bit warp candidate key with `ld.shared.b64` | NVPTX shared memory | Integer key from finite BF16 rank, token, and source bits | Not compiled or qualified |
| `load_shared_nonfinite` | Load one warp's minimum index with `ld.shared.u32` | NVPTX shared memory | Direct first-nonfinite index scan | Not compiled or qualified |

## Evidence and limits

- Source base revision: `0625c0af545c0650ff2e350ab07ae36e6f5c7bf4`; this worker's
  uncommitted source revision is pending parent integration.
- Device, driver, and NVPTX toolchain: not measured or recorded.
- Build/test commands: none; the worker was assigned no Cargo/build/GPU slot.
- Expected: both kernels compile for the selected NVPTX target and exactly
  match the independent CPU/reference and existing sampling result for finite
  vectors; any nonfinite input yields status 1 and the lowest bad index.
- Observed: source files and host fixtures are authored; runtime behavior and
  emitted PTX remain unverified.
- Evidence paths: `kernels/nvptx/greedy_bf16.rs` and
  `reference/greedy_bf16.rs`.


## Parent integration checkpoint

Registered both exports and the independent reference. Host tests (266) and
Clippy pass; Just PTX compilation succeeds on nightly-2026-09-25 for SM120a.
The first xtask wrapper used the wrong Result alias; it was corrected to the
existing DynResult, with the failed compiler log retained in ignored output.

`greedy-check` compares the kernel's finite token/bits and first nonfinite index
against both independent reference and existing CPU sampling, including scratch
reuse, all-finite-code orderings, vocabulary tails, signed zeros, subnormals,
and all-invalid inputs. The host Selector owns checked persistent scratch and
reads only the final16-byte status. CUDA failures drain and poison its lease.

Prepared ordinary-model integration is behind `MESH_SPECIALIZE_GPU_GREEDY=on`
(default off). A distinct `forward_selected` interface returns token/past without
promising host logits. Full-logit forwards and diagnostic observers preserve
existing output semantics. Selection completes before the cursor transaction
commits; nonfinite status returns an error. Model-profile adds an independent
full-logit versus device-only token/state check when enabled. Modelbench uses
the selected-output interface and reports the policy flag. MTP rejects enabling
this policy until its separate selection path is integrated. No serving default
or arithmetic profile changed. Linux compilation, standalone GPU checks, all
three sanitizers, model equivalence, and paired throughput remain pending.


Standalone device qualification `greedy-check-1` at source `11b84609f` passed
all77cases and three-repeat scratch reuse in normal execution, memcheck,
racecheck and synccheck; all tools reported zero errors/hazards. RTX5090,
driver615.71.09, PTX SHA256
`961d1652408eeb9ec8d72aa9c14d32ced2c2f0efb3e8e2a32dd0d5cdcc0c4a05`.
Tile/finish kernels use24/20registers,64sharedbytes each and zero local bytes.
Ninfer stayed inactive and ComfyUI remained resident. Linux Clippy and release
tools build passed. The ordinary-model paired ablation is running separately;
standalone selection evidence is not model throughput or serving qualification.

The model-profile event capture continues to describe its full-logit diagnostic
forward, even when the GPU-selection option is enabled. Its separate
`gpu_selection_check` executes the device-only path on identical inputs and
compares selected tokens and complete state. Do not attribute the benchmark's
GPU-selection effect from those diagnostic event totals.
