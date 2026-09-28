# Encoded embedding and FP32 GDN parameter consumers

Status: implemented, not compiled or GPU-qualified by this worker. This is bounded
weight-representation compatibility, not full Ninfer arithmetic parity or a profile
promotion. Parent owns the direct `.ninfer` reader, normalized weight views,
registration in the central inventory, compilation, integration, and qualification.
No Ninfer source is imported. No raw-profile admission checks are loosened.

## Representation contract

Dispatch depends on verified logical tensor metadata, not the source container.
Both legacy and stream execution bind through `ResidentWeights.object` followed by
`ResidentWeights.tensor`. The latter checks tensor kind, dtype, shape, byte length,
`safetensors-row-major-v1` layout, and the arena region. Existing execution
context checks retain arena/module/state ownership. A direct-file reader may
expose these normalized logical views without conversion to `.mspec`.

- `tensors/model.language_model.embed_tokens.weight`: existing BF16 path stays
  unchanged. The new encoded path requires FP8 E4M3 `[248320, 5120]`, exactly
  1,271,398,400 bytes, and the sibling `.weight_scale` tensor must be BF16
  `[248320, 1]`, exactly 496,640 bytes. The BF16 path needs no scale tensor.
- Each GDN `.A_log` and `.dt_bias`: matched BF16 is the legacy default. Matched
  F32 is accepted only at `[48]`, 192 bytes each. Mixed precision, another dtype,
  incorrect extent, or incorrect layout fails binding. Neither parameter is
  narrowed to BF16. Both resident GDN and stream GDN use the same binding helper.
- All norm parameters remain BF16. Input/post/final/Q/K norm uses `1 + weight`;
  GDN output norm uses direct gamma. No beta or gated-norm rounding change occurs.

Each selected embedding scalar follows
`BF16_RNE(float(E4M3FN_code) * float(BF16_row_scale))` **before** normalization.
The decoder covers signed zero, subnormals, and finite E4M3FN exponent 15. It does
not interpret E4M3FN as IEEE infinity-at-max-exponent or unsigned-zero E4M3FNUZ.
Only requested rows are gathered. There is no resident BF16 vocabulary expansion.

Legacy embedding allocates temporary `rows * width * 2` BF16 bytes and sequential
u32 row IDs, gathers, synchronizes, and passes them to the existing fused BF16
embedding/norm operation. Temporaries live through normalization completion.

Stream entry gathers into existing persistent `s.hidden`, then invokes the same
embedding/norm with `s.row_ids`. `hidden`, token IDs, and row IDs are already
whole-forward arena slots. No workspace slot or planner change is needed.
The only alias is `table == residual`: identity row IDs ensure each thread copies
its own source element back with identical bits; no other CTA owns that row;
the reduction barriers precede the second table read. Normalized/raw outputs
remain disjoint. This is source-level alias-safety reasoning, not a substitute
for the parent's device sanitizer checks. Arbitrary token IDs are not alias-safe.

Stream handles are resolved once, including both new entries. Its report names
embedding encoding, per-GDN parameter precision, unchanged BF16 beta, and
`full_ninfer_arithmetic_parity: false`. Source encoding and container are separate.

## Exact kernel ABIs

All pointers are 64-bit device addresses and scalars are 32-bit. Arguments below
are in declaration/launch order; no implicit rows or vocabulary arguments exist
for the gather. Host bounds and context checks remain mandatory.

### `fp8_embedding_gather`

Source: `kernels/nvptx/fp8_embedding_gather.rs`.

1. `codes: *const u8`, row-major `[vocabulary, width]` E4M3FN
2. `scales: *const u16`, BF16 `[vocabulary, 1]`
3. `tokens: *const u32`, `[rows]`
4. `output: *mut u16`, BF16 `[rows, width]`
5. `width: u32`

Grid `[rows, 1, 1]`, block `[256, 1, 1]`, dynamic shared bytes 0. Every source token
must be in range; gather input/output extents are mutually disjoint. The emitted
BF16 rows, including signed zero, are the normalization input and residual bits.

### `gdn_gates_f32_params`

Source: `kernels/nvptx/gdn_gates_f32_params.rs`.

1. `a: *const u16`, BF16 `[rows, heads]`
2. `b: *const u16`, BF16 `[rows, heads]`
3. `a_log: *const f32`, F32 `[heads]`
4. `dt_bias: *const f32`, F32 `[heads]`
5. `beta: *mut u16`, BF16 `[rows, heads]`
6. `g: *mut f32`, F32 `[rows, heads]`
7. `decay: *mut f32`, F32 `[rows, heads]`
8. `rows: u32`
9. `heads: u32`

Grid `[ceil(rows * heads / 256), 1, 1]`, block `[256, 1, 1]`, dynamic shared bytes 0.
The legacy `gdn_gates` ABI is unchanged, with BF16 pointers at positions 3 and 4.
Both call one internally factored `gate_values` body with identical operation
order: stable sigmoid to BF16 beta, FP32 addition, unchanged stable softplus,
negative exp(A_log), rounded multiplication, and exp(g). Existing exp2/log2
approximations, softplus threshold/polynomial, and subnormal behavior remain.
The F32 entry changes parameter loads only.

## Assembly and source inventory

Neither new device file contains an inline-assembly site. Both are Rust modules
registered in `kernels/nvptx/probes.rs`. Their emitted assembly reuses these exact
existing source sites, already represented in the central assembly inventory:

- Gather: `embedding_norm.rs::thread_and_row` (`mov.u32` tid.x/ctaid.x), and
  `embedding_norm.rs::fp32_multiply_rn` (`mul.rn.f32`). BF16 encoding is integer
  Rust through `encode_bf16_rne`; E4M3 decoding is independent scalar Rust.
- F32 gates: `gdn_prepare.rs::block_and_thread` (coordinate moves), then
  `fp32_add_rn`, `fp32_multiply_rn`, `fp32_divide_rn`, `fp32_exp2_approx`, and
  `fp32_log2_approx` through `gate_values`. No Q/K shared-memory/barrier site is
  used by the gates. There is no new architecture requirement or vendor library.

The parent should cross-link the new consumers/oracles in the central inventory.
This worker does not edit inventory/schedule or the loader/assembler-owned files.

Exact changed source files, relative to `crates/mesh-specialize/`:

- New: `kernels/nvptx/fp8_embedding_gather.rs`
- New: `kernels/nvptx/gdn_gates_f32_params.rs`
- New: `reference/fp8_embedding_gather.rs`
- New: `reference/gdn_gates_f32_params.rs`
- `kernels/nvptx/embedding_norm.rs` (three helper visibilities; narrow alias contract)
- `kernels/nvptx/gdn_prepare.rs` (coordinate helper visibility; shared gate body)
- `kernels/nvptx/probes.rs` (two module registrations)
- `reference/gdn_prepare.rs` (oracle gate-helper visibility only)
- `src/lib.rs` (two reference module exports)
- `src/kernels/cuda/resident_embedding.rs`
- `src/kernels/cuda/resident_gdn_core.rs`
- `src/kernels/cuda/stream_forward/weights.rs`
- `src/kernels/cuda/stream_forward/functions.rs`
- `src/kernels/cuda/stream_forward/layers.rs`
- `src/kernels/cuda/stream_forward.rs` (representation report field only)
- New: this knowledge entry

The follow-up standalone harness additionally changes:

- New: `src/kernels/cuda/native_parameter_trial.rs`
- `src/kernels/cuda/mod.rs` (trial module registration)
- `src/kernels.rs` (Linux entry point and non-Linux rejection)
- Repo-relative `tools/xtask/src/specialize/probe.rs` (existing probe writer reuse)
- Repo-relative `tools/xtask/src/specialize.rs` (command dispatch)

## Standalone qualification command

`xtask specialize native-parameter-check --ptx PATH --device ORDINAL --output NEW_FILE`

The command uses the existing probe parser/report writer, refuses to overwrite
an output file, records the PTX hash, and exits unsuccessfully if any case fails.
No model import or native container is needed. Parent builds/runs it and applies
compute-sanitizer; this worker has not executed it.

Nine bounded synthetic cases cover embedding widths 1, 7, 257 and 5120 with six
requested rows, and 48-head gates at one/seven rows. Gate cases separately perturb
only A_log, only dt_bias, and both with nonzero low F32 bits. Representable
parameters require byte-identical beta/g/decay between old/new gate kernels.
Low-bit cases must differ from the narrowed legacy control in g and decay while
beta remains identical. Both versions independently meet the CPU oracle budget.

Every allocation has 256-byte prefix/suffix canaries. Outputs start with BF16 or
FP32 NaN poison, are re-poisoned in the same allocations, and must repeat bitwise.
Inputs must remain byte-identical including guards. Gather outputs must match the
independent scalar oracle exactly, including signed zero and positive/negative
ties. Normalization uses the gathered BF16 rows and checks all three outputs:
exact residual, raw FP32 within `2e-6 + 2e-6 * abs(reference)`, BF16 within one ULP
of the independent norm oracle and exactly rounded from device raw output. The
identity-ID in-place residual variant must be byte-identical to the disjoint
variant and preserve its input rows. Gate beta allows one BF16 ULP; log-decay and
decay allow `3e-6 + 5e-6 * abs(reference)`, with explicit finiteness/domain checks.
Poison and extent-mismatch rejection have pure host tests.

The report records per-case decisions, input/guard integrity, repeat identity,
precision differences, JIT log, device identity, and resources for all four
kernel entries. It contains no timing or performance measurement.

## Tests and remaining qualification

Added pure scalar tests cover signed zero, E4M3 extremes/subnormals, NaNs,
positive/negative BF16 halfway ties, BF16 subnormal/overflow rounding, repeated
and endpoint token IDs, malformed extents, and the rounded-row normalization
boundary. The independent decoder uses logical powers-of-two arithmetic; its
BF16 oracle compares discarded bits rather than copying the device encoder.
The F32 gate oracle uses the independent existing FP64 exp/log1p reference, never
PTX helpers. Tests perturb each parameter with bits not representable in BF16,
check that narrowing changes g/decay while beta stays unchanged, cover the linear
softplus branch, and compare exactly representable parameters with the old oracle.
Pure host binding tests cover BF16 defaults, bounded FP8 dimensions, matched F32
48-head parameters, rejected mixed precisions, unsupported dtypes and wrong heads.
CUDA-module binding tests require the parent's Linux host build but no GPU calls.

Worker validation: Rust 2024 rustfmt only. No Cargo, Git, SSH, GPU execution, or
delegation. Rust tests, host/PTX compilation, emitted-PTX inspection, legacy
regression comparisons, stream/resident equivalence, and memcheck/racecheck/
synccheck remain for the parent. Source revision: uncommitted parent worktree;
commit not queried. Target GPU SM120a; driver/toolchain/clocks not measured here.
Performance before/after: not measured. This change makes no speed, memory-free,
quality, full arithmetic parity, or automatic profile-promotion claim.
