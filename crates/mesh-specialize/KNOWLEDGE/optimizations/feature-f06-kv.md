# F06: row-scaled FP8 KV codec candidate

Status: Rust candidate and independent CPU oracle implemented; parent PTX build,
GPU correctness, attention integration, sanitizers and measurements are pending.
This entry records a bounded codec deliverable, not F06 qualification or Ninfer
parity. The worker did not run Cargo, compile PTX, access a GPU or inspect Ninfer.

## Candidate contract

The row is one token/head vector of 256 channels, stored contiguously in
row-major order. K and V are separate arrays using the same format. Encoding
accepts BF16 `u16` elements and writes one signed E4M3FN byte per element, one
IEEE binary16 scale (`u16`) per row, and one `u32` status per row. Decoding
consumes those arrays and produces row-major BF16 `u16` output plus one `u32`
status per row. The stored scale is always used for both encoding and decoding.

Scale selection uses `max(abs(row))`. An all-zero row has scale `1.0`; otherwise
the scale is the smallest finite positive binary16 value `s` for which
`s * 448 >= max(abs(row))`. This explicitly rounds the represented scale upward
when nearest-even binary16 conversion would clip the row. If the required scale
exceeds binary16 `65504`, the row is unsupported and returns a status instead of
silently overflowing. The minimum positive binary16 subnormal is valid and is
used if the rounded scale underflows. E4M3FN values are rounded nearest-even;
encoding saturates at finite magnitude 448 and never emits the reserved NaN
codes. Decoding rounds the represented FP32 product to BF16 nearest-even.

Status bits are stable across the kernel candidate and CPU oracle:

| Bit | Meaning | Encode or decode behavior |
| --- | --- | --- |
| `1` | Input BF16 row contains NaN or infinity | Encode zero codes and scale `1.0` |
| `2` | Finite row needs a scale greater than `65504` | Encode zero codes and scale `1.0` |
| `4` | Decode row contains E4M3FN NaN code `0x7f` or `0xff` | Decode an all-zero BF16 row |
| `8` | Decode scale is negative, zero, infinity or NaN | Decode an all-zero BF16 row |

If multiple decode errors occur, their bits are ORed. Decode accepts positive
binary16 subnormal scales. Shape and host-buffer extent errors are rejected by
the CPU oracle; device pointer extents remain part of the launch safety contract.

## Kernel ABI and launch geometry

`kernels/nvptx/kv_fp8.rs` defines two unmangled PTX kernels:

```text
kv_fp8_encode_bf16_width256(
  input_bf16: *const u16, codes: *mut u8, scales_f16: *mut u16,
  status: *mut u32, rows: u32)
kv_fp8_decode_bf16_width256(
  codes: *const u8, scales_f16: *const u16, output_bf16: *mut u16,
  status: *mut u32, rows: u32)
```

Launch both with block `[256, 1, 1]` and grid x at least `rows`. CTA x owns
exactly one row; padded CTAs return without memory access. Input/output payloads
cover `rows * 256` elements and scales/status cover `rows`. Every allocation
must be live, correctly aligned and pairwise disjoint for the duration of the
launch. The reduction and decode-validation scratch is CTA-local shared memory.
The codec does not define cache page mapping; the caller supplies contiguous
logical rows and owns any paging-aware address translation.

## Independent oracle and fixtures

`reference/kv_fp8.rs` derives E4M3FN and binary16 values from their logical sign,
exponent and fraction fields. It chooses a represented scale by searching the
finite binary16 encoding order, then performs an independent FP64 distance
comparison for E4M3 nearest-even rounding. Decode similarly reconstructs logical
values and applies BF16 rounding in the host reference; it does not reuse device
helpers.

Unit cases cover signed zero, E4M3 subnormals, even-code ties, finite saturation,
scale-up and binary16 range boundaries, unsupported BF16 input range, nonfinite
BF16 rows, invalid E4M3 codes/scales, valid positive-subnormal scales, row-specific
scales, an odd final row, and malformed extents. They are present in the source
but have not been run by this worker.

## Evidence and limits

- Source revision: exact commit is pending parent integration; the worktree had
  concurrent changes when this worker started.
- Device, driver and compiler: not measured or recorded for this source-only step.
- Command: `rustfmt --edition 2024` on the two new Rust files; completed without
  diagnostics.
- Commands not run: Cargo tests, PTX build, GPU, sanitizer, or benchmark.
- Expected next result: parent registers the kernel and reference, compiles the
  Rust NVPTX crate, and compares encode/decode outputs and status rows against
  this oracle before attention integration.
- Observed: source files and these oracle tests are present; runtime behavior is
  unverified.
- Evidence paths: `kernels/nvptx/kv_fp8.rs` and `reference/kv_fp8.rs`.

The storage payload is one byte per value plus two scale bytes per 256-value
row, versus two bytes per BF16 value (258 versus 512 bytes/row before alignment).
That arithmetic suggests about half the cache payload size; it does not establish
reduced attention latency, a 2x speedup, long-context accuracy, or performance
parity. The existing BF16 KV path remains the control until parent qualification.

Parent integration caught a test fixture encoding error: BF16 3/1024 is `0x3b40`, not `0x3ac0`. The fixture was corrected; the codec rounding rule was unchanged.

Parent integration: 244 CPU tests pass after correcting the BF16 test fixture. Host Clippy and NVPTX compilation pass; GPU qualification remains pending.
