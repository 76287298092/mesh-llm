# Native MTP packed-parent residency

Status: packed-parent GPU residency and readback qualified for the recorded
snapshot on RTX 5090, under normal execution and memcheck. Native MTP admission
remains closed; whole native MTP qualification remains open. This residency
evidence does not qualify operators, scoring, or model execution.

`NativeModelSource::native_mtp_parents` uses existing verified resolution and
returns borrowed read-only parent metadata with object ID, bytes, and SHA-256.
The existing retained-descriptor copy rechecks each parent's saved hash and
dirties the source on failed hash or destination writes.

`ResidentNativeMtp::load` builds the existing 256-byte-aligned `Layout` from
deduplicated physical IDs, allocates one owned driver `Buffer`, and streams each
parent through the existing `BufferRegionSink`. It returns the owner only after
all copies and sink completion checks succeed. Any failure drops the local
allocation. Packed codes, FP16 scales, norms, and token-map bytes stay unchanged.

Parent, Q8, and Q4 bindings borrow the allocation owner. Q8/Q4 binding requires
full equality with a retained verified selection, then borrows the saved view.
An equal clone is accepted; changed row mappings, shapes, or plane descriptors
are foreign views. Code and scale pointers use checked parent-relative ranges
and checked device-pointer additions. Diagnostic readback uses caller-owned
memory, checks parent bounds and arena offset conversion, and caps each transfer
at 1048576 bytes. There is no dense dequantization.

## Evidence and limitations

The prior host qualification recorded 14 parents and 808307712 bytes. See
[packed views](native-mtp-views.md). That host-only evidence did not qualify the
device loader. The initial residency implementation's host tests were unrun at
worker delivery because Cargo, builds, SSH, and GPU use were prohibited. The
completed parent trial below supplies separate device evidence; it does not
retroactively turn the earlier host-only checks into GPU qualification.

Added host tests cover parent range boundaries, range and pointer overflow,
foreign Q8/Q4 metadata, equal-clone selection, and destination rejection during
verified parent copy. The parent reports Linux validation passing with test
counts of 589 + 26 + 72, Clippy, and the build. These are
parent-reported results, not checks rerun for this documentation update. Parent
LSP attempts timed out and do not constitute a clean diagnostic result.

### Completed GPU readback trial

Evidence directory:
[`native-residency-trial-20260930-a`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/).
[`normal.json`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/normal.json)
and
[`memcheck.json`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/memcheck.json)
both report `all_passed=true`, 14 physical parents totaling 808307712 bytes,
and all 14 readback SHA-256 hashes matching their expected source hashes. Each
aligned arena is 808307712 bytes with zero alignment padding. Both reports record
eight logical Q8 views and one logical Q4 view with saved-view equality,
parent-relative pointer checks, and foreign-view rejection passing. Readback
uses 1048576-byte chunks and a 1048576-byte host scratch buffer.

[`results.txt`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/results.txt)
records normal and memcheck passing and `ALL-DONE`.
[`memcheck.log`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/memcheck.log)
records Compute Sanitizer `ERROR SUMMARY: 0 errors`. This is memcheck evidence
only, not racecheck or synccheck qualification.

Normal-run measurements:

| Measurement | Recorded value |
| --- | --- |
| Packed-parent load | 0.394033099 seconds |
| Readback | 0.378582298 seconds |
| Source verification | 54.526130657 seconds |
| Free device memory before load | 32751681536 bytes |
| Free device memory while resident | 31942180864 bytes |
| Free device memory after drop | 32751681536 bytes |

These are diagnostic load, readback, and source-verification timings, not model
prefill, decode, or throughput. The memory values are checkpoints, not a measured
peak. Normal-run free memory returned to its pre-load value; the memcheck run's
separate memory checkpoints do not show that equality.

[`versions.log`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/versions.log)
records Carrack Linux 7.2.6-1-cachyos, RTX 5090 compute capability 12.0,
GPU UUID `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`, NVIDIA driver 615.71.09,
CUDA tools 13.4.92, rustc 1.98.1, and Compute Sanitizer 2026.3.0.0. The JSON
reports CUDA driver API version 13040. Controlled trial clocks are not measured.

Source identity is the uncommitted snapshot described by
[`native-residency-source-20260930-v2.sha256`](../evidence/native-residency-20260930/native-residency-source-20260930-v2.sha256),
with baseline `b188c2925aa05108ebb5b1fe7afd195c60254e6c`, not a new commit.
The parent supplies manifest SHA-256
`84377dcfb93ee50618ff1e3bd2daa8128d0b1ee64d7bf22f9135167178ae2efa`.
[`hashes.txt`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/hashes.txt)
records the trial `xtask` executable SHA-256
`4682e811bd081b05a4908bbe2073c44a33ba90090a552a67465f0256afff1256`
and source artifact SHA-256
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
The artifact is `/data/ai/ninfer/models/qwen3_8_27b_nvfp4.ninfer`.

[`units-after.txt`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/units-after.txt)
records both `ninfer-qwen38.service` and `battlecity-comfy.service` active before
and after the trial. The available health marker is
[`ninfer-health-restored.txt`](../evidence/native-residency-20260930/native-residency-trial-20260930-a/ninfer-health-restored.txt),
which contains `1`; `health-restored.txt` is not present. The parent separately
reports restored Ninfer HTTP 200. No service or HTTP checks were rerun here.

Both JSON reports explicitly retain `native_mtp_admitted=false`,
`operator_execution=false`, `model_executable=false`,
`dense_dequantization=false`, `text_tensors_loaded=false`, and
`full_ninfer_arithmetic_parity=false`. This qualifies packed-byte residency,
bindings, and readback only. It does not establish native MTP admission,
operator arithmetic, dense execution, draft/verify/rollback, decode correctness,
quality, serving readiness, or model timing. Parent-owned failure-path and wider
sanitizer qualification remain separate gates.

Durable rule: only return a resident owner after every packed physical parent
has passed streaming verification; metadata bindings must borrow that owner.
