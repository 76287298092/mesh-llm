# Connecting persistent decoder execution

Status: implementation in progress. Existing component and selected whole-layer
arithmetic checks remain qualified; the new connected execution is not yet qualified.

The next stage separates GPU operation execution from the old component-check
harnesses. Weight views bind exact dtype, row-major layout, shape and byte extent.
Forward calls validate input/state extents and CUDA context ownership, use the
same previously qualified kernels, and never substitute scalar or downloaded
intermediates. Host scalar reads are limited to immutable global scales at binding.

The persistent state arena has disjoint named histories, recurrent matrices and
K/V regions. Causal convolution must not alias old and next history: it writes
temporary next-history storage, synchronizes, then uses a checked device-to-device
copy into the persistent region. Recurrent updates operate on the existing region.
Device copies require distinct allocations, matching contexts and in-range slices.

The intended connected path is embedding, ordered 64-layer attention/GDN and
MLP/residual execution, final norm and logits. Existing kernel diagnostics remain
allocated temporarily to preserve kernel ABIs; allocation reuse and diagnostics-free
kernels are later optimizations. Full model correctness and partition equivalence
must pass before any model-throughput or usable-context claim.

No new device arithmetic is planned in this extraction. Pure bounds tests,
Linux builds and live connected-path/sanitizer evidence remain required. Record
exact source and hardware evidence when those gates run; do not infer them from
the earlier check-harness successes.
