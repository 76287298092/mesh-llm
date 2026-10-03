# DeepSeek-V4.1 Runtime Compatibility Plan

## Status

Design and source-difference assessment only. The exact target GGUF has not
been opened by the Mesh-pinned runtime, and no DeepSeek-V4.1 inference has
been run. Do not add `deepseek41` to `split-certified.json` or route it through
the `deepseek4` graph as a compatibility workaround.

The target's read-only GGUF inventory reports architecture `deepseek41`,
40 transformer layers, 384 experts, and six selected experts per token. Its
seven source shards total 264,515,279,456 bytes. Mesh's package writer has
produced a byte-preserving, 40-layer package; independent package verification
passed for all seven source files, 46 package artifacts, and 1,046 tensors.
Package-only certification selected and integrity-checked two half-model
admissions (26 and 22 artifacts). It did not load the native model, execute
either stage, or prove that the model can be split at those boundaries.

## Compatibility evidence

Mesh currently pins llama.cpp at
`8212c7802455255460ab8e18fc34754560031b34` and applies its own ordered Skippy
patch queue. The installed Mesh native runtime is built for the Mesh/Skippy
ABI, but its split-family certification roster includes `deepseek4`, not
`deepseek41`.

The isolated ik_llama experiment under the build directory contains
DeepSeek-V4.1 support. Comparing that implementation with Mesh's current
surface shows this is not an architecture-name alias: support adds model
registration and tensor layouts, V4.1-specific hyperparameter/metadata
validation, model tensor loading, runtime cache/state, and a distinct graph
path. The relevant work also spans the existing V4 graph and common runtime
state. That experiment's DLLs do not implement the Mesh Skippy ABI and are not
drop-in Mesh runtimes.

Therefore the viable options are:

1. Port the required behavior onto Mesh's exact pinned llama.cpp base and
   replay the existing Skippy patch queue; or
2. Deliberately update the llama.cpp pin to a revision containing V4.1,
   replay the full Skippy patch queue, resolve every conflict, and repeat the
   native ABI, graph, and Mesh stage-contract validation.

Do not copy an unrelated runtime DLL into the Mesh runtime directory or mark
the family certified before the following gates pass.

## Required native implementation

Keep DeepSeek-V4.1 as its own architecture while sharing only behavior proven
identical to DeepSeek-V4:

1. **Architecture and metadata.** Register `deepseek41` in the GGUF
   architecture table. Parse and validate V4 geometry, compression settings,
   hyper-connection parameters, expert gating, and all V4.1-specific engram
   metadata. Reject missing, contradictory, or out-of-range metadata before
   model allocation.
2. **Tensor inventory and loading.** Define required and optional tensors for
   attention compressors, indexers, hyper-connections, shared/routed experts,
   engram tables, and any MTP tensors. Validate tensor names, shapes, layer
   ownership, quantization types, and complete shard coverage. Large shared
   tables must have an explicit memory-placement policy.
3. **Graph semantics.** Add the V4.1 graph as a separate graph path. Validate
   its compressed-attention ratios, hierarchical indexer inputs, top-k carry,
   engram lookup, MoE routing, and hyper-connection collapse against the
   target's reference behavior. Similar tensor names do not prove that the
   V4 graph is mathematically equivalent.
4. **Decode state and handoff.** Enumerate every state value that survives
   token steps, including raw/compressed attention state, indexer/top-k state,
   recurrent carries, and hyper-connection state. Either include them in the
   Skippy stage-boundary contract and authenticated handoff, or explicitly
   reject split execution until a lossless contract exists.
5. **Backends and quantization.** Establish an operation/type support matrix
   for the exact Q2_K tensors and every graph operation on CPU first. Add GPU
   backends only after CPU reference parity. Unsupported quantized operations
   must fail during preflight rather than falling back to an incorrect path.
6. **Skippy ABI and certification.** Build a Mesh-native runtime with the
   matching Skippy ABI. Register and certify the architecture only after
   single-stage graph tests and state-boundary tests pass. Update the family
   roster from generated test evidence, not by editing the roster alone.

## SSD and memory constraints

The package is correctly separated by layer, but it is not yet proof of
low-memory inference. Its two certification ranges account for approximately
165.1 GB and 100.5 GB of artifacts respectively; layer 1 itself is split
across three package artifacts, one about 32 GB. Ordinary stage loading that
eagerly allocates every tensor would still exceed this host's approximately
16 GiB physical RAM.

Before loading any target weights, the native runtime needs an enforced,
observable per-process and per-stage memory budget, including model buffers,
active expert weights, engram data, graph workspace, KV/compressed state,
activations, and allocator overhead. A Windows working-set trim or ordinary
`mmap` flag alone is not a reliable budget. The initial safe execution order is:

1. metadata and tensor-shape preflight without reading weight payloads;
2. synthetic tiny V4.1 graph tests and quantized-kernel parity tests;
3. bounded single-layer loader tests, including the oversized expert layer;
4. one-node, tiny-context generation under an enforced memory stop condition;
5. two-node stage-chain generation with per-node RAM and SSD I/O evidence.

If a stage cannot fit even its measured minimum working set, reject it. Do not
substitute full-model mmap overcommit, plugin-reported aggregate RAM, or
replica routing for a verified distributed stage.

## Acceptance gates

- Exact target metadata and tensor inventory pass native preflight.
- Native model open succeeds with a recorded runtime fingerprint and bounded
  memory use.
- Tiny reference inputs produce numerically compatible V4.1 outputs using the
  target Q2_K weights.
- Every inter-layer and decode-state dependency is either handed off exactly
  or the corresponding split is rejected.
- Every admitted stage fits a measured local budget; cold and warm storage
  reads, active expert residency, and process high-water marks are recorded.
- Only after the above pass may `deepseek41` be added to family certification.
- Multi-node acceptance requires one request to traverse two or more real
  Skippy stages and complete generation; package resolution or session
  replica-routing does not satisfy this gate.
