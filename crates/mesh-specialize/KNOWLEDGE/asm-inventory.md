# Assembly inventory

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, lane read | Thread identity | NVPTX | Lane-indexed input/output contract | Unqualified |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, MMA | 16x8x64 block-scaled FP4 multiply | SM120a, PTX8.7 | Independent scalar decoded matrices, pending integration | Unqualified |
| `kernels/nvptx/probes.rs:panic` | Fail-fast trap | NVPTX | Unexpected panic must fail launch | Unqualified |

Compiler emission alone is not qualification. Keep execution evidence and any
failed attempts in a findings/dead-ends entry before promoting these rows.
