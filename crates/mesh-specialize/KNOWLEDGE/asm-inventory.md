# Assembly inventory

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, lane read | Thread identity | NVPTX | Lane-indexed input/output contract | Qualified within the 32-case probe |
| `kernels/nvptx/probes.rs:probe_nvfp4_mma`, MMA | 16x8x64 block-scaled FP4 multiply | SM120a, PTX8.7 | Independent scalar decoded matrices | 4,096 exact matches; see [evidence](findings/rust-nvfp4-probe.md) |
| `kernels/nvptx/probes.rs:panic` | Fail-fast trap | NVPTX | Unexpected panic must fail launch | Unqualified |
| `kernels/nvptx/memory.rs:probe_shared_load`, lane/address/copy/barrier | `cp.async.ca/cg`, commit and wait, shared synchronization | SM120a | Two seeded 256-halfword arrays | 16 cases pass; sanitizers clean |
| `kernels/nvptx/memory.rs:load_x2/load_x4` | Normal/transposed `ldmatrix` x2/x4 | SM120a | Logical 8x8 row/column mapping in `memory_fixtures.rs` | 2,048 exact outputs with copies |
| `kernels/nvptx/ordinary_mma.rs` | BF16/FP16 m16n8k16 and INT8 m16n8k32 MMA, lane read | SM120a | Twelve independent scalar matrix products in `ordinary_fixtures.rs` | 1,536 exact outputs; sanitizers clean |
| `kernels/nvptx/register_budget.rs` | `setmaxnreg` dec24/inc64, barriers, thread read | SM120a | 128 exact XOR outputs; launch requires reported allocation >=64 registers | 128 outputs pass; 64 registers reported; sanitizers clean |
| `kernels/nvptx/rms_norm.rs` | Shared reduction, barriers, explicit rounded FP32 arithmetic, thread/block coordinates | SM120a | Independent f64 RMSNorm | Qualified; see [representative kernels](findings/representative-kernels.md) and Qwen entry regression |
| `kernels/nvptx/nvfp4_gemm.rs` | Repeated NVFP4 MMA, lane/block coordinates | SM120a | Independent logical GEMM and separate cuBLAS reference | Qualified; see [representative kernels](findings/representative-kernels.md) and Qwen entry regression |

Compiler emission alone is not qualification. Keep execution evidence and any
failed attempts in a findings/dead-ends entry before promoting these rows.

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/embedding_norm.rs` | Thread/block coordinates, shared 256-thread reduction and barriers, explicit rounded FP32 add/multiply/divide/square-root | SM120a | Independent scalar real-weight embedding and zero-centered RMSNorm in `reference/embedding_norm.rs` | 696,320 values pass; memory/race/sync checks clean; see [entry trial](findings/qwen-entry.md) |

| Source symbol | Operation | Required target | Reference | Status |
| --- | --- | --- | --- | --- |
| `kernels/nvptx/fp8_quantize.rs` | CTA coordinates, shared max reduction/barriers, rounded FP32 division | SM120a | Independent exhaustive nearest-value FP8 encoder; finite/tie/zero/tiny GPU fixtures | Exact codes/scales; all three sanitizers clean; see [projections](findings/qwen-projections.md) |
| `kernels/nvptx/fp8_linear.rs` | Warp/CTA coordinates, E4M3 m16n8k32 MMA, rounded FP32 scale multiplication | SM120a | Logical f64 dot products and NVIDIA fragment mapping | 294,912 real outputs meet tolerance; all three sanitizers clean; see [projections](findings/qwen-projections.md) |
| `kernels/nvptx/bf16_linear.rs` | Warp/CTA coordinates and BF16 m16n8k16 MMA | SM120a | Logical f64 BF16 dot products and previously qualified BF16 fragment mapping | 1,728 real outputs meet tolerance; tail fixture and all three sanitizers pass; see [projections](findings/qwen-projections.md) |
| `kernels/nvptx/causal_conv4.rs` | CTA/thread coordinates, rounded FP32 multiply/add/divide, approximate FTZ exp2 | SM120a | Independent f64 convolution/SiLU and raw-state reference; whole/chunk/single-token equivalence | 184,320 real outputs meet tolerance; state exact; all three sanitizers clean; see [convolution](findings/causal-convolution.md) |
| `kernels/nvptx/gdn_prepare.rs` | CTA/thread coordinates, two shared norm reductions/barriers, rounded arithmetic/sqrt, approximate exp2/log2 with subnormal support | SM120a | Independent f64 norm/exp/log1p reference; head/zero/extreme/underflow fixtures | GPU execution pending; see [preparation](findings/gdn-preparation.md) |
