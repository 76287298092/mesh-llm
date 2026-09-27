//! Device-specific instruction qualification, separate from model execution.

/// Validated by the scalar reference before any GPU allocation or launch.
pub struct EmbeddingNormInput {
    pub table: Vec<u8>,
    pub weight: Vec<u8>,
    pub width: usize,
    pub epsilon: f32,
    pub batches: Vec<Vec<u32>>,
}

pub struct Fp8Projection {
    pub name: String,
    pub weights: Vec<u8>,
    pub scales: Vec<u8>,
    pub channels: usize,
}

pub struct ProjectionInput {
    pub entry: EmbeddingNormInput,
    pub projections: Vec<Fp8Projection>,
    pub bf16_projections: Vec<Bf16Projection>,
    pub convolution: Option<CausalConv4Weights>,
    pub gdn: Option<GdnWeights>,
    pub gdn_output: Option<GdnOutputWeights>,
}

pub struct GdnOutputWeights {
    pub z_projection: usize,
    pub norm: Vec<u8>,
    pub epsilon: f32,
    pub projection: Fp8Projection,
}

pub struct GdnWeights {
    pub a_projection: usize,
    pub b_projection: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
    pub a_log: Vec<u8>,
    pub dt_bias: Vec<u8>,
}

/// Fixed-width causal convolution attached to one FP8 projection output.
pub struct CausalConv4Weights {
    pub projection: usize,
    pub weights: Vec<u8>,
}

pub struct Bf16Projection {
    pub name: String,
    pub weights: Vec<u8>,
    pub channels: usize,
}

pub fn projection_check(
    ptx: &str,
    device: i32,
    input: &ProjectionInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::projections::run(ptx, device, input);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, input);
        anyhow::bail!("Qwen projection GPU trial requires Linux")
    }
}

pub fn embedding_norm_check(
    ptx: &str,
    device: i32,
    input: &EmbeddingNormInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::embedding_norm::run(ptx, device, input);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, input);
        anyhow::bail!("Qwen entry GPU trial requires Linux")
    }
}

#[cfg(target_os = "linux")]
mod cuda;
#[cfg(any(target_os = "linux", test))]
mod fixtures;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(
    feature = "validation",
    allow(
        dead_code,
        reason = "Dense fixture materialization is used by the separate validation binary"
    )
)]
mod gemm_fixtures;
#[cfg(any(target_os = "linux", test))]
mod memory_fixtures;
#[cfg(any(target_os = "linux", test))]
mod nvfp4_layout;
#[cfg(any(target_os = "linux", test))]
mod ordinary_fixtures;
#[cfg(any(target_os = "linux", test))]
mod rms_norm_fixtures;

/// Snapshot the exact selected CUDA device and its current free memory. This
/// creates and destroys a context, but loads no model or device kernel.
pub fn device_probe(device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::device_probe(device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        anyhow::bail!("CUDA device admission trials require Linux")
    }
}

/// Run representative RMSNorm and tiled GEMM fixtures with resident GPU timing.
/// These are kernel workloads, not model prefill/decode measurements.
pub fn workload_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::workloads::run(ptx, device, true);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA workload trials require Linux")
    }
}

/// Check each representative workload once, for bounded sanitizer execution.
pub fn workload_check(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::workloads::run(ptx, device, false);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA workload trials require Linux")
    }
}

/// Qualify shared-memory copies/loads, ordinary MMA, and register budgeting.
pub fn instruction_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::instructions::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA instruction trials require Linux")
    }
}

/// JIT and numerically check the Rust NVFP4 single-warp probe on SM120.
/// Results are instruction evidence, not model throughput.
pub fn nvfp4_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA instruction trials require Linux")
    }
}
