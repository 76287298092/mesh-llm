//! Device-specific instruction qualification, separate from model execution.

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
