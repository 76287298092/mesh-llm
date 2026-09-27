//! Device-specific instruction qualification, separate from model execution.

#[cfg(target_os = "linux")]
mod cuda;
#[cfg(any(target_os = "linux", test))]
mod fixtures;
#[cfg(any(target_os = "linux", test))]
mod nvfp4_layout;

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
