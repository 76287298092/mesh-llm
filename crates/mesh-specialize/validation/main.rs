//! Separate validation executable. This file and its cuBLAS module are never
//! compiled into the runtime library, even with the validation feature enabled.

#[cfg(target_os = "linux")]
mod cublas;
#[cfg(target_os = "linux")]
mod runner;
// Share allocation ownership and fixtures without exposing a public device API.
// The validator uses a subset of the driver/layout methods; the runtime has its
// own compilation of these sources and has no dependency on this executable.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
#[path = "../src/kernels/cuda/driver.rs"]
mod driver;
#[cfg(target_os = "linux")]
#[path = "../src/kernels/gemm_fixtures.rs"]
mod gemm_fixtures;
#[cfg(target_os = "linux")]
#[allow(dead_code)]
#[path = "../src/kernels/nvfp4_layout.rs"]
mod nvfp4_layout;
#[cfg(target_os = "linux")]
#[path = "../src/kernels/rms_norm_fixtures.rs"]
mod rms_norm_fixtures;
#[cfg(target_os = "linux")]
use mesh_specialize::reference;

fn main() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    return runner::run(&std::env::args().skip(1).collect::<Vec<_>>());
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!("CUDA library validation requires Linux")
}
