//! Shared single-machine serving defaults used by Skippy and Mesh.

pub const CTX_SIZE: u32 = 4096;
pub const BATCH: u32 = 512;
pub const UBATCH: u32 = 512;
pub const PARALLEL: usize = 32;
pub const PREFILL_CHUNK_SIZE: usize = 64;
pub const PREFILL_ADAPTIVE_START: usize = 64;
pub const PREFILL_ADAPTIVE_STEP: usize = 64;
pub const PREFILL_ADAPTIVE_MAX: usize = 512;
pub const PREFILL_ADAPTIVE_TARGET_MS: f64 = 100.0;
pub const PREFILL_CHUNK_POLICY: &str = "fixed";
