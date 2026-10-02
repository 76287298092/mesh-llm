//! Shared single-machine serving defaults used by Skippy and Mesh.

pub const CTX_SIZE: u32 = 4096;
pub const BATCH: u32 = 512;
pub const UBATCH: u32 = 512;
// Automatic local serving uses the shared KV planner; this remains the
// fallback for callers that construct a stage without model metadata.
pub const PARALLEL: usize = 4;
pub const PREFILL_CHUNK_SIZE: usize = 64;
pub const PREFILL_ADAPTIVE_START: usize = 64;
pub const PREFILL_ADAPTIVE_STEP: usize = 64;
pub const PREFILL_ADAPTIVE_MAX: usize = 512;
pub const PREFILL_ADAPTIVE_TARGET_MS: f64 = 100.0;
pub const PREFILL_CHUNK_POLICY: &str = "fixed";
