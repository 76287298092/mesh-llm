//! Internal specialized-runtime experiments. No serving or Skippy ABI yet.

pub mod artifact;
pub mod checkpoint;
pub mod kernels;
pub mod packages;

#[path = "../reference/embedding_norm.rs"]
pub mod entry_reference;

#[path = "../reference/arithmetic.rs"]
pub mod reference;
