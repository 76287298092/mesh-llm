//! Fixed Qwen3.8-27B text model. Full execution is not yet implemented.
mod entry;
pub mod inventory;
pub mod projections;
pub use entry::trial;
pub mod attention;
pub mod decoder;
pub mod fp8_mlp;
pub mod model_reference;
mod model_weights;
pub mod residency;
pub mod resident_attention;
pub mod resident_gdn;
pub mod schedule;
