//! Fixed Qwen3.8-27B text decoder and bounded independent correctness trials.
mod entry;
pub mod inventory;
pub mod projections;
pub use entry::trial;
pub mod attention;
pub mod decoder;
pub mod fp8_mlp;
pub mod model_benchmark;
pub mod model_profile;
pub mod model_reference;
pub mod model_score;
mod model_weights;
pub mod residency;
pub mod resident_attention;
pub mod resident_gdn;
pub mod schedule;

pub mod mtp;

pub mod native_source;
pub(crate) mod native_views;
