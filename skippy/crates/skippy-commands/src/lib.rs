//! Standalone Skippy command execution and output formatting.
//!
//! Argument parsing lives in `skippy-cli`; this crate executes the parsed
//! commands against the shared Skippy API and renders their JSON output.
//! It intentionally has no dependency on `skippy-serving` and adds no
//! serving options types of its own.

pub mod console;
mod model_catalog;
pub mod models;
pub mod prompt;
pub mod runtime;
pub mod split;
