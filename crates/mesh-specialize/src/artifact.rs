//! Internal `.mspec` placement format. Integrity verification does not qualify
//! a model, authorize its origin, or make its execution schedule data-driven.

pub mod header;
mod identity;
pub mod model_source;
pub mod ninfer;
pub mod reader;
pub mod schema;
pub mod writer;

/// Content identity for a schema-validated logical inventory. Placement offsets
/// are excluded; tensor metadata, bytes' checksums and recipe remain bound.
pub fn weights_identity(directory: &schema::Directory, payload_len: u64) -> anyhow::Result<String> {
    directory.validate(payload_len)?;
    Ok(identity::calculate(directory))
}
