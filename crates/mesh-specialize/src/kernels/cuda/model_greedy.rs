//! Process-fixed experimental ordinary-token selection policy.
use anyhow::Result;
use std::sync::OnceLock;

pub(super) fn enabled() -> Result<bool> {
    static VALUE: OnceLock<Result<bool, String>> = OnceLock::new();
    match VALUE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_GPU_GREEDY") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "off" => Ok(false),
        Ok(v) if v == "on" => Ok(true),
        _ => Err("MESH_SPECIALIZE_GPU_GREEDY must be off or on".into()),
    }) {
        Ok(v) => Ok(*v),
        Err(e) => Err(anyhow::anyhow!(e.clone())),
    }
}
