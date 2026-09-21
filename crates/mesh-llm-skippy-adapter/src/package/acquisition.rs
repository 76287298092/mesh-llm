//! Mesh cache policy for local package validation and snapshot selection.
pub use skippy_api::package::acquisition::{
    StagePackageRef, is_layer_package_ref, is_metadata_only_package_inspection,
    manifest_artifact_bytes, resolve_local_package_files,
};
use std::path::{Path, PathBuf};

fn integrity_cache_dir() -> PathBuf {
    skippy_model_hf::store::mesh_llm_cache_dir().join("skippy-package-integrity")
}

pub fn verify_resolved_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> anyhow::Result<String> {
    skippy_api::package::acquisition::verify_resolved_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
        Some(&integrity_cache_dir()),
    )
}

pub fn verify_cached_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> anyhow::Result<Option<String>> {
    skippy_api::package::acquisition::verify_cached_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
        Some(&integrity_cache_dir()),
    )
}

pub mod cache_resolution {
    use super::*;
    pub use skippy_api::package::acquisition::cache_resolution::cached_package_snapshots;
    pub fn resolve_cached_hf_package_snapshot(
        package_dir: &Path,
        layer_start: u32,
        layer_end: u32,
        include_embeddings: bool,
        include_output: bool,
    ) -> anyhow::Result<Option<String>> {
        skippy_api::package::acquisition::cache_resolution::resolve_cached_hf_package_snapshot(
            package_dir,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
            Some(&integrity_cache_dir()),
        )
    }
}
