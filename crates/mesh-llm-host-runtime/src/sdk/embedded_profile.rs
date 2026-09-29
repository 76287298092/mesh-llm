//! Per-instance storage for SDK callers that do not supply a durable profile.

use super::EmbeddedMeshNodeConfig;
use anyhow::{Context, Result};
use tempfile::TempDir;

pub(super) fn prepare_isolated_config(
    config: &mut EmbeddedMeshNodeConfig,
) -> Result<Option<TempDir>> {
    if config.storage.config_path.is_some() || !config.storage.isolated_config {
        return Ok(None);
    }
    let directory = tempfile::Builder::new()
        .prefix("mesh-embedded-profile-")
        .tempdir()
        .context("create isolated embedded profile")?;
    let path = directory.path().join("config.toml");
    std::fs::write(
        &path,
        b"[[plugin]]\nname = \"telemetry\"\nenabled = false\n\n[[plugin]]\nname = \"blobstore\"\nenabled = false\n",
    )
    .context("write isolated embedded mesh config")?;
    config.storage.config_path = Some(path);
    Ok(Some(directory))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_instances_have_distinct_profile_roots() {
        let mut first = EmbeddedMeshNodeConfig::default();
        let mut second = EmbeddedMeshNodeConfig::default();
        let a = prepare_isolated_config(&mut first).unwrap().unwrap();
        let b = prepare_isolated_config(&mut second).unwrap().unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(first.storage.config_path.unwrap().parent(), Some(a.path()));
        assert_eq!(second.storage.config_path.unwrap().parent(), Some(b.path()));
        #[cfg(feature = "payments")]
        {
            let first =
                mesh_llm_payments::service::PaymentService::open(&a.path().join("payments"))
                    .expect("first isolated ledger");
            let second =
                mesh_llm_payments::service::PaymentService::open(&b.path().join("payments"))
                    .expect("second isolated ledger must not contend with first");
            drop((first, second));
        }
    }

    #[test]
    fn explicit_profile_is_not_replaced_by_isolation() {
        let path = std::path::PathBuf::from("durable/config.toml");
        let mut config = EmbeddedMeshNodeConfig::builder().config_path(&path).build();
        assert!(prepare_isolated_config(&mut config).unwrap().is_none());
        assert_eq!(config.storage.config_path, Some(path));
    }
}
