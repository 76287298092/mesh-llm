use anyhow::{Context, Result, ensure};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

mod acquisition;
mod discovery;
mod maintenance;
mod package;
mod stage_cache;

pub use discovery::ModelSearchRequest;
pub use package::ModelPackageRequest;
pub use stage_cache::ModelCertificationRequest;

/// Parsed `skippy models` action, decoupled from clap.
#[derive(Debug, Clone)]
pub enum ModelAction {
    Package(ModelPackageRequest),
    Certify(ModelCertificationRequest),
    Prune {
        yes: bool,
    },
    Cleanup {
        unused_since: Option<String>,
        yes: bool,
    },
    /// Resolve a Hub revision and download its selected model files.
    Download {
        /// Hub reference: org/repo@revision:filename-or-quantization.
        model_ref: String,
        draft: bool,
        direct: bool,
    },
    /// Preview or remove exactly one installed model, using Mesh's shared delete policy.
    Delete {
        model: String,
        yes: bool,
    },
    /// List local model repositories and snapshots without contacting the Hub.
    Installed,
    Recommended,
    Search(ModelSearchRequest),
    Updates {
        repo: Option<String>,
        all: bool,
        check: bool,
    },
    Show {
        model_ref: String,
    },
}

pub struct DownloadedModel {
    pub primary_path: PathBuf,
    pub load_path: PathBuf,
    pub projector_path: Option<PathBuf>,
    pub report: serde_json::Value,
}

/// Mesh and Skippy use the same Hugging Face cache resolution policy.
pub fn model_cache_dir() -> PathBuf {
    skippy_model_hf::huggingface_hub_cache_dir()
}

struct DownloadProgress {
    last_percent: Mutex<Option<u64>>,
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let (current, total) = match event {
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => (*bytes_completed, *total_bytes),
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                let Some(file) = files.last() else { return };
                (file.bytes_completed, file.total_bytes)
            }
            _ => return,
        };
        if total == 0 {
            return;
        }
        let percent = current.min(total).saturating_mul(100) / total;
        let Ok(mut last) = self.last_percent.lock() else {
            return;
        };
        if *last == Some(percent) {
            return;
        }
        *last = Some(percent);
        let _ = crate::console::progress("Model download", current, total);
    }
}

pub async fn download_model(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
) -> Result<DownloadedModel> {
    let mut downloaded = download_primary_model(cache, model_ref, sha256, size_bytes).await?;
    let artifact = &downloaded.report["artifact"];
    let catalog = skippy_model_hf::remote_catalog::find_model_exact(model_ref).or_else(|| {
        let repo = artifact["source_repo"].as_str()?;
        let revision = artifact["source_revision"].as_str();
        let file = artifact["primary_file"].as_str()?;
        skippy_model_hf::remote_catalog::matching_model_for_huggingface(repo, revision, file)
    });
    if let Some(asset) = catalog.and_then(|model| model.mmproj) {
        let revision = asset
            .revision
            .as_deref()
            .map(|revision| format!("@{revision}"))
            .unwrap_or_default();
        let projector_ref = format!("{}{}/{}", asset.repo, revision, asset.source_file);
        let projector = download_primary_model(cache, &projector_ref, None, None).await?;
        downloaded.report["projector_path"] = serde_json::json!(projector.primary_path);
        downloaded.projector_path = Some(projector.primary_path);
    }
    Ok(downloaded)
}

async fn download_primary_model(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
) -> Result<DownloadedModel> {
    validate_digest(sha256)?;
    crate::console::status(&format!("📦 Resolving {model_ref}"))?;
    let _cache_lock = skippy_model_hf::local_cache::lock_cache(cache)?;
    let repository = skippy_model_hf::HfModelRepository::builder()
        .cache_dir(cache)
        .retry_max_attempts(6)
        .retry_base_delay(Duration::from_millis(500))
        .build()?;
    let resolved_ref = skippy_model_hf::remote_catalog::find_model_exact(model_ref)
        .map(|model| model.exact_ref())
        .or_else(|| {
            (!model_ref.contains('/'))
                .then(|| skippy_model_hf::remote_catalog::resolve_model_download(model_ref))
                .flatten()
                .map(|model| {
                    let revision = model
                        .revision
                        .as_deref()
                        .map(|revision| format!("@{revision}"))
                        .unwrap_or_default();
                    format!("{}{}/{}", model.repo, revision, model.file)
                })
        })
        .unwrap_or_else(|| model_ref.to_string());
    let artifact =
        skippy_model_artifact::resolve_model_artifact_ref(&resolved_ref, &repository).await?;
    let progress = hf_hub::progress::Progress::new(DownloadProgress {
        last_percent: Mutex::new(None),
    });
    let paths = if artifact.format == skippy_model_artifact::ModelFormat::Safetensors {
        repository
            .download_checkpoint_with_progress(&artifact, Some(progress))
            .await?
            .into_iter()
            .map(|downloaded| (downloaded.file, downloaded.path))
            .collect::<Vec<_>>()
    } else {
        let downloaded_paths = repository
            .download_artifact_files_with_progress(&artifact, Some(progress))
            .await?;
        ensure!(
            downloaded_paths.len() == artifact.files.len(),
            "downloaded artifact file count mismatch"
        );
        downloaded_paths
            .into_iter()
            .zip(artifact.files.iter().cloned())
            .map(|(path, file)| (file, path))
            .collect::<Vec<_>>()
    };
    ensure!(!paths.is_empty(), "downloaded artifact file list is empty");
    let mut files = Vec::with_capacity(paths.len());
    let mut managed_paths = Vec::with_capacity(paths.len());
    let mut primary_path = None;
    for (file, path) in paths {
        let primary = file.path == artifact.primary_file;
        let expected_size = if primary {
            size_bytes.or(file.size_bytes)
        } else {
            file.size_bytes
        };
        let expected_sha = if primary {
            sha256.or(file.sha256.as_deref())
        } else {
            file.sha256.as_deref()
        };
        files.push(verify_file(&path, expected_size, expected_sha)?);
        managed_paths.push(path.clone());
        if primary {
            primary_path = Some(path);
        }
    }
    let primary_path = primary_path.context("download did not include the primary model file")?;
    let load_path = if artifact.format == skippy_model_artifact::ModelFormat::Safetensors {
        primary_path
            .parent()
            .context("SafeTensors checkpoint has no parent directory")?
            .to_path_buf()
    } else {
        primary_path.clone()
    };
    if let Err(error) = skippy_model_hf::store::usage::track_managed_model_usage(
        &primary_path,
        &managed_paths,
        &artifact.model_id,
        Some(&artifact.canonical_ref),
        "huggingface",
    ) {
        crate::console::status(&format!("⚠ Could not record model usage: {error:#}"))?;
    }
    let report = serde_json::json!({
        "cache_dir": cache, "artifact": artifact, "primary_path": primary_path,
        "load_path": load_path, "files": files
    });
    Ok(DownloadedModel {
        primary_path,
        load_path,
        projector_path: None,
        report,
    })
}

fn validate_digest(digest: Option<&str>) -> Result<()> {
    if let Some(digest) = digest {
        ensure!(
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "--sha256 must be exactly 64 hexadecimal characters"
        );
    }
    Ok(())
}

fn verify_file(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha: Option<&str>,
) -> Result<serde_json::Value> {
    validate_digest(expected_sha)?;
    let bytes = path
        .metadata()
        .with_context(|| format!("inspect downloaded file {}", path.display()))?
        .len();
    if let Some(expected) = expected_size {
        ensure!(
            bytes == expected,
            "downloaded file size mismatch for {}: expected {expected}, got {bytes}",
            path.display()
        );
    }
    let digest = skippy_api::package::file_sha256(path)?;
    if let Some(expected) = expected_sha {
        ensure!(
            digest.eq_ignore_ascii_case(expected),
            "downloaded file SHA-256 mismatch for {}",
            path.display()
        );
    }
    Ok(
        serde_json::json!({"path": path, "bytes": bytes, "sha256": digest, "expected_sha256_verified": expected_sha.is_some()}),
    )
}

pub async fn run(command: ModelAction) -> Result<()> {
    let cache = model_cache_dir();
    match command {
        ModelAction::Package(request) => package::run(request).await,
        ModelAction::Certify(request) => stage_cache::certify(request).await,
        ModelAction::Prune { yes } => stage_cache::prune(yes),
        ModelAction::Cleanup { unused_since, yes } => {
            maintenance::cleanup(unused_since.as_deref(), yes)
        }
        ModelAction::Download {
            model_ref,
            draft,
            direct,
        } => acquisition::download_command(&cache, &model_ref, draft, direct).await,
        ModelAction::Delete { model, yes } => delete_model(&model, yes).await,
        ModelAction::Installed => {
            let artifacts = skippy_model_hf::store::installed::scan_installed_artifacts_in(&cache);
            let report = serde_json::json!({"cache_dir": cache, "results": artifacts});
            crate::console::present(&report, |output| {
                if artifacts.is_empty() {
                    writeln!(output, "No models installed.")?;
                }
                for artifact in &artifacts {
                    writeln!(output, "📦 {}", artifact.model_ref)?;
                    writeln!(output, "   {}", artifact.path.display())?;
                }
                Ok(())
            })
        }
        ModelAction::Recommended => {
            skippy_model_hf::remote_catalog::ensure_catalog()?;
            let models = skippy_model_hf::remote_catalog::loaded_models()?;
            let report = serde_json::json!({
                "source": "catalog",
                "results": models.iter().map(|model| serde_json::json!({
                    "name": model.name,
                    "size": model.size,
                    "description": model.description,
                    "draft": model.draft,
                    "ref": model.exact_ref(),
                })).collect::<Vec<_>>()
            });
            crate::console::present(&report, |output| {
                for model in &models {
                    writeln!(
                        output,
                        "⭐ {}  {}",
                        model.name,
                        model.size.as_deref().unwrap_or("unknown size")
                    )?;
                    if let Some(description) = &model.description {
                        writeln!(output, "   {description}")?;
                    }
                    writeln!(output, "   skippy serve --model {}", model.exact_ref())?;
                }
                Ok(())
            })
        }
        ModelAction::Search(request) => discovery::search_models(request).await,
        ModelAction::Updates { repo, all, check } => maintenance::updates(repo, all, check),
        ModelAction::Show { model_ref } => show_model(&cache, &model_ref).await,
    }
}

async fn delete_model(model: &str, yes: bool) -> Result<()> {
    use skippy_model_hf::store::delete::{self, CuratedDeleteCatalog};

    let paths = delete::resolve_model_identifier_with_catalog(model, &CuratedDeleteCatalog).await?;
    ensure!(!paths.is_empty(), "Model not found: {model}");
    let stage_cache = stage_cache::stage_cache_dir();
    let derived_stage_paths =
        skippy_api::materialized_cache::materialized_stages_for_sources(&stage_cache, &paths)?;
    if !yes {
        let report = serde_json::json!({
            "model": model, "paths": paths,
            "derived_stage_paths": derived_stage_paths, "dry_run": true
        });
        return crate::console::present(&report, |output| {
            writeln!(
                output,
                "🔎 Would delete {} local model file(s):",
                paths.len()
            )?;
            for path in &paths {
                writeln!(output, "   {}", path.display())?;
            }
            if !derived_stage_paths.is_empty() {
                writeln!(
                    output,
                    "   Also remove {} derived stage file(s)",
                    derived_stage_paths.len()
                )?;
            }
            writeln!(output, "   Run with --yes to delete them")
        });
    }
    let removed_derived_cache_files =
        skippy_api::materialized_cache::remove_materialized_stages_for_sources(
            &stage_cache,
            &paths,
        )?;
    let result = delete::delete_model_by_identifier_with_catalog_in(
        model,
        &CuratedDeleteCatalog,
        &skippy_model_hf::application_cache_dir(),
    )
    .await?;
    let report = serde_json::json!({
        "model": model,
        "deleted_paths": result.deleted_paths,
        "reclaimed_bytes": result.reclaimed_bytes,
        "removed_metadata_files": result.removed_metadata_files,
        "removed_usage_records": result.removed_usage_records,
        "removed_derived_cache_files": removed_derived_cache_files,
        "dry_run": false,
    });
    crate::console::present(&report, |output| {
        writeln!(
            output,
            "🗑️ Deleted {} local model file(s)",
            result.deleted_paths.len()
        )
    })
}

async fn show_model(cache: &Path, model_ref: &str) -> Result<()> {
    let repository = skippy_model_hf::HfModelRepository::builder()
        .cache_dir(cache)
        .build()?;
    let resolved_ref = skippy_model_hf::remote_catalog::find_model_exact(model_ref)
        .map(|model| model.exact_ref())
        .unwrap_or_else(|| model_ref.to_string());
    let artifact =
        skippy_model_artifact::resolve_model_artifact_ref(&resolved_ref, &repository).await?;
    crate::console::present(&artifact, |output| {
        writeln!(output, "📦 {model_ref}")?;
        writeln!(output, "   Repository: {}", artifact.source_repo)?;
        writeln!(output, "   Revision: {}", artifact.source_revision)?;
        writeln!(output, "   Primary file: {}", artifact.primary_file)?;
        writeln!(output, "   Files: {}", artifact.files.len())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_file_is_rehashed_and_rejects_corruption() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        std::fs::write(&path, b"first").unwrap();
        let digest = skippy_api::package::file_sha256(&path).unwrap();
        assert!(verify_file(&path, Some(5), Some(&digest)).is_ok());
        std::fs::write(&path, b"other").unwrap();
        assert!(
            verify_file(&path, Some(5), Some(&digest))
                .unwrap_err()
                .to_string()
                .contains("SHA-256 mismatch")
        );
        assert!(
            verify_file(&path, Some(6), None)
                .unwrap_err()
                .to_string()
                .contains("size mismatch")
        );
    }

    #[test]
    fn invalid_digest_is_rejected_before_file_access() {
        assert!(
            verify_file(Path::new("/nonexistent/model.gguf"), None, Some("bad"))
                .unwrap_err()
                .to_string()
                .contains("64 hexadecimal")
        );
    }
}
