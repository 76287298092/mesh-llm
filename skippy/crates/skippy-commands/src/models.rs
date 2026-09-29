use anyhow::{Context, Result, ensure};
use futures::StreamExt;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

/// Parsed `skippy models` action, decoupled from clap.
#[derive(Debug, Clone)]
pub enum ModelAction {
    /// Resolve a Hub revision and download its selected model files.
    Download {
        /// Hub reference: org/repo@revision:filename-or-quantization.
        model_ref: String,
        /// Expected SHA-256 of the primary model file; checked on cache hits too.
        sha256: Option<String>,
        /// Expected byte count of the primary model file.
        size_bytes: Option<u64>,
    },
    /// Remove all cached revisions of one local model repository; never deletes from the Hub.
    Remove {
        repo: String,
        dry_run: bool,
    },
    /// List local model repositories and snapshots without contacting the Hub.
    Installed,
    Recommended,
    Search {
        query: String,
        limit: usize,
    },
    Show {
        model_ref: String,
    },
}

pub struct DownloadedModel {
    pub primary_path: PathBuf,
    pub load_path: PathBuf,
    pub report: serde_json::Value,
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
    validate_digest(sha256)?;
    let _cache_lock = skippy_model_hf::local_cache::lock_cache(cache)?;
    let repository = skippy_model_hf::HfModelRepository::builder()
        .cache_dir(cache)
        .retry_max_attempts(6)
        .retry_base_delay(Duration::from_millis(500))
        .build()?;
    let artifact = skippy_model_artifact::resolve_model_artifact_ref(
        crate::model_catalog::resolve(model_ref),
        &repository,
    )
    .await?;
    crate::console::status(&format!("📦 Resolving {}", artifact.model_id))?;
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
    let report = serde_json::json!({
        "cache_dir": cache, "artifact": artifact, "primary_path": primary_path,
        "load_path": load_path, "files": files
    });
    Ok(DownloadedModel {
        primary_path,
        load_path,
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

pub async fn run(explicit_cache: Option<PathBuf>, command: ModelAction) -> Result<()> {
    let cache = skippy_config::paths::model_cache_dir(explicit_cache)?;
    match command {
        ModelAction::Download {
            model_ref,
            sha256,
            size_bytes,
        } => {
            let downloaded =
                download_model(&cache, &model_ref, sha256.as_deref(), size_bytes).await?;
            crate::console::present(&downloaded.report, |output| {
                writeln!(output, "✅ Model ready: {model_ref}")?;
                writeln!(output, "   {}", downloaded.primary_path.display())
            })
        }
        ModelAction::Remove { repo, dry_run } => {
            let report = skippy_model_hf::local_cache::remove_repository(&cache, &repo, dry_run)?;
            crate::console::present(&report, |output| {
                if dry_run {
                    writeln!(output, "🔎 Would remove cached revisions of {repo}")
                } else {
                    writeln!(output, "🗑️ Removed cached revisions of {repo}")
                }
            })
        }
        ModelAction::Installed => {
            // This operation scans only the explicit local root; it issues no Hub request.
            let _ = skippy_model_hf::configure_hf_tls_provider();
            let client = hf_hub::HFClient::builder().cache_dir(&cache).build()?;
            let scan = client.scan_cache().send().await?;
            let repos = scan.repos.iter().filter(|r| r.repo_type == "model").map(|r| {
                serde_json::json!({"repo": r.repo_id, "path": r.repo_path, "bytes": r.size_on_disk,
                    "revisions": r.revisions.iter().map(|v| serde_json::json!({
                        "revision": v.commit_hash, "path": v.snapshot_path, "refs": v.refs,
                        "files": v.files.iter().map(|f| serde_json::json!({"file":f.file_name,"path":f.file_path,"bytes":f.size_on_disk})).collect::<Vec<_>>()
                    })).collect::<Vec<_>>()})
            }).collect::<Vec<_>>();
            let report = serde_json::json!({"cache_dir":cache,"repositories":repos,"warnings":scan.warnings});
            crate::console::present(&report, |output| {
                if let Some(repositories) = report["repositories"].as_array() {
                    if repositories.is_empty() {
                        writeln!(output, "No models installed.")?;
                    }
                    for repository in repositories {
                        writeln!(
                            output,
                            "📦 {}",
                            repository["repo"].as_str().unwrap_or("unknown")
                        )?;
                    }
                }
                Ok(())
            })
        }
        ModelAction::Recommended => {
            let models = crate::model_catalog::STARTERS;
            crate::console::present(&models, |output| {
                for model in models {
                    writeln!(
                        output,
                        "⭐ {}  ({:.1} GB)",
                        model.name,
                        model.size_bytes as f64 / 1e9
                    )?;
                    writeln!(output, "   {}", model.description)?;
                    writeln!(output, "   skippy serve --model {}", model.name)?;
                }
                Ok(())
            })
        }
        ModelAction::Search { query, limit } => search_models(&query, limit).await,
        ModelAction::Show { model_ref } => show_model(&cache, &model_ref).await,
    }
}

async fn search_models(query: &str, limit: usize) -> Result<()> {
    ensure!(
        limit > 0 && limit <= 100,
        "--limit must be between 1 and 100"
    );
    let client = hf_hub::HFClient::builder().build()?;
    let stream = client
        .list_models()
        .search(query.to_string())
        .filter("gguf")
        .limit(limit)
        .send()?;
    futures::pin_mut!(stream);
    let mut results = Vec::new();
    while let Some(item) = stream.next().await {
        let model = item?;
        results.push(serde_json::json!({"repo":model.id,"downloads":model.downloads}));
    }
    let report = serde_json::json!({"query":query,"results":results});
    crate::console::present(&report, |output| {
        if results.is_empty() {
            writeln!(output, "No GGUF repositories found for {query}.")?;
        }
        for item in &results {
            writeln!(output, "🔎 {}", item["repo"].as_str().unwrap_or("unknown"))?;
        }
        Ok(())
    })
}

async fn show_model(cache: &Path, model_ref: &str) -> Result<()> {
    let repository = skippy_model_hf::HfModelRepository::builder()
        .cache_dir(cache)
        .build()?;
    let artifact = skippy_model_artifact::resolve_model_artifact_ref(
        crate::model_catalog::resolve(model_ref),
        &repository,
    )
    .await?;
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
