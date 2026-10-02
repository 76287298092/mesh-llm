//! Standalone access to Skippy-owned package certification and stage caches.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_api::{
    materialized_cache,
    package::{acquisition, certification},
};

#[derive(Debug, Clone)]
pub struct ModelCertificationRequest {
    pub model: String,
    pub report_out: Option<PathBuf>,
    pub package_only: bool,
    pub api_base: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
}

pub(super) fn stage_cache_dir() -> PathBuf {
    skippy_model_hf::application_cache_dir().join("skippy-stages")
}

pub(super) fn prune(yes: bool) -> Result<()> {
    let cache_dir = stage_cache_dir();
    if !yes {
        return crate::console::present(
            &json!({"dry_run": true, "cache_dir": cache_dir, "apply": "skippy models prune --yes"}),
            |output| {
                writeln!(output, "🧹 Derived stage cache prune preview")?;
                writeln!(output, "📁 Cache: {}", cache_dir.display())?;
                writeln!(output, "Apply with: skippy models prune --yes")
            },
        );
    }
    let removed = materialized_cache::prune_unpinned_materialized_stages(&cache_dir)?;
    crate::console::present(
        &json!({"dry_run": false, "cache_dir": cache_dir, "removed_files": removed}),
        |output| {
            writeln!(
                output,
                "✅ Derived stage cache pruned: {removed} file(s) removed"
            )
        },
    )
}

pub(super) async fn certify(request: ModelCertificationRequest) -> Result<()> {
    validate_certification(&request)?;
    let package_ref = resolve_package_ref(&request.model)?;
    let acquisition = acquisition::remote::PackageAcquisition::new(
        skippy_model_hf::huggingface_hub_cache_dir(),
        skippy_model_hf::build_hf_sync_api,
    );
    let report = certification::certify_layer_package(
        certification::SkippyCertificationRequest {
            model_ref: request.model,
            package_only: request.package_only,
            api_base: request.api_base,
            prompt: request.prompt,
            max_tokens: request.max_tokens,
        },
        package_ref,
        acquisition,
        None,
    )
    .await?;
    if let Some(path) = &request.report_out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
    }
    crate::console::present(&report, |output| {
        writeln!(output, "Skippy package certification: {:?}", report.status)?;
        writeln!(output, "Model: {}", report.model_id)?;
        writeln!(output, "Package: {}", report.resolved_package_ref)?;
        writeln!(output, "Manifest: {}", report.manifest_sha256)?;
        writeln!(output, "Layers: {}", report.layer_count)?;
        if let Some(path) = &request.report_out {
            writeln!(output, "Report: {}", path.display())?;
        }
        Ok(())
    })?;
    if report.status != certification::CertificationGateStatus::Passed {
        bail!("skippy package certification {:?}", report.status);
    }
    Ok(())
}

fn resolve_package_ref(model: &str) -> Result<String> {
    if let Ok(parsed) = acquisition::StagePackageRef::parse(model) {
        return parsed
            .as_package_ref()
            .context("direct GGUF inputs are not layer-package certification targets");
    }
    skippy_model_hf::remote_catalog::find_layer_package(model)
        .with_context(|| format!("no layer package found for {model:?}"))
}

fn validate_certification(request: &ModelCertificationRequest) -> Result<()> {
    if request.package_only {
        if request.api_base.is_some() {
            bail!("do not combine --package-only with --api-base");
        }
        return Ok(());
    }
    let api_base = request
        .api_base
        .as_deref()
        .context("models certify requires --package-only or --api-base")?;
    let parsed = reqwest::Url::parse(api_base)
        .with_context(|| format!("invalid --api-base {api_base:?}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        bail!("--api-base must be an http(s) URL with a host");
    }
    if request.prompt.trim().is_empty() || request.max_tokens == 0 {
        bail!("--prompt must not be empty and --max-tokens must be positive");
    }
    Ok(())
}
