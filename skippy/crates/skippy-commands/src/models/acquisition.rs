//! Model-command download policy, including catalog layer-package substitution.

use std::path::Path;

use anyhow::Result;
use serde_json::json;
use skippy_api::package::acquisition;

use super::download_model;

pub(super) async fn download_command(
    cache: &Path,
    model_ref: &str,
    include_draft: bool,
    direct: bool,
) -> Result<()> {
    if !direct && let Some(package_ref) = layer_package_ref(model_ref) {
        let package_dir = tokio::task::spawn_blocking({
            let package_ref = package_ref.clone();
            let cache = cache.to_path_buf();
            move || {
                acquisition::remote::PackageAcquisition::new(
                    cache,
                    skippy_model_hf::build_hf_sync_api,
                )
                .download_package_v2_to_local(&package_ref)
            }
        })
        .await??;
        if include_draft {
            crate::console::status("⚠ Draft download is not available for layer packages")?;
        }
        return crate::console::present(
            &json!({
                "requested_ref": model_ref,
                "type": "layer_package",
                "package_ref": package_ref,
                "path": package_dir,
            }),
            |out| {
                writeln!(out, "✅ Layer package ready: {package_ref}")?;
                writeln!(out, "   {}", package_dir.display())
            },
        );
    }

    let mut downloaded = download_model(cache, model_ref, None, None).await?;
    if include_draft {
        let draft = recommended_draft(model_ref, &downloaded.report);
        downloaded.report["draft"] = if let Some(draft_ref) = draft {
            let draft_download = download_model(cache, &draft_ref, None, None).await?;
            json!({"name": draft_ref, "path": draft_download.primary_path})
        } else {
            crate::console::status(&format!("⚠ No draft model available for {model_ref}"))?;
            serde_json::Value::Null
        };
    }
    crate::console::present(&downloaded.report, |out| {
        writeln!(out, "✅ Model ready: {model_ref}")?;
        writeln!(out, "   {}", downloaded.primary_path.display())
    })
}

fn layer_package_ref(model_ref: &str) -> Option<String> {
    if acquisition::is_layer_package_ref(model_ref) {
        return Some(model_ref.to_string());
    }
    skippy_model_hf::remote_catalog::resolve_layer_package_download(model_ref)
}

fn recommended_draft(model_ref: &str, report: &serde_json::Value) -> Option<String> {
    let catalog = skippy_model_hf::remote_catalog::find_model_exact(model_ref).or_else(|| {
        let artifact = &report["artifact"];
        skippy_model_hf::remote_catalog::matching_model_for_huggingface(
            artifact["source_repo"].as_str()?,
            artifact["source_revision"].as_str(),
            artifact["primary_file"].as_str()?,
        )
    })?;
    let draft = catalog.draft?;
    Some(
        skippy_model_hf::remote_catalog::find_model_exact(&draft)
            .map(|model| model.exact_ref())
            .unwrap_or(draft),
    )
}
