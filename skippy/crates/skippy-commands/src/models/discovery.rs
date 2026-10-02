//! Search presentation over the shared Hugging Face repository query policy.

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use skippy_model_artifact::{ModelArtifactFile, selection};
use skippy_model_hf::search::{ArtifactFilter, Sort};

#[derive(Debug, Clone)]
pub struct ModelSearchRequest {
    pub query: Vec<String>,
    pub filter: ArtifactFilter,
    pub catalog_only: bool,
    pub limit: usize,
    pub sort: Sort,
}

pub(super) async fn search_models(request: ModelSearchRequest) -> Result<()> {
    ensure!(
        request.limit > 0 && request.limit <= 100,
        "--limit must be between 1 and 100"
    );
    let query = request.query.join(" ");
    let filter = match request.filter {
        ArtifactFilter::Gguf => "gguf",
        ArtifactFilter::Mlx => "mlx",
    };
    let sort = sort_name(request.sort);
    let (source, results) = if request.catalog_only {
        (
            "catalog",
            catalog_results(&query, request.filter, request.limit)?,
        )
    } else {
        crate::console::status(&format!("🔎 Searching Hugging Face {filter} repositories"))?;
        let repos = skippy_model_hf::search::search_repositories(
            &query,
            request.limit,
            request.filter,
            request.sort,
        )
        .await?;
        (
            "huggingface",
            hub_results(repos, request.filter, request.sort, request.limit),
        )
    };
    let report = json!({
        "query": query, "filter": filter, "sort": sort, "source": source,
        "results": results,
    });
    crate::console::present(&report, |out| {
        if results.is_empty() {
            writeln!(out, "No {filter} models found for {query}.")?;
        }
        for result in &results {
            let model_ref = result["ref"].as_str().unwrap_or("unknown");
            let size = result["size"].as_str().unwrap_or("unknown size");
            writeln!(out, "🔎 {model_ref}  {size}")?;
            writeln!(out, "   skippy models show {model_ref}")?;
        }
        Ok(())
    })
}

fn catalog_results(query: &str, filter: ArtifactFilter, limit: usize) -> Result<Vec<Value>> {
    skippy_model_hf::remote_catalog::ensure_catalog()?;
    let query = query.to_ascii_lowercase();
    let mut models = skippy_model_hf::remote_catalog::loaded_models()?
        .into_iter()
        .filter(|model| {
            let is_mlx = model.source_file().ends_with(".safetensors")
                || model.source_file().ends_with(".safetensors.index.json");
            is_mlx == (filter == ArtifactFilter::Mlx)
                && (model.name.to_ascii_lowercase().contains(&query)
                    || model.file.to_ascii_lowercase().contains(&query)
                    || model
                        .description
                        .as_deref()
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                        .contains(&query))
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models
        .into_iter()
        .take(limit)
        .map(|model| {
            let model_ref = model.exact_ref();
            json!({
                "name": model.name, "repo_id": model.source_repo(),
                "type": if filter == ArtifactFilter::Mlx { "mlx" } else { "gguf" },
                "size": model.size, "description": model.description,
                "draft": model.draft, "ref": model_ref,
                "show": format!("skippy models show {model_ref}"),
                "download": format!("skippy models download {model_ref}"),
            })
        })
        .collect())
}

fn hub_results(
    repos: Vec<hf_hub::repository::ModelInfo>,
    filter: ArtifactFilter,
    sort: Sort,
    limit: usize,
) -> Vec<Value> {
    let mut hits = repos
        .into_iter()
        .filter_map(|repo| {
            let siblings = repo.siblings.as_deref()?;
            let files = siblings
                .iter()
                .map(|sibling| ModelArtifactFile {
                    path: sibling.rfilename.clone(),
                    size_bytes: sibling.size,
                    sha256: None,
                })
                .collect::<Vec<_>>();
            let file = select_search_file(&files, filter)?;
            let model_ref = search_model_ref(&repo.id, &file.path, filter);
            let size_bytes = if filter == ArtifactFilter::Gguf {
                selection::gguf_variant_size_bytes(&file.path, &files)
            } else {
                file.size_bytes
            };
            let variants = (filter == ArtifactFilter::Gguf).then(|| {
                files
                    .iter()
                    .filter_map(|file| skippy_model_ref::quant_selector_from_gguf_file(&file.path))
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            });
            Some(json!({
                "repo_id": repo.id,
                "repo_url": format!("https://huggingface.co/{}", repo.id),
                "type": if filter == ArtifactFilter::Mlx { "mlx" } else { "gguf" },
                "variant_count": variants, "size": size_bytes.map(format_size),
                "downloads": repo.downloads, "likes": repo.likes,
                "ref": model_ref,
                "show": format!("skippy models show {model_ref}"),
                "download": format!("skippy models download {model_ref}"),
            }))
        })
        .collect::<Vec<_>>();
    if matches!(sort, Sort::ParametersDesc | Sort::ParametersAsc) {
        hits.sort_by(|left, right| {
            let count = |hit: &Value| {
                skippy_model_hf::search::approximate_parameter_count_b_from_text(&format!(
                    "{} {}",
                    hit["repo_id"], hit["ref"]
                ))
                .unwrap_or(-1.0)
            };
            let order = count(left)
                .partial_cmp(&count(right))
                .unwrap_or(std::cmp::Ordering::Equal);
            if sort == Sort::ParametersDesc {
                order.reverse()
            } else {
                order
            }
        });
    }
    hits.truncate(limit);
    hits
}

fn select_search_file(
    files: &[ModelArtifactFile],
    filter: ArtifactFilter,
) -> Option<ModelArtifactFile> {
    if filter == ArtifactFilter::Gguf {
        return selection::select_default_gguf_file(files, 0);
    }
    files
        .iter()
        .find(|file| file.path == "model.safetensors")
        .or_else(|| {
            files.iter().find(|file| {
                file.path.starts_with("model-00001-of-") && file.path.ends_with(".safetensors")
            })
        })
        .cloned()
}

fn search_model_ref(repo: &str, file: &str, filter: ArtifactFilter) -> String {
    if filter == ArtifactFilter::Mlx {
        return repo.to_string();
    }
    if let Some(quant) = skippy_model_ref::quant_selector_from_gguf_file(file) {
        return format!("{repo}:{quant}");
    }
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    let stem = stem
        .split_once("-00001-of-")
        .map_or(stem, |(prefix, _)| prefix);
    format!("{repo}/{stem}")
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0}MB", bytes as f64 / 1e6)
    }
}

fn sort_name(sort: Sort) -> &'static str {
    match sort {
        Sort::Trending => "trending",
        Sort::Downloads => "downloads",
        Sort::Likes => "likes",
        Sort::Created => "created",
        Sort::Updated => "updated",
        Sort::ParametersDesc => "parameters-desc",
        Sort::ParametersAsc => "parameters-asc",
    }
}
