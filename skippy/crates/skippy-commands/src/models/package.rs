//! Hugging Face Jobs front end for Skippy layer packages.

use std::{path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use futures::StreamExt;
use serde_json::json;
use skippy_model_package::{
    jobs::{HfJobsClient, JobStage},
    permissions,
    prepare::{self, PrepareParams},
    script,
};

#[derive(Debug, Clone)]
pub struct ModelPackageRequest {
    pub source_repo: Option<String>,
    pub quant: Option<String>,
    pub target: Option<String>,
    pub model_id: Option<String>,
    pub generation_defaults: Option<PathBuf>,
    pub flavor: String,
    pub timeout: String,
    pub mesh_llm_ref: String,
    pub experimental: bool,
    pub dry_run: bool,
    pub confirm: bool,
    pub follow: bool,
    pub status: Option<String>,
    pub logs: Option<String>,
    pub cancel: Option<String>,
    pub list: bool,
    pub update_script: bool,
}

pub(super) async fn run(request: ModelPackageRequest) -> Result<()> {
    if request.update_script {
        let client = skippy_model_package::build_hf_client()?;
        let perms = permissions::check_permissions(&client).await?;
        ensure!(
            perms.is_meshllm_member,
            "only meshllm org members can update the bucket script"
        );
        script::update_bucket_script(&client).await?;
        return crate::console::present(&json!({"updated": true}), |out| {
            writeln!(out, "✅ Bucket script updated")
        });
    }
    if request.status.is_some()
        || request.logs.is_some()
        || request.cancel.is_some()
        || request.list
    {
        return manage_job(&request).await;
    }
    prepare_job(request).await
}

async fn manage_job(request: &ModelPackageRequest) -> Result<()> {
    let jobs = HfJobsClient::from_env()?;
    if let Some(job_id) = request.status.as_deref() {
        let (namespace, id) = parse_job_id(job_id).await?;
        let job = jobs.inspect(&namespace, &id).await?;
        return crate::console::present(&json!({"namespace": namespace, "job": job}), |out| {
            writeln!(out, "Job: {}", job.id)?;
            writeln!(out, "Status: {}", job.status.stage)
        });
    }
    if let Some(job_id) = request.cancel.as_deref() {
        let (namespace, id) = parse_job_id(job_id).await?;
        jobs.cancel(&namespace, &id).await?;
        return crate::console::present(
            &json!({"namespace": namespace, "job_id": id, "canceled": true}),
            |out| writeln!(out, "✅ Job {id} canceled"),
        );
    }
    if let Some(job_id) = request.logs.as_deref() {
        ensure!(
            crate::console::mode() != crate::console::OutputMode::Json,
            "streamed job logs require --output human or --output jsonl"
        );
        let (namespace, id) = parse_job_id(job_id).await?;
        let stream = jobs.stream_logs(&namespace, &id).await?;
        futures::pin_mut!(stream);
        while let Some(line) = stream.next().await {
            let line = line?;
            if crate::console::mode() == crate::console::OutputMode::Human {
                crate::console::write_line(&line)?;
            } else {
                crate::console::event(
                    "log",
                    &json!({"namespace": namespace, "job_id": id, "text": line}),
                )?;
            }
        }
        return Ok(());
    }
    let client = skippy_model_package::build_hf_client()?;
    let perms = permissions::check_permissions(&client).await?;
    let listed = jobs.list(&perms.namespace).await?;
    crate::console::present(
        &json!({"namespace": perms.namespace, "jobs": listed}),
        |out| {
            if listed.is_empty() {
                writeln!(out, "No jobs found in namespace '{}'.", perms.namespace)?;
            }
            for job in &listed {
                writeln!(out, "{}  {}", job.id, job.status.stage)?;
            }
            Ok(())
        },
    )
}

async fn prepare_job(request: ModelPackageRequest) -> Result<()> {
    let source_ref = request
        .source_repo
        .as_deref()
        .context("source repo is required for job submission")?;
    let source = skippy_model_ref::ModelRef::parse(source_ref)
        .with_context(|| format!("invalid source model ref: {source_ref}"))?;
    let quant = match (source.selector.as_deref(), request.quant.as_deref()) {
        (Some(left), Some(right)) if left != right => {
            bail!("source selector {left:?} conflicts with --quant {right:?}")
        }
        (Some(selector), _) | (_, Some(selector)) => Some(selector.to_string()),
        (None, None) => None,
    };
    let client = skippy_model_package::build_hf_client()?;
    let Some(quant) = quant else {
        let inventory =
            prepare::list_inventory(&client, &source.repo, source.revision.as_deref()).await?;
        return crate::console::present(
            &json!({
                "source_repo": source.repo, "source_revision": source.revision,
                "quants": inventory.quants, "projectors": inventory.projectors,
            }),
            |out| {
                writeln!(out, "📦 Available quants in {}:", source.repo)?;
                for variant in &inventory.quants {
                    writeln!(
                        out,
                        "   {}  {} file(s)  {}",
                        variant.name,
                        variant.shard_count,
                        prepare::format_size(variant.total_bytes)
                    )?;
                }
                Ok(())
            },
        );
    };
    ensure!(
        !request.follow || request.confirm && !request.dry_run,
        "--follow requires --confirm without --dry-run"
    );
    ensure!(
        !request.follow || crate::console::mode() != crate::console::OutputMode::Json,
        "--follow requires --output human or --output jsonl"
    );
    let perms = permissions::check_permissions(&client).await?;
    let jobs = (request.confirm && !request.dry_run)
        .then(HfJobsClient::from_env)
        .transpose()?;
    let defaults = request
        .generation_defaults
        .as_deref()
        .map(read_generation_defaults)
        .transpose()?;
    let params = PrepareParams {
        source_repo: source.repo,
        source_revision: source.revision,
        quant: Some(quant),
        target: request.target,
        model_id: request.model_id,
        generation_defaults: defaults,
        flavor: request.flavor,
        timeout_seconds: parse_timeout(&request.timeout)?,
        mesh_llm_ref: request.mesh_llm_ref,
        experimental: request.experimental,
        hf_token: jobs.as_ref().map(|jobs| jobs.token().to_string()),
    };
    let job = prepare::resolve(&client, params, &perms).await?;
    if let Some(jobs) = jobs {
        ensure_bucket_script_current(&client).await?;
        let info = jobs.submit(&job.namespace, &job.spec).await?;
        let url = format!("{}/jobs/{}/{}", jobs.endpoint(), job.namespace, info.id);
        crate::console::present(
            &json!({
                "submitted": true, "job": info, "job_url": url, "namespace": job.namespace,
                "source_repo": job.source_repo, "target_repo": job.target_repo,
            }),
            |out| {
                writeln!(out, "🚀 Submitted: {}", info.id)?;
                writeln!(out, "   Console: {url}")
            },
        )?;
        if request.follow {
            follow_job(&jobs, &job.namespace, &info.id).await?;
        }
        return Ok(());
    }
    let mut spec = job.spec.clone();
    for value in spec.secrets.values_mut() {
        *value = "****".to_string();
    }
    crate::console::present(
        &json!({
            "dry_run": true, "confirm_required": true, "source_repo": job.source_repo,
            "source_revision": job.source_revision, "source_file": job.source_file,
            "projectors": job.projectors, "target_repo": job.target_repo,
            "model_id": job.model_id, "experimental": job.experimental,
            "generation_defaults": job.generation_defaults, "job_plan": job.job_plan,
            "spec": spec,
        }),
        |out| {
            writeln!(out, "🔍 Dry run — no HF Job was submitted")?;
            writeln!(
                out,
                "   Source: {}@{}/{}",
                job.source_repo, job.source_revision, job.source_file
            )?;
            writeln!(out, "   Target: {}", job.target_repo)?;
            writeln!(out, "   Add --confirm to submit")
        },
    )
}

fn read_generation_defaults(
    path: &std::path::Path,
) -> Result<skippy_package_format::GenerationRequestDefaults> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read generation defaults {}", path.display()))?;
    let defaults: skippy_package_format::GenerationRequestDefaults =
        serde_json::from_slice(&bytes)?;
    defaults.validate()?;
    Ok(defaults)
}

async fn parse_job_id(value: &str) -> Result<(String, String)> {
    if let Some((namespace, id)) = value.split_once('/') {
        return Ok((namespace.to_string(), id.to_string()));
    }
    let client = skippy_model_package::build_hf_client()?;
    let perms = permissions::check_permissions(&client).await?;
    Ok((perms.namespace, value.to_string()))
}

async fn ensure_bucket_script_current(client: &hf_hub::HFClient) -> Result<()> {
    let freshness = script::check_bucket_script(client).await;
    if freshness.is_ok_and(|freshness| freshness.is_current) {
        return Ok(());
    }
    script::update_bucket_script(client).await
}

async fn follow_job(client: &HfJobsClient, namespace: &str, id: &str) -> Result<()> {
    loop {
        let info = client.inspect(namespace, id).await?;
        if info.status.stage.is_terminal() {
            ensure!(
                info.status.stage == JobStage::Completed,
                "job {id} finished: {}",
                info.status.stage
            );
            return Ok(());
        }
        if info.status.stage == JobStage::Running {
            let stream = client.stream_logs(namespace, id).await?;
            futures::pin_mut!(stream);
            while let Some(line) = stream.next().await {
                let line = line?;
                if crate::console::mode() == crate::console::OutputMode::Human {
                    crate::console::write_line(&line)?;
                } else {
                    crate::console::event("log", &json!({"job_id": id, "text": line}))?;
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

fn parse_timeout(input: &str) -> Result<u64> {
    let input = input.trim();
    if let Ok(seconds) = input.parse::<u64>() {
        ensure!(seconds > 0, "timeout must be positive");
        return Ok(seconds);
    }
    let mut total = 0u64;
    let mut digits = String::new();
    for character in input.chars() {
        if character.is_ascii_digit() {
            digits.push(character);
            continue;
        }
        let value: u64 = digits
            .parse()
            .with_context(|| format!("invalid timeout {input:?}"))?;
        digits.clear();
        let factor = match character.to_ascii_lowercase() {
            'h' => 3600,
            'm' => 60,
            's' => 1,
            _ => bail!("invalid timeout unit {character:?}"),
        };
        total = total
            .checked_add(value.checked_mul(factor).context("timeout overflow")?)
            .context("timeout overflow")?;
    }
    if !digits.is_empty() {
        total = total
            .checked_add(digits.parse()?)
            .context("timeout overflow")?;
    }
    ensure!(total > 0, "timeout must be positive");
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::parse_timeout;

    #[test]
    fn package_timeout_matches_mesh_syntax() {
        assert_eq!(parse_timeout("3h").unwrap(), 10_800);
        assert_eq!(parse_timeout("2h30m").unwrap(), 9_000);
        assert_eq!(parse_timeout("7200").unwrap(), 7_200);
        assert!(parse_timeout("0").is_err());
        assert!(parse_timeout("4x").is_err());
    }
}
