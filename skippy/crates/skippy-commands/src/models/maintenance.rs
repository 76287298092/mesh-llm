//! Managed-model cleanup using the shared Skippy/Hugging Face usage ledger.

use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::json;
use skippy_model_hf::store::updates::{self, UpdateEvent};
use skippy_model_hf::store::usage;

pub(super) fn updates(repo: Option<String>, all: bool, check: bool) -> Result<()> {
    let cache_root = skippy_model_hf::huggingface_hub_cache_dir();
    let report = skippy_model_hf::blocking::run_hf_sync({
        let repo = repo.clone();
        move || {
            updates::run_update_in(
                &cache_root,
                repo.as_deref(),
                all,
                check,
                render_update_event,
            )
        }
    })?;
    let output = json!({
        "status": "ok", "mode": if check { "check" } else { "update" },
        "target": { "repo": repo, "all": all },
    });
    crate::console::present(&output, |out| {
        if check {
            writeln!(out, "📬 Checked {} cached repo(s)", report.checked_repos)?;
            writeln!(out, "   Updates available: {}", report.updates_available)?;
        } else {
            writeln!(out, "✅ Update complete")?;
            writeln!(out, "   Refreshed files: {}", report.refreshed_files)?;
            if report.missing_meta > 0 {
                writeln!(out, "   Missing config.json: {}", report.missing_meta)?;
            }
        }
        Ok(())
    })
}

fn render_update_event(event: UpdateEvent) -> Result<()> {
    if crate::console::mode() == crate::console::OutputMode::Jsonl {
        crate::console::event("model_update", &event)?;
        return Ok(());
    }
    if crate::console::mode() != crate::console::OutputMode::Human {
        return Ok(());
    }
    match event {
        UpdateEvent::Empty { cache_dir } => {
            crate::console::status(&format!(
                "📦 No cached Hugging Face model repos found in {}",
                cache_dir.display()
            ))?;
        }
        UpdateEvent::Checking { current, total, .. } => {
            crate::console::progress_with_unit(
                "Checking updates",
                current as u64,
                total as u64,
                "repos",
            )?;
        }
        UpdateEvent::Available {
            repo,
            remote_revision,
            ..
        } => {
            crate::console::status(&format!(
                "🆕 {}: {} → {}",
                repo.repo_id,
                short_revision(&repo.local_revision),
                short_revision(&remote_revision)
            ))?;
            crate::console::status(&format!("   skippy models updates {}", repo.repo_id))?;
        }
        UpdateEvent::Refreshing {
            current,
            total,
            repo,
        } => {
            crate::console::status(&format!(
                "🧭 [{current}/{total}] {}@{}",
                repo.repo_id, repo.ref_name
            ))?;
        }
        UpdateEvent::NoCachedFiles { repo_id } => {
            crate::console::status(&format!("⚠️ {repo_id} has no cached files to refresh"))?;
        }
        UpdateEvent::FileStarted {
            current,
            total,
            file,
        } => {
            crate::console::status(&format!("   ↻ [{current}/{total}] {file}"))?;
        }
        UpdateEvent::FileRefreshed { path } => {
            crate::console::status(&format!("   ✅ {}", path.display()))?;
        }
        UpdateEvent::ConfigMissing { repo_id, error } => {
            if let Some(error) = error {
                crate::console::status(&format!("   ⚠️ {repo_id}/config.json: {error}"))?;
            } else {
                crate::console::status(&format!("   ℹ️ no config.json published for {repo_id}"))?;
            }
        }
    }
    Ok(())
}

fn short_revision(revision: &str) -> &str {
    revision.get(..revision.len().min(12)).unwrap_or(revision)
}

pub(super) fn cleanup(unused_since: Option<&str>, yes: bool) -> Result<()> {
    let age = unused_since.map(parse_age).transpose()?;
    let plan = usage::plan_model_cleanup(age)?;
    let result = yes.then(|| usage::execute_model_cleanup(age)).transpose()?;
    let report = json!({
        "hf_cache_dir": skippy_model_hf::huggingface_hub_cache_dir(),
        "model_cache_dir": skippy_model_hf::application_cache_dir(),
        "mesh_managed_only": true,
        "unused_since": unused_since,
        "dry_run": !yes,
        "plan": plan,
        "result": result,
    });
    crate::console::present(&report, |out| {
        if yes {
            writeln!(out, "✅ Model cleanup complete")?;
        } else {
            writeln!(out, "🧹 Model cleanup preview")?;
        }
        writeln!(out, "🛡️ Scope: managed model files only")?;
        for candidate in &plan.candidates {
            writeln!(out, "📦 {}", candidate.display_name)?;
            writeln!(
                out,
                "   {} file(s), {} bytes",
                candidate.file_count, candidate.total_bytes
            )?;
            writeln!(out, "   {}", candidate.primary_path.display())?;
        }
        if !yes {
            writeln!(out, "Apply with: skippy models cleanup --yes")?;
        }
        Ok(())
    })
}

fn parse_age(value: &str) -> Result<Duration> {
    let value = value.trim();
    let split = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    if split == 0 || split == value.len() {
        bail!("use a cleanup age like 12h, 7d, or 30m");
    }
    let amount: u64 = value[..split].parse()?;
    let factor = match value[split..].to_ascii_lowercase().as_str() {
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 60 * 60,
        "d" | "day" | "days" => 60 * 60 * 24,
        "w" | "week" | "weeks" => 60 * 60 * 24 * 7,
        other => bail!("unsupported cleanup age unit {other:?}; use m, h, d, or w"),
    };
    Ok(Duration::from_secs(amount.saturating_mul(factor)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_age_matches_mesh_units() {
        assert_eq!(parse_age("12h").unwrap(), Duration::from_secs(43_200));
        assert_eq!(parse_age("7d").unwrap(), Duration::from_secs(604_800));
        assert!(parse_age("4x").is_err());
    }
}
