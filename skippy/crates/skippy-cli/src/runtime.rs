use crate::cli::NativeRuntimeArgs;
#[cfg(feature = "dynamic-native-runtime")]
use anyhow::Context;
use anyhow::Result;
use skippy_api::native_runtime::NativeRuntimeOptions;
use skippy_commands::runtime::RuntimeRunOptions;
use std::path::{Path, PathBuf};
#[cfg(feature = "dynamic-native-runtime")]
use std::sync::Arc;

pub fn resolve_options(
    args: NativeRuntimeArgs,
    include_adjacent: bool,
) -> Result<NativeRuntimeOptions> {
    let executable = if include_adjacent {
        std::env::current_exe().ok()
    } else {
        None
    };
    resolve_options_from_executable(args, executable.as_deref())
}

fn resolve_options_from_executable(
    args: NativeRuntimeArgs,
    executable: Option<&Path>,
) -> Result<NativeRuntimeOptions> {
    let mut options: NativeRuntimeOptions = args.into();
    if options.cache_dir.is_none() {
        options.cache_dir = Some(skippy_config::paths::native_runtime_cache_default()?);
    }
    options
        .bundle_dirs
        .extend(skippy_config::paths::native_runtime_bundle_dirs_from_env());
    if let Some(executable) = executable
        && let Some(adjacent) = adjacent_runtime_bundle(executable)
    {
        options.bundle_dirs.push(adjacent);
    }
    Ok(options)
}

fn adjacent_runtime_bundle(executable: &Path) -> Option<PathBuf> {
    let root = executable.parent()?.join("native-runtimes");
    root.is_dir().then_some(root)
}

/// Explicit conversion from the resolved native runtime options; a free
/// function because both endpoint types are foreign to this crate (orphan
/// rule).
pub fn command_options(options: &NativeRuntimeOptions) -> RuntimeRunOptions {
    RuntimeRunOptions {
        cache_dir: options.cache_dir.clone(),
        release: options.release.clone(),
        bundle_dirs: options.bundle_dirs.clone(),
        selection: options.selection.clone(),
    }
}

pub fn doctor(options: &NativeRuntimeOptions) -> Result<()> {
    let hardware = skippy_runtime_install::host_runtime_profile();
    let runtime = skippy_api::native_runtime::local_native_runtime_plan(options);
    let runtime_summary = runtime.as_ref().ok().map(|plan| {
        serde_json::json!({
            "id": plan.native_runtime_id,
            "path": plan.root,
        })
    });
    let issue = runtime.err().map(|error| format!("{error:#}"));
    let model_cache = skippy_config::paths::model_cache_dir(None)?;
    let report = serde_json::json!({
        "hardware": hardware,
        "native_runtime": runtime_summary,
        "runtime_issue": issue,
        "runtime_cache": options.cache_dir,
        "model_cache": model_cache,
    });
    skippy_commands::console::present(&report, |output| {
        writeln!(output, "🩺 Skippy doctor")?;
        writeln!(output, "   Machine: {} {}", hardware.os, hardware.arch)?;
        for gpu in &hardware.gpus {
            writeln!(output, "   GPU: {}", gpu.display_name)?;
        }
        if let Some(runtime) = runtime_summary.as_ref() {
            writeln!(
                output,
                "   Runtime: {}",
                runtime["id"].as_str().unwrap_or("unknown")
            )?;
            writeln!(
                output,
                "   Path: {}",
                runtime["path"].as_str().unwrap_or("unknown")
            )?;
        } else {
            writeln!(output, "   Runtime: none compatible")?;
            writeln!(
                output,
                "   Try: skippy runtime install --manifest-url <catalog-url>"
            )?;
        }
        writeln!(output, "   Model cache: {}", model_cache.display())
    })
}

#[cfg(feature = "dynamic-native-runtime")]
pub async fn prepare_native_runtime(options: &NativeRuntimeOptions, automatic: bool) -> Result<()> {
    if skippy_api::native_runtime::local_native_runtime_plan(options).is_err() && automatic {
        use skippy_runtime_install::{
            NativeRuntimeCatalog, NativeRuntimeInstallOptions, RuntimeSelection,
            install_native_runtime_explicit,
        };
        let release = skippy_runtime_install::runtime_release_version();
        let catalog = NativeRuntimeCatalog {
            release_tags: Default::default(),
            releases_url: "https://github.com/Mesh-LLM/mesh-llm/releases".into(),
            rolling_release: None,
        };
        let mut install = NativeRuntimeInstallOptions::new(release, catalog.clone());
        install.manifest_url = Some(catalog.manifest_url(release));
        install.cache_dir = options.cache_dir.clone();
        install.selection = RuntimeSelection::Recommended;
        install.skippy_abi_version = Some(skippy_runtime_install::current_skippy_abi_version());
        install.progress = Some(Arc::new(|progress| {
            if let Some(total) = progress.total_bytes {
                let _ = skippy_commands::console::progress(
                    "Native runtime",
                    progress.downloaded_bytes,
                    total,
                );
            }
        }));
        skippy_commands::console::status("🔎 Selecting a compatible native runtime")?;
        install_native_runtime_explicit(install).await.with_context(|| {
            format!(
                "could not install a released runtime for this Skippy build (ABI {}); run `just skippy` to package a matching local runtime, or pass --runtime-bundle",
                skippy_runtime_install::current_skippy_abi_version()
            )
        })?;
    }
    skippy_api::native_runtime::load_local_native_runtime(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_just_packaged_runtime_beside_executable() {
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("skippy");
        let bundles = root.path().join("native-runtimes");
        assert_eq!(adjacent_runtime_bundle(&executable), None);
        std::fs::create_dir(&bundles).unwrap();
        assert_eq!(adjacent_runtime_bundle(&executable), Some(bundles));
    }

    #[test]
    fn selects_verified_adjacent_runtime_without_a_release_catalog() {
        let root = tempfile::tempdir().unwrap();
        let bundle_root = root.path().join("native-runtimes");
        let bundle = bundle_root.join("test-runtime");
        std::fs::create_dir_all(bundle.join("lib")).unwrap();
        std::fs::write(bundle.join("lib/runtime.bin"), b"fixture, never loaded").unwrap();
        let profile = skippy_runtime_install::host_runtime_profile();
        let manifest: skippy_native_runtime::NativeRuntimeManifest =
            serde_json::from_value(serde_json::json!({"schema_version": 2, "runtime": {
                "id": "adjacent-test-runtime",
                "release_version": skippy_runtime_install::runtime_release_version(),
                "skippy_abi": skippy_runtime_install::current_skippy_abi_version(),
                "platform": {"os": profile.os, "arch": profile.arch,
                    "min_glibc": profile.glibc_version},
                "backend": {"kind": "cpu"},
                "libraries": ["lib/runtime.bin"]
            }}))
            .unwrap();
        manifest.write_to_dir(&bundle).unwrap();
        let args = NativeRuntimeArgs {
            cache_dir: Some(root.path().join("empty-cache")),
            ..Default::default()
        };
        let options =
            resolve_options_from_executable(args, Some(&root.path().join("skippy"))).unwrap();
        let plan = skippy_api::native_runtime::local_native_runtime_plan(&options).unwrap();
        assert_eq!(plan.native_runtime_id, "adjacent-test-runtime");
        assert_eq!(
            plan.root.canonicalize().unwrap(),
            bundle.canonicalize().unwrap()
        );
    }
}
