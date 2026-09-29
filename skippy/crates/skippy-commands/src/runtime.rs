//! Standalone native-runtime command execution.

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use skippy_runtime_install::{NativeRuntimeCache, NativeRuntimeManifest};

/// Resolved native runtime inputs for command execution, decoupled from clap.
///
/// The CLI assembles this from parsed arguments plus the `skippy-config`
/// path policy and hands it over with the parsed [`RuntimeAction`].
#[derive(Debug, Clone, Default)]
pub struct RuntimeRunOptions {
    /// Resolved runtime cache root; `None` fails the command at execution.
    pub cache_dir: Option<PathBuf>,
    pub release: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub selection: Option<String>,
}

/// Parsed `skippy runtime` action, decoupled from clap.
#[derive(Debug, Clone)]
pub enum RuntimeAction {
    List,
    /// Install a checksum-verified runtime from an explicit release catalog.
    Install {
        manifest: Option<PathBuf>,
        manifest_url: Option<String>,
    },
    /// Copy a verified bundle into the Skippy cache; leave the source unchanged.
    Import {
        source: PathBuf,
        dry_run: bool,
    },
}

pub async fn run(command: RuntimeAction, options: &RuntimeRunOptions) -> Result<()> {
    let cache = NativeRuntimeCache::new(
        options
            .cache_dir
            .as_ref()
            .context("runtime cache not resolved")?,
    );
    match command {
        RuntimeAction::List => {
            let installed = cache.installed()?;
            crate::console::present(&installed, |output| {
                if installed.is_empty() {
                    writeln!(output, "No native runtimes installed.")?;
                }
                for runtime in &installed {
                    writeln!(
                        output,
                        "⚙️  {} ({})",
                        runtime.native_runtime_id, runtime.flavor
                    )?;
                    writeln!(output, "   {}", runtime.path.display())?;
                }
                Ok(())
            })
        }
        RuntimeAction::Install {
            manifest,
            manifest_url,
        } => {
            use skippy_runtime_install::{
                NativeRuntimeBundleInstallPolicy, NativeRuntimeCatalog,
                NativeRuntimeInstallOptions, RuntimeSelection,
            };
            anyhow::ensure!(
                manifest.is_some() != manifest_url.is_some(),
                "supply exactly one runtime catalog file or URL"
            );
            let release = options
                .release
                .as_deref()
                .unwrap_or(skippy_runtime_install::runtime_release_version());
            // The CLI requires an explicit catalog; this default URL is never used.
            let catalog = NativeRuntimeCatalog {
                releases_url: String::new(),
                release_tags: Default::default(),
                rolling_release: None,
            };
            let mut install = NativeRuntimeInstallOptions::new(release, catalog);
            install.manifest_path = manifest;
            install.manifest_url = manifest_url;
            install.cache_dir = options.cache_dir.clone();
            install.bundle_dirs = options.bundle_dirs.clone();
            install.selection = RuntimeSelection::parse(options.selection.as_deref())?;
            install.skippy_abi_version = Some(skippy_runtime_install::current_skippy_abi_version());
            install.bundle_install_policy =
                NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache;
            install.progress = Some(Arc::new(|progress| {
                if let Some(total) = progress.total_bytes {
                    let _ = crate::console::progress(
                        "Native runtime",
                        progress.downloaded_bytes,
                        total,
                    );
                }
            }));
            let outcome = skippy_runtime_install::install_native_runtime_explicit(install).await?;
            crate::console::present(&outcome, |output| {
                writeln!(
                    output,
                    "✅ Native runtime ready: {}",
                    outcome.runtime.native_runtime_id
                )?;
                writeln!(output, "   {}", outcome.runtime.path.display())
            })
        }
        RuntimeAction::Import { source, dry_run } => {
            let manifest = NativeRuntimeManifest::read_from_dir(&source)?;
            let outcome =
                skippy_runtime_install::import_runtime_copy(&source, &manifest, &cache, dry_run)?;
            crate::console::present(&outcome, |output| {
                if dry_run {
                    writeln!(
                        output,
                        "🔎 Would import runtime to {}",
                        outcome.destination.display()
                    )
                } else {
                    writeln!(
                        output,
                        "✅ Imported runtime to {}",
                        outcome.destination.display()
                    )
                }
            })
        }
    }
}
