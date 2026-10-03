use anyhow::{Context, Result};
use clap::Parser;

mod cli;
mod generation_manifest;
mod glm_dsa_contract;
mod glm_dsa_generation_policy;
mod hash;
mod inspect;
mod package;
mod package_v2;
mod part_writer;
mod progress;
mod source_inventory;
mod tensor_payload;
#[cfg(test)]
mod test_gguf;
mod verify_v2;
mod write;

use cli::{Args, Command};
use package::{ArtifactHook, ExplicitSourceIdentity};

#[cfg(feature = "runtime-dynamic")]
fn native_runtime_library_paths(runtime_root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let root = runtime_root.canonicalize().with_context(|| {
        format!(
            "resolve native runtime directory {}",
            runtime_root.display()
        )
    })?;
    let manifest_path = root.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read native runtime manifest {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse native runtime manifest {}", manifest_path.display()))?;
    let declared_root = manifest
        .pointer("/runtime/libraries")
        .and_then(serde_json::Value::as_array)
        .context("native runtime manifest is missing runtime.libraries")?;
    anyhow::ensure!(
        !declared_root.is_empty(),
        "native runtime manifest declares no libraries"
    );

    let mut paths = Vec::with_capacity(declared_root.len());
    for entry in declared_root {
        let relative = entry
            .as_str()
            .context("native runtime library path must be a string")?;
        let relative_path = std::path::Path::new(relative);
        anyhow::ensure!(
            !relative_path.as_os_str().is_empty()
                && relative_path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
            "native runtime library path must remain package-relative: {relative:?}"
        );
        let candidate = root.join(relative_path);
        let library = candidate
            .canonicalize()
            .with_context(|| format!("resolve native runtime library {}", candidate.display()))?;
        anyhow::ensure!(
            library.starts_with(&root) && library.is_file(),
            "native runtime library is not a file inside {}: {}",
            root.display(),
            library.display()
        );
        paths.push(library);
    }
    Ok(paths)
}

#[cfg(feature = "runtime-dynamic")]
fn load_native_runtime() -> Result<()> {
    let runtime_root = std::env::var_os("MESH_LLM_NATIVE_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .context(
            "MESH_LLM_NATIVE_RUNTIME_DIR must point to a trusted Mesh native runtime directory containing manifest.json",
        )?;
    let libraries = native_runtime_library_paths(&runtime_root)?;
    // The library list comes from the operator-selected, locally installed
    // runtime manifest and is validated to stay inside that runtime directory.
    unsafe { skippy_runtime::load_native_runtime_libraries(&libraries) }
        .with_context(|| format!("load Mesh native runtime from {}", runtime_root.display()))?;
    eprintln!("Loaded Mesh native runtime {}", runtime_root.display());
    Ok(())
}

#[cfg(not(feature = "runtime-dynamic"))]
fn load_native_runtime() -> Result<()> {
    Ok(())
}

fn prepare_model_download_directories() {
    let prepared = match model_hf::prepare_download_directories() {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!(
                "⚠ Unable to prepare model download directories: {error:#}. \
                 Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
            );
            return;
        }
    };
    for fallback in &prepared.fallbacks {
        eprintln!("⚠ {fallback}");
    }
    // SAFETY: runs before any Tokio runtime, process is single-threaded.
    unsafe { prepared.apply_to_process_environment() };
}

// ponytail: main runs on a child thread because the Windows main thread has a
// 1 MB stack. sha256 over a multi-GB GGUF plus FFI slice writing blows that
// stack in debug builds. 8 MB matches the mesh-llm runtime default. If a real
// recursion sink appears, raise this or fix the recursion — don't go lower.
const MAIN_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() -> Result<()> {
    let args = Args::parse();
    load_native_runtime()?;
    // Local inspection and verification must not touch download caches.
    if !matches!(
        args.command,
        Command::Inspect { .. } | Command::VerifyPackageV2 { .. }
    ) {
        prepare_model_download_directories();
    }

    let handle = std::thread::Builder::new()
        .stack_size(MAIN_STACK_SIZE)
        .spawn(move || run(args))
        .context("spawn skippy-model-package worker thread")?;
    handle.join().unwrap_or_else(|panic| {
        std::panic::resume_unwind(panic);
    })
}

fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Inspect { model } => inspect::inspect(model),
        Command::WritePackage {
            model,
            out_dir,
            projectors,
            publisher_metadata,
            after_artifact_command,
            transform_artifact_command,
            model_id,
            source_repo,
            source_revision,
            source_file,
            generation_defaults,
            resume_existing_artifacts,
            max_artifact_bytes,
        } => package_v2::write_package(
            model,
            out_dir,
            package_v2::PackageSidecars {
                projectors,
                publisher_metadata,
            },
            ArtifactHook {
                command: after_artifact_command,
            },
            ArtifactHook {
                command: transform_artifact_command,
            },
            package_v2::PackageWriteOptions {
                explicit: ExplicitSourceIdentity {
                    model_id,
                    source_repo,
                    source_revision,
                    source_file,
                },
                generation_defaults,
                resume_existing_artifacts,
                max_artifact_bytes,
            },
        ),
        Command::VerifyPackageV2 {
            package,
            source,
            source_file,
            source_projectors,
        } => {
            let report = verify_v2::verify_package(
                &package,
                &source,
                source_file.as_deref(),
                &source_projectors,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::ValidateGlmDsaContract {
            package,
            require_generation_policy,
        } => {
            let report = glm_dsa_contract::validate_path_with_options(
                &package,
                glm_dsa_contract::GlmDsaContractOptions {
                    require_generation_policy,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report.valid,
                "GLM-DSA contract validation failed for {}",
                package.display()
            );
            Ok(())
        }
        Command::RepairGlmDsaGenerationPolicy { package, in_place } => {
            glm_dsa_generation_policy::repair_package(&package, in_place)
        }
    }
}

#[cfg(test)]
#[cfg(feature = "runtime-dynamic")]
mod native_runtime_tests {
    use super::native_runtime_library_paths;
    use std::fs;

    #[test]
    fn runtime_paths_follow_manifest_order_and_stay_inside_runtime_root() {
        let root = tempfile::tempdir().expect("create temporary runtime");
        fs::create_dir(root.path().join("lib")).expect("create library directory");
        fs::write(root.path().join("lib/ggml.dll"), []).expect("write dependency fixture");
        fs::write(root.path().join("lib/llama.dll"), []).expect("write primary fixture");
        fs::write(
            root.path().join("manifest.json"),
            br#"{"runtime":{"libraries":["lib/ggml.dll","lib/llama.dll"]}}"#,
        )
        .expect("write runtime manifest");

        let paths = native_runtime_library_paths(root.path()).expect("resolve native libraries");
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("lib/ggml.dll"));
        assert!(paths[1].ends_with("lib/llama.dll"));
    }

    #[test]
    fn runtime_manifest_rejects_parent_traversal() {
        let root = tempfile::tempdir().expect("create temporary runtime");
        fs::write(
            root.path().join("manifest.json"),
            br#"{"runtime":{"libraries":["../outside.dll"]}}"#,
        )
        .expect("write runtime manifest");

        let error = native_runtime_library_paths(root.path()).expect_err("reject path traversal");
        assert!(format!("{error:#}").contains("package-relative"));
    }
}
