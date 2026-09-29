//! Exercise the standalone command's output streams without loading a model.
#[test]
fn example_config_is_one_json_document_on_stdout() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .arg("example-config")
        .output()
        .expect("start standalone command");
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    let config: skippy_protocol::StageConfig =
        serde_json::from_slice(&output.stdout).expect("stdout contains only stage config JSON");
    assert!(!config.stage_id.is_empty());
}

#[cfg(feature = "dynamic-native-runtime")]
#[test]
fn standalone_rejects_missing_runtime_before_reading_stage_config() {
    let temp = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
            "/nonexistent/mesh-runtime",
        )
        .args([
            "--output",
            "human",
            "--runtime-release",
            "999.999.999-test",
            "--runtime-cache",
        ])
        .arg(temp.path())
        .args(["serve", "--config", "/nonexistent/skippy-stage.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("no compatible local Skippy runtime"),
        "{error}"
    );
    assert!(!error.contains("load stage config"), "{error}");
    assert!(std::fs::read_dir(temp.path()).unwrap().next().is_none());
}

#[test]
fn standalone_selection_uses_verified_bundle_and_rejects_abi_mismatch() {
    use skippy_api::native_runtime::{NativeRuntimeOptions, local_native_runtime_plan};
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("runtime");
    std::fs::create_dir_all(bundle.join("lib")).unwrap();
    std::fs::write(bundle.join("lib/runtime.bin"), b"fixture, never executed").unwrap();
    let profile = skippy_runtime_install::host_runtime_profile();
    let mut manifest: skippy_native_runtime::NativeRuntimeManifest = serde_json::from_value(
        serde_json::json!({"schema_version": 2, "runtime": {
            "id": "standalone-test-runtime", "release_version": "999.999.999-test",
            "skippy_abi": skippy_runtime_install::current_skippy_abi_version(),
            "platform": {"os": profile.os, "arch": profile.arch, "min_glibc": profile.glibc_version},
            "backend": {"kind": "cpu"}, "libraries": ["lib/runtime.bin"]
        }})
    ).unwrap();
    manifest.write_to_dir(&bundle).unwrap();
    let args = NativeRuntimeOptions {
        bundle_dirs: vec![bundle.clone()],
        cache_dir: Some(temp.path().join("empty-cache")),
        release: Some("999.999.999-test".into()),
        selection: Some("cpu".into()),
    };
    let plan = local_native_runtime_plan(&args).unwrap();
    assert_eq!(
        plan.root.canonicalize().unwrap(),
        bundle.canonicalize().unwrap()
    );
    assert_eq!(plan.native_runtime_id, "standalone-test-runtime");
    assert_eq!(plan.libraries.len(), 1);
    #[cfg(feature = "dynamic-native-runtime")]
    {
        // A digest-valid non-library reaches the loader, then fails before model access.
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
            .arg("--runtime-bundle")
            .arg(&bundle)
            .env(
                "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
                "/nonexistent/mesh-runtime",
            )
            .args([
                "--output",
                "human",
                "--runtime-release",
                "999.999.999-test",
                "--runtime-cache",
            ])
            .arg(temp.path().join("empty-cache"))
            .args(["serve", "--config", "/nonexistent/skippy-stage.json"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("load Skippy runtime standalone-test-runtime"),
            "{error}"
        );
        assert!(!error.contains("load stage config"), "{error}");
    }
    manifest.runtime.skippy_abi = "0.0.0".into();
    manifest.write_to_dir(&bundle).unwrap();
    assert!(
        local_native_runtime_plan(&args)
            .unwrap_err()
            .to_string()
            .contains("no compatible local Skippy runtime")
    );
}

#[test]
fn legacy_import_is_not_a_subcommand() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["runtime", "import-legacy"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"));
}

#[test]
fn old_serve_commands_are_not_subcommands() {
    for command in ["serve-openai", "serve-binary"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
            .arg(command)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"));
    }
}

#[test]
fn jsonl_error_is_one_versioned_event() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output", "jsonl", "serve"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stderr.is_empty());
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1);
    let event: serde_json::Value = serde_json::from_slice(lines[0]).unwrap();
    assert_eq!(event["schema_version"], 1);
    assert_eq!(event["sequence"], 1);
    assert_eq!(event["type"], "error");
    assert!(
        event["data"]["message"]
            .as_str()
            .unwrap()
            .contains("provide --model")
    );
}

#[test]
fn jsonl_syntax_error_is_structured_too() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output=jsonl", "serve", "--not-a-switch"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stderr.is_empty());
    let event: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(event["type"], "error");
    assert!(
        event["data"]["message"]
            .as_str()
            .unwrap()
            .contains("unexpected argument")
    );
}

#[test]
fn recommended_models_have_human_and_json_presentations() {
    let human = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output", "human", "models", "recommended"])
        .output()
        .unwrap();
    assert!(human.status.success());
    assert!(String::from_utf8_lossy(&human.stdout).contains("⭐ Qwen3-0.6B-Q4_K_M"));
    let json = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .args(["--output", "json", "models", "recommended"])
        .output()
        .unwrap();
    assert!(json.status.success());
    let models: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert!(models.as_array().unwrap().len() >= 3);
}

#[test]
fn runtime_list_uses_skippy_cache_override_without_loading_native_code() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("SKIPPY_NATIVE_RUNTIME_CACHE_DIR", dir.path().join("empty"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            "/nonexistent/mesh-cache",
        )
        .args(["runtime", "list"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let runtimes: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert!(runtimes.is_empty());
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn models_list_uses_explicit_cache_without_native_runtime_or_network() {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("models");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("SKIPPY_MODEL_CACHE_DIR", root.path().join("ignored"))
        .args(["models", "--cache-dir"])
        .arg(&cache)
        .arg("installed")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["cache_dir"], cache.to_string_lossy().as_ref());
    assert_eq!(value["repositories"], serde_json::json!([]));
    assert!(!cache.exists());
}

#[test]
fn model_pull_rejects_invalid_pin_without_contacting_hub() {
    let root = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .args(["models", "--cache-dir"])
        .arg(root.path())
        .args(["download", "org/repo", "--sha256", "bad"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("64 hexadecimal"));
    assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
}

#[test]
fn model_remove_is_local_and_reports_repository_scope() {
    let root = tempfile::tempdir().unwrap();
    let repo = root.path().join("models--org--model/snapshots/revision");
    std::fs::create_dir_all(&repo).unwrap();
    let run = |dry: bool| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"));
        cmd.env("HF_ENDPOINT", "http://127.0.0.1:1")
            .args(["models", "--cache-dir"])
            .arg(root.path())
            .args(["remove", "org/model"]);
        if dry {
            cmd.arg("--dry-run");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
    };
    let preview = run(true);
    assert_eq!(preview["scope"], "all-local-revisions");
    assert_eq!(preview["status"], "planned");
    assert!(repo.exists());
    assert_eq!(run(false)["status"], "removed");
    assert!(!repo.exists());
}

#[test]
fn runtime_install_requires_exactly_one_explicit_catalog() {
    for args in [
        vec!["runtime", "install"],
        vec![
            "runtime",
            "install",
            "--manifest",
            "a.json",
            "--manifest-url",
            "https://example.invalid/catalog.json",
        ],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
    }
}

#[test]
fn runtime_install_does_not_fall_back_from_explicit_empty_catalog_to_mesh_policy() {
    let root = tempfile::tempdir().unwrap();
    let catalog = root.path().join("catalog.json");
    let empty = skippy_runtime_install::NativeRuntimeReleaseManifest {
        release_version: skippy_runtime_install::runtime_release_version().into(),
        skippy_abi: skippy_runtime_install::current_skippy_abi_version(),
        artifacts: Vec::new(),
    };
    std::fs::write(&catalog, serde_json::to_vec(&empty).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
            "/nonexistent/mesh-bundle",
        )
        .env(
            "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL",
            "http://127.0.0.1:1/forbidden-catalog.json",
        )
        .args(["--runtime-cache"])
        .arg(root.path().join("cache"))
        .args(["runtime", "install", "--manifest"])
        .arg(&catalog)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("no native runtime manifest entries found")
    );
    assert!(!root.path().join("cache").exists());
}
