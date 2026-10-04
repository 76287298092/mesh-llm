use std::fs;
use std::path::Path;

fn main() {
    // Keep Cargo's default package change detection. Watching `dist`
    // explicitly makes a missing directory look changed on every invocation,
    // so test loops rebuild this crate and every dependent crate indefinitely.
    // The default still notices when `dist` appears or its contents change.
    //
    // That default is also why this script can hand out a stale answer. Cargo
    // re-runs a build script when a file it already knows about changes; a
    // `dist/` that appears later is a new directory, not a change to a known
    // file, so the script keeps emitting the empty fallback it chose while
    // `dist` was absent. The host then embeds an empty console: it compiles, it
    // starts, it logs "web console ready", and every console route answers 500
    // because index.html is missing from the embedded tree.
    //
    // `cargo clean -p mesh-llm-ui` clears that state. The checks below turn the
    // silent half of it -- building with `embed-assets` on and nothing to embed
    // -- into a build error instead of a runtime 500.
    configure_console_dist();
}

fn configure_console_dist() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir");
    let console_dist = Path::new(&manifest_dir).join("dist");
    let embed_assets = std::env::var_os("CARGO_FEATURE_EMBED_ASSETS").is_some();

    if console_dist.is_dir() {
        if embed_assets && !console_dist.join("index.html").is_file() {
            panic!(
                "embed-assets is enabled but {} contains no index.html; build the console \
                 first (`pnpm install --frozen-lockfile && pnpm run build` in crates/mesh-llm-ui), \
                 or depend on mesh-llm-ui with default-features = false",
                console_dist.display()
            );
        }
        println!(
            "cargo:rustc-env=MESH_LLM_UI_DIST={}",
            console_dist.display()
        );
        return;
    }

    if embed_assets {
        panic!(
            "embed-assets is enabled but {} does not exist. A binary built from this state \
             would embed an empty console and answer 500 on every console route, so this is \
             a build error instead. Build the console first \
             (`pnpm install --frozen-lockfile && pnpm run build` in crates/mesh-llm-ui), or \
             depend on mesh-llm-ui with default-features = false.",
            console_dist.display()
        );
    }

    let fallback =
        Path::new(&std::env::var("OUT_DIR").expect("cargo out dir")).join("empty-ui-dist");
    fs::create_dir_all(&fallback).expect("create fallback UI dist dir");
    println!("cargo:rustc-env=MESH_LLM_UI_DIST={}", fallback.display());
}
