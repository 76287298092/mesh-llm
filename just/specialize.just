# Internal experiment only. Does not build or publish a MeshLLM product.
specialize-tools-build:
    just with-lld cargo build --release -p xtask

specialize-baseline plan output:
    ./target/release/xtask specialize baseline --plan "{{plan}}" --output "{{output}}"

# Compile Rust device source to textual PTX. The driver performs final assembly.
specialize-ptx:
    mkdir -p target/specialize
    RUSTC="$(rustup which --toolchain nightly-2026-09-25 rustc)" RUSTC_WRAPPER= RUSTFLAGS='-Ctarget-cpu=sm_120a' rustup run nightly-2026-09-25 cargo build --manifest-path crates/mesh-specialize/kernels/nvptx/Cargo.toml --target nvptx64-nvidia-cuda -Zbuild-std=core --release --target-dir target/specialize/nvptx
    cp target/specialize/nvptx/nvptx64-nvidia-cuda/release/mesh_specialize_nvptx.ptx target/specialize/probes.ptx
