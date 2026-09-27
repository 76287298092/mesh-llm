# Internal experiment only. Does not build or publish a MeshLLM product.
specialize-tools-build:
    just with-lld cargo build --release -p xtask

# cuBLAS exists only in this separate validation executable, never the runtime.
specialize-validation-build:
    just with-lld cargo build --release -p mesh-specialize --features validation --bin mesh-specialize-validate

specialize-baseline plan output:
    ./target/release/xtask specialize baseline --plan "{{plan}}" --output "{{output}}"

# Compile Rust device source to textual PTX. The driver performs final assembly.
specialize-ptx:
    mkdir -p target/specialize
    RUSTC="$(rustup which --toolchain nightly-2026-09-25 rustc)" RUSTC_WRAPPER= RUSTFLAGS='-Ctarget-cpu=sm_120a' rustup run nightly-2026-09-25 cargo build --manifest-path crates/mesh-specialize/kernels/nvptx/Cargo.toml --target nvptx64-nvidia-cuda -Zbuild-std=core --release --target-dir target/specialize/nvptx
    cp target/specialize/nvptx/nvptx64-nvidia-cuda/release/mesh_specialize_nvptx.ptx target/specialize/probes.ptx

# Offline inspection of Rust-emitted PTX; no CUDA C/C++ compilation.
specialize-sass ptx="target/specialize/probes.ptx":
    ptxas --gpu-name sm_120a --maxrregcount 64 --verbose "{{ptx}}" --output-file target/specialize/probes.cubin
    nvdisasm target/specialize/probes.cubin > target/specialize/probes.sass
