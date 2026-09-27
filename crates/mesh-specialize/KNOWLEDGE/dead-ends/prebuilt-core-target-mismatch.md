# Prebuilt NVPTX core rejects architecture-specific device compilation

Status: reproduced compiler failure, superseded by rebuilding core for the target.
Model/GPU execution: not applicable, no kernel executed. Target SM120a.
Host toolchain: nightly-2026-09-25, rustc 1.100.0-nightly f7575a9da, macOS arm64.
Source base: 2d9ea2073 plus initial instruction-probe working changes.

An initial `just specialize-ptx` used standalone rustc, the prebuilt NVPTX core,
`-Ctarget-cpu=sm_120a`, and `--emit=asm`. Expected textual PTX; compiler rejected
ABI mismatches because core and compiler_builtins were built without that target
CPU. Explicit `+ptx87` also emitted an unstable target-feature warning.

Do not suppress this with `unsafe-allow-abi-mismatch`. The corrected recipe uses
an isolated Rust device workspace, pinned nightly, `-Zbuild-std=core`, and
`-Ctarget-cpu=sm_120a` for the complete device compilation. The architecture sets
its minimum PTX version; inspect the emitted version rather than adding an
unnecessary unstable feature flag. GPU execution remains a separate gate.

The first Cargo retry selected Homebrew's stable `rustc` from PATH even though
Cargo itself ran through `rustup run nightly-2026-09-25`. It failed on `-Z` after
finding the stable sysroot. The recipe now sets `RUSTC` to `rustup which` for the
pinned nightly explicitly; it does not change the user's global default compiler.

Durable rule: compile device core with the same architecture/ABI flags as kernels.
