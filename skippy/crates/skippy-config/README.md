# skippy-config

Standalone Skippy settings, shared by the CLI and serving. `validate_config` checks a `StageConfig` (and optionally its `StageTopology`) against the same rules standalone serving enforces, including layer-package and artifact-slice load-mode requirements. `load_json` reads typed configuration documents and `example_config` emits the canonical single-stage example.

Model downloads use the shared Hugging Face cache policy in [`skippy-model-hf`](../skippy-model-hf/README.md). Native-runtime caches and bundle discovery use [`skippy-runtime-install`](../skippy-runtime-install/README.md). There are no Skippy-only cache overrides.

The crate sits below serving and the lifecycle API: it depends only on protocol primitives and path policy, never on `skippy-api` or `skippy-serving`, and reads no configuration beyond the documented environment variables.
