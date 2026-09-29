use std::{net::SocketAddr, path::PathBuf};

use skippy_serving::frontend::DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(about = "Skippy model serving and runtime management")]
pub struct Cli {
    /// Output presentation for humans or automation.
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Auto)]
    pub output: OutputFormat,
    #[command(flatten)]
    pub native_runtime: NativeRuntimeArgs,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    Auto,
    Human,
    Json,
    Jsonl,
}

impl From<OutputFormat> for skippy_commands::console::OutputMode {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Auto => Self::Auto,
            OutputFormat::Human => Self::Human,
            OutputFormat::Json => Self::Json,
            OutputFormat::Jsonl => Self::Jsonl,
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Inspect hardware, caches, and the selected native runtime.
    Doctor,
    /// Prompt a running Skippy OpenAI endpoint interactively.
    Prompt(PromptArgs),
    /// Serve OpenAI and Anthropic APIs, or an explicitly selected stage transport.
    Serve(Box<ServeCommandArgs>),
    ExampleConfig,
    /// Download models or inspect the shared Hugging Face cache.
    Models {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Plan and admit a direct GGUF split for explicit worker endpoints.
    PlanSplit(PlanSplitArgs),
    /// List, install, remove, or prune verified native runtime bundles.
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StageTransport {
    Binary,
    Http,
}

#[derive(Parser)]
pub struct ServeCommandArgs {
    /// Local model path or Hugging Face repository reference.
    #[arg(long, conflicts_with_all = ["model_path", "config"])]
    pub model: Option<String>,
    /// Open an interactive prompt after the public API is ready.
    #[arg(long)]
    pub prompt: bool,
    /// Internal stage transport for a prepared stage configuration.
    #[arg(long, value_enum, requires = "config")]
    pub stage_transport: Option<StageTransport>,
    /// Run an internal stage without a public inference API.
    #[arg(long, requires = "config")]
    pub worker_only: bool,
    #[command(flatten)]
    #[command(next_help_heading = "Model and public API")]
    pub public: ServeOpenAiArgs,
    #[command(flatten)]
    #[command(next_help_heading = "Binary stage tuning")]
    pub stage: ServeBinaryArgs,
}

#[derive(Parser)]
pub struct PromptArgs {
    #[arg(long, default_value = "http://127.0.0.1:9337/v1")]
    pub endpoint: String,
    #[arg(
        long,
        help = "Model ID; defaults to the first model returned by /models"
    )]
    pub model: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: u32,
    #[arg(long, help = "Use /completions with each line as a raw prompt")]
    pub raw: bool,
    #[arg(long, help = "Disable model thinking through reasoning_effort=none")]
    pub no_think: bool,
    #[arg(long)]
    pub history_path: Option<PathBuf>,
}

#[derive(Parser)]
pub struct ServeArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
}

#[derive(clap::Args)]
pub struct ServeBinaryArgs {
    #[arg(skip)]
    pub config: PathBuf,
    #[arg(skip)]
    pub topology: Option<PathBuf>,
    #[arg(skip)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(skip)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(skip)]
    pub telemetry_queue_capacity: usize,
    #[arg(skip)]
    pub telemetry_level: TelemetryLevel,
    #[arg(skip)]
    pub worker_only: bool,
    #[arg(skip)]
    pub api_bind_addr: Option<SocketAddr>,
    #[arg(long, default_value_t = 4)]
    pub max_inflight: usize,
    #[arg(long)]
    pub reply_credit_limit: Option<usize>,
    #[arg(
        long,
        help = "Forward eligible non-final prefill activation frames on a bounded background writer. Enabled by default."
    )]
    pub async_prefill_forward: bool,
    #[arg(
        long,
        help = "Disable async forwarding for eligible non-final prefill activation frames."
    )]
    pub no_async_prefill_forward: bool,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Artificial downstream write delay in milliseconds per binary stage message."
    )]
    pub downstream_wire_delay_ms: f64,
    #[arg(
        long,
        help = "Artificial downstream activation bandwidth cap in megabits per second."
    )]
    pub downstream_wire_mbps: Option<f64>,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Mean of an exponentially distributed extra per-message downstream delay in milliseconds (models link jitter)."
    )]
    pub downstream_wire_jitter_ms: f64,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Extra burst-stall delay in milliseconds applied with --downstream-wire-stall-p probability per message."
    )]
    pub downstream_wire_stall_ms: f64,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Probability in [0, 1] that a downstream message is hit by --downstream-wire-stall-ms."
    )]
    pub downstream_wire_stall_p: f64,
    #[arg(long, default_value_t = 60)]
    pub downstream_connect_timeout_secs: u64,
    #[arg(
        long,
        help = "Also serve the OpenAI-compatible HTTP surface from this stage process. Intended for stage 0."
    )]
    pub openai_bind_addr: Option<SocketAddr>,
    #[arg(
        long,
        help = "Served OpenAI model id. Defaults to the stage config model_id."
    )]
    pub openai_model_id: Option<String>,
    #[arg(long, default_value_t = 16)]
    pub openai_default_max_tokens: u32,
    #[arg(
        long,
        help = "Maximum number of concurrent OpenAI chat generation requests hosted by this stage. Defaults to the KV-derived lane count."
    )]
    pub openai_generation_concurrency: Option<usize>,
    #[arg(
        long,
        help = "Adapt active OpenAI generation permits under sustained queued load, up to --openai-generation-concurrency. Disabled by default."
    )]
    pub openai_adaptive_generation_concurrency: bool,
    #[arg(
        long,
        help = "Initial committed generation permits when adaptive OpenAI generation concurrency is enabled. Defaults to 1; higher values require an externally validated hardware/model certificate."
    )]
    pub openai_adaptive_generation_min_concurrency: Option<usize>,
    #[arg(
        long,
        help = "Maximum number of additional OpenAI generation requests allowed to wait. Defaults to clamp(8 * resolved generation concurrency, 16, 256)."
    )]
    pub openai_generation_queue_capacity: Option<usize>,
    #[arg(
        long,
        default_value_t = DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
        help = "Maximum seconds an OpenAI generation request may wait for admission; 0 waits until cancellation."
    )]
    pub openai_generation_admission_timeout_secs: u64,
    #[arg(long, default_value_t = 256)]
    pub openai_prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "OpenAI prefill chunk policy: fixed, schedule, or adaptive-ramp. Passing --openai-prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub openai_prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated OpenAI prefill chunk schedule. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub openai_prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub openai_prefill_adaptive_max: usize,
    #[arg(
        long,
        default_value_t = 100.0,
        help = "Target maximum compute time in milliseconds for one adaptive prefill chunk at the slowest measured stage."
    )]
    pub openai_prefill_adaptive_target_ms: f64,
    #[arg(
        long,
        help = "Draft GGUF to use for speculative decoding in the embedded stage-0 OpenAI surface."
    )]
    pub openai_draft_model_path: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub openai_speculative_window: usize,
    #[arg(long)]
    pub openai_adaptive_speculative_window: bool,
    #[arg(
        long,
        help = "Override n_gpu_layers for the embedded OpenAI draft model. Defaults to the stage config n_gpu_layers."
    )]
    pub openai_draft_n_gpu_layers: Option<i32>,
    #[arg(
        long,
        help = "Native MTP sidecar GGUF to attach to the stage-0 model. Unlike --openai-draft-model-path this is not opened as a standalone draft model; its MTP heads are attached to the served model."
    )]
    pub openai_native_mtp_draft_model_path: Option<PathBuf>,
    #[arg(
        long,
        help = "JSON file containing the complete resolved speculative decode plan."
    )]
    pub openai_speculative_config: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct ServeOpenAiArgs {
    /// Prepared stage configuration; mutually exclusive with --model-path and --model.
    #[arg(long, conflicts_with = "model_path")]
    pub config: Option<PathBuf>,
    /// Local GGUF (first shard) or safetensors checkpoint to prepare and serve.
    #[arg(long, conflicts_with = "config")]
    pub model_path: Option<PathBuf>,
    /// Context size for a local model. Defaults to 4096.
    #[arg(long, conflicts_with = "config")]
    pub ctx_size: Option<u32>,
    /// GPU layers for a local model; -1 offloads all supported layers.
    #[arg(long, conflicts_with = "config", allow_hyphen_values = true)]
    pub n_gpu_layers: Option<i32>,
    /// Optional advisory digest cache directory for local checkpoint files.
    #[arg(long, conflicts_with = "config")]
    pub hash_cache: Option<PathBuf>,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub bind_addr: SocketAddr,
    #[arg(
        long,
        help = "Served model id to advertise and accept, for example org/repo:Q4_K_M. Defaults to config model_id."
    )]
    pub model_id: Option<String>,
    #[arg(
        long,
        help = "JSON file containing a complete resolved speculative decode plan."
    )]
    pub speculative_config: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pub default_max_tokens: u32,
    #[arg(
        long,
        help = "Maximum number of concurrent chat generation requests. Defaults to the KV-derived lane count."
    )]
    pub generation_concurrency: Option<usize>,
    #[arg(
        long,
        help = "Adapt active generation permits under sustained queued load, up to --generation-concurrency. Disabled by default."
    )]
    pub adaptive_generation_concurrency: bool,
    #[arg(
        long,
        help = "Initial committed generation permits when adaptive generation concurrency is enabled. Defaults to 1; higher values require an externally validated hardware/model certificate."
    )]
    pub adaptive_generation_min_concurrency: Option<usize>,
    #[arg(
        long,
        help = "Maximum number of additional generation requests allowed to wait. Defaults to clamp(8 * resolved generation concurrency, 16, 256)."
    )]
    pub generation_queue_capacity: Option<usize>,
    #[arg(
        long,
        default_value_t = DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
        help = "Maximum seconds a generation request may wait for admission; 0 waits until cancellation."
    )]
    pub generation_admission_timeout_secs: u64,
    #[arg(long, default_value_t = 256)]
    pub prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "Prefill chunk policy for split OpenAI serving: fixed, schedule, or adaptive-ramp. Passing --prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated prefill chunk schedule for split OpenAI serving. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub prefill_adaptive_max: usize,
    #[arg(long, default_value_t = 100.0)]
    pub prefill_adaptive_target_ms: f64,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
    #[arg(
        long = "openai-guardrails",
        value_enum,
        default_value_t = OpenAiGuardrailsCliMode::Metrics,
        help = "OpenAI compatibility guardrail mode for standalone serving: disabled, metrics, or enforce."
    )]
    pub openai_guardrails: OpenAiGuardrailsCliMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OpenAiGuardrailsCliMode {
    Disabled,
    Metrics,
    Enforce,
}

#[derive(Clone, Debug, Default, clap::Args)]
pub struct NativeRuntimeArgs {
    /// Directory containing a verified native runtime bundle (repeatable).
    #[arg(long = "runtime-bundle", global = true)]
    pub bundle_dirs: Vec<PathBuf>,
    /// Native runtime cache root; model caches are separate.
    #[arg(long = "runtime-cache", global = true)]
    pub cache_dir: Option<PathBuf>,
    /// Required Skippy runtime release. Defaults to this build's runtime metadata.
    #[arg(long = "runtime-release", global = true)]
    pub release: Option<String>,
    /// Runtime backend or exact artifact ID.
    #[arg(long = "runtime-selection", global = true)]
    pub selection: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum TelemetryLevel {
    Off,
    #[default]
    Summary,
    Debug,
}

#[derive(Subcommand)]
pub enum RuntimeCommand {
    /// List locally discoverable or available release runtimes.
    List {
        #[arg(long, conflicts_with = "installed")]
        available: bool,
        #[arg(long, conflicts_with = "available")]
        installed: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Install the recommended runtime or an explicit flavor/runtime ID.
    Install {
        runtime: Option<String>,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Remove an installed native runtime.
    Remove {
        native_runtime_id: String,
        #[arg(long)]
        release: Option<String>,
    },
    /// Prune old native runtimes from the cache.
    Prune {
        #[arg(long)]
        active_only: bool,
        #[arg(long)]
        release: Option<String>,
    },
}

#[derive(Parser)]
pub struct PlanSplitArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long)]
    pub model_id: Option<String>,
    /// Ordered worker listen endpoints, one per stage. Use routable addresses across machines.
    #[arg(long = "worker", required = true)]
    pub workers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 512)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 1)]
    pub lanes: u32,
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    /// New directory for stage configs and their admission descriptors; never overwritten.
    #[arg(long)]
    pub output_dir: PathBuf,
}

impl From<ModelCommand> for skippy_commands::models::ModelAction {
    fn from(command: ModelCommand) -> Self {
        match command {
            ModelCommand::Download { model_ref } => Self::Download { model_ref },
            ModelCommand::Delete { model, yes } => Self::Delete { model, yes },
            ModelCommand::Installed => Self::Installed,
            ModelCommand::Recommended => Self::Recommended,
            ModelCommand::Search { query, limit } => Self::Search { query, limit },
            ModelCommand::Show { model_ref } => Self::Show { model_ref },
        }
    }
}

impl From<RuntimeCommand> for skippy_commands::runtime::RuntimeAction {
    fn from(command: RuntimeCommand) -> Self {
        match command {
            RuntimeCommand::List {
                available,
                manifest,
                ..
            } => Self::List {
                available,
                manifest,
            },
            RuntimeCommand::Install { runtime, manifest } => Self::Install { runtime, manifest },
            RuntimeCommand::Remove {
                native_runtime_id,
                release,
            } => Self::Remove {
                native_runtime_id,
                release,
            },
            RuntimeCommand::Prune {
                active_only,
                release,
            } => Self::Prune {
                active_only,
                release,
            },
        }
    }
}

impl From<PlanSplitArgs> for skippy_commands::split::PlanSplitCommand {
    fn from(args: PlanSplitArgs) -> Self {
        Self {
            model_path: args.model_path,
            model_id: args.model_id,
            workers: args.workers,
            ctx_size: args.ctx_size,
            lanes: args.lanes,
            n_gpu_layers: args.n_gpu_layers,
            output_dir: args.output_dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_accepts_a_running_endpoint_without_model_files() {
        let cli = Cli::try_parse_from([
            "skippy",
            "prompt",
            "--endpoint",
            "http://127.0.0.1:9337/v1",
            "--no-think",
        ])
        .unwrap();
        let Command::Prompt(args) = cli.command else {
            panic!("expected prompt command");
        };
        assert_eq!(args.endpoint, "http://127.0.0.1:9337/v1");
        assert!(args.no_think);
        assert!(args.model.is_none());
    }

    #[test]
    fn model_reference_accepts_local_serving_tuning() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--model",
            "Qwen3-0.6B-Q4_K_M",
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "-1",
            "--prompt",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(args.model.as_deref(), Some("Qwen3-0.6B-Q4_K_M"));
        assert_eq!(args.public.ctx_size, Some(8192));
        assert_eq!(args.public.n_gpu_layers, Some(-1));
        assert!(args.prompt);
    }

    #[test]
    fn openai_prefill_policy_defaults_to_adaptive_ramp() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage.json",
            "--stage-transport",
            "binary",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let args = args.stage;
        assert_eq!(args.openai_prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.openai_prefill_adaptive_start, 128);
        assert_eq!(args.openai_prefill_adaptive_step, 128);
        assert_eq!(args.openai_prefill_adaptive_max, 384);
        assert_eq!(args.openai_prefill_adaptive_target_ms, 100.0);
        assert_eq!(args.openai_generation_concurrency, None);
        assert!(!args.openai_adaptive_generation_concurrency);
        assert_eq!(args.openai_adaptive_generation_min_concurrency, None);
        assert_eq!(args.openai_generation_queue_capacity, None);
        assert_eq!(args.openai_generation_admission_timeout_secs, 0);

        let cli = Cli::try_parse_from(["skippy", "serve", "--config", "stage.json"]).unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let args = args.public;
        assert_eq!(args.prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.prefill_adaptive_start, 128);
        assert_eq!(args.prefill_adaptive_step, 128);
        assert_eq!(args.prefill_adaptive_max, 384);
        assert_eq!(args.prefill_adaptive_target_ms, 100.0);
        assert_eq!(args.generation_concurrency, None);
        assert!(!args.adaptive_generation_concurrency);
        assert_eq!(args.adaptive_generation_min_concurrency, None);
        assert_eq!(args.generation_queue_capacity, None);
        assert_eq!(args.generation_admission_timeout_secs, 0);
        assert_eq!(args.openai_guardrails, OpenAiGuardrailsCliMode::Metrics);
    }

    #[test]
    fn serve_openai_accepts_explicit_guardrail_mode() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage.json",
            "--openai-guardrails",
            "enforce",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(
            args.public.openai_guardrails,
            OpenAiGuardrailsCliMode::Enforce
        );
    }

    #[test]
    fn standalone_commands_accept_resolved_speculative_config_files() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage.json",
            "--stage-transport",
            "binary",
            "--openai-speculative-config",
            "decode-plan.json",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(
            args.stage.openai_speculative_config,
            Some(PathBuf::from("decode-plan.json"))
        );

        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage.json",
            "--speculative-config",
            "decode-plan.json",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(
            args.public.speculative_config,
            Some(PathBuf::from("decode-plan.json"))
        );
    }
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// Resolve a Hub revision and download its selected model files.
    Download {
        /// Hub reference: org/repo@revision:filename-or-quantization.
        model_ref: String,
    },
    /// Preview or delete one installed model, never a remote Hub repository.
    Delete {
        model: String,
        /// Skip the dry-run preview and delete the selected local files.
        #[arg(long)]
        yes: bool,
    },
    /// List local model repositories and snapshots without contacting the Hub.
    Installed,
    /// Show recommended models from the shared remote catalog.
    Recommended,
    /// Find GGUF repositories on Hugging Face.
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Resolve one model reference and show its selected artifact.
    Show { model_ref: String },
}
