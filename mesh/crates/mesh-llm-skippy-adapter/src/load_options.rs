//! Mesh configuration translated into Skippy preparation inputs.
use crate::{SkippyPackageIdentity, config};
use skippy_protocol::{FlashAttentionType, StageKvCacheConfig};
use skippy_serving::serving_hooks::SharedModelServingHooksFactory;
use skippy_serving::{DEFAULT_EMBEDDED_MAX_TOKENS, OpenAiGuardrailsConfig};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippyDeviceDescriptor {
    pub backend_device: String,
    pub stable_id: Option<String>,
    pub index: Option<usize>,
    pub vram_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SkippyModelLoadOptions {
    pub model_id: String,
    pub model_path: PathBuf,
    pub ctx_size: u32,
    pub n_gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub repack: bool,
    pub op_offload: Option<bool>,
    pub no_host_buffer: bool,
    pub check_tensors: bool,
    pub checkpoint_quantization: Option<String>,
    pub checkpoint_imatrix: Option<String>,
    pub direct_io: bool,
    pub main_gpu: Option<u32>,
    pub split_mode: skippy_protocol::SplitMode,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub flash_attn_type: FlashAttentionType,
    pub kv_offload: Option<bool>,
    pub kv_unified: Option<bool>,
    pub swa_full: Option<bool>,
    pub cache_idle_slots: Option<u32>,
    pub generation_concurrency: usize,
    pub default_max_tokens: u32,
    pub kv_cache: Option<StageKvCacheConfig>,
    pub embedded_openai: Option<config::ResolvedEmbeddedOpenAiArgs>,
    pub layer_start: u32,
    pub layer_end: Option<u32>,
    pub selected_device: Option<SkippyDeviceDescriptor>,
    pub package_identity: Option<SkippyPackageIdentity>,
    pub projector_path: Option<PathBuf>,
    pub projector_use_gpu: Option<bool>,
    pub media_marker: Option<String>,
    pub image_min_tokens: Option<u32>,
    pub image_max_tokens: Option<u32>,
    pub batch_max_tokens: Option<u32>,
    pub glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub generation_signal_window: Option<u32>,
    pub telemetry: SkippyTelemetryOptions,
    pub openai_guardrails: Option<OpenAiGuardrailsConfig>,
    pub native_mtp_enabled: bool,
    pub serving_hooks_factory: Option<SharedModelServingHooksFactory>,
}

pub use skippy_api::serving::ServingTelemetryOptions as SkippyTelemetryOptions;

impl SkippyModelLoadOptions {
    pub fn for_direct_gguf(model_id: impl Into<String>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_id: model_id.into(),
            model_path: model_path.into(),
            ctx_size: 4096,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            generation_concurrency: 1,
            default_max_tokens: DEFAULT_EMBEDDED_MAX_TOKENS,
            kv_cache: None,
            embedded_openai: None,
            layer_start: 0,
            layer_end: None,
            selected_device: None,
            package_identity: None,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            telemetry: SkippyTelemetryOptions::off(),
            openai_guardrails: Some(OpenAiGuardrailsConfig::disabled_for_skippy()),
            native_mtp_enabled: true,
            serving_hooks_factory: None,
        }
    }

    pub fn with_ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub fn with_generation_concurrency(mut self, generation_concurrency: usize) -> Self {
        self.generation_concurrency = generation_concurrency;
        self
    }

    pub fn with_cache_types(mut self, cache_type_k: &str, cache_type_v: &str) -> Self {
        self.cache_type_k = cache_type_k.to_string();
        self.cache_type_v = cache_type_v.to_string();
        self
    }

    pub fn with_batch_sizes(mut self, n_batch: Option<u32>, n_ubatch: Option<u32>) -> Self {
        self.n_batch = n_batch;
        self.n_ubatch = n_ubatch;
        self
    }

    pub fn with_thread_counts(
        mut self,
        n_threads: Option<usize>,
        n_threads_batch: Option<usize>,
    ) -> Self {
        self.n_threads = n_threads;
        self.n_threads_batch = n_threads_batch;
        self
    }

    pub fn with_flash_attn_type(mut self, flash_attn_type: FlashAttentionType) -> Self {
        self.flash_attn_type = flash_attn_type;
        self
    }

    pub fn with_kv_session_controls(
        mut self,
        kv_offload: Option<bool>,
        kv_unified: Option<bool>,
        swa_full: Option<bool>,
    ) -> Self {
        self.kv_offload = kv_offload;
        self.kv_unified = kv_unified;
        self.swa_full = swa_full;
        self
    }

    pub fn with_cache_idle_slots(mut self, cache_idle_slots: Option<u32>) -> Self {
        self.cache_idle_slots = cache_idle_slots;
        self
    }

    pub fn with_layer_end(mut self, layer_end: u32) -> Self {
        self.layer_end = Some(layer_end);
        self
    }

    pub fn with_layer_range(mut self, layer_start: u32, layer_end: u32) -> Self {
        self.layer_start = layer_start;
        self.layer_end = Some(layer_end);
        self
    }

    pub fn with_selected_device(mut self, selected_device: SkippyDeviceDescriptor) -> Self {
        self.selected_device = Some(selected_device);
        self
    }

    pub fn with_projector_path(mut self, projector_path: impl Into<PathBuf>) -> Self {
        self.projector_path = Some(projector_path.into());
        self
    }

    pub fn with_telemetry(mut self, telemetry: SkippyTelemetryOptions) -> Self {
        self.telemetry = telemetry;
        self
    }

    pub fn with_kv_cache(mut self, kv_cache: Option<StageKvCacheConfig>) -> Self {
        self.kv_cache = kv_cache;
        self
    }

    pub fn with_embedded_openai(
        mut self,
        embedded_openai: config::ResolvedEmbeddedOpenAiArgs,
    ) -> Self {
        self.embedded_openai = Some(embedded_openai);
        self
    }

    pub fn with_openai_guardrails(mut self, openai_guardrails: OpenAiGuardrailsConfig) -> Self {
        self.openai_guardrails = Some(openai_guardrails);
        self
    }

    pub fn with_serving_hooks_factory(
        mut self,
        factory: Option<SharedModelServingHooksFactory>,
    ) -> Self {
        self.serving_hooks_factory = factory;
        self
    }

    pub fn with_package_identity(mut self, package_identity: SkippyPackageIdentity) -> Self {
        self.package_identity = Some(package_identity);
        self
    }
}
