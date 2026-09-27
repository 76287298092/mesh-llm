//! Device-specific instruction qualification, separate from model execution.

pub mod attention_profile;
pub mod fp8_profile;

pub enum DecoderBlockKind {
    Gdn,
    Attention,
}
pub enum DecoderMlpKind {
    Nvfp4,
    Fp8,
}
pub struct DecoderLayer {
    pub prefix: String,
    pub state_prefix: String,
    pub block: DecoderBlockKind,
    pub mlp: DecoderMlpKind,
}
pub struct DecoderConfig {
    pub layers: Vec<DecoderLayer>,
    pub gdn_shape: GdnShape,
    pub attention_shape: ResidentAttentionShape,
    pub embedding_table: String,
    pub first_norm: String,
    pub final_norm: String,
    pub head_prefix: String,
    pub hidden: usize,
    pub vocabulary: usize,
    pub capacity: usize,
    pub state_layout: crate::engine::layout::Layout,
}
pub fn model_check(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &DecoderConfig,
    reference: &crate::packages::qwen3_8_27b::model_reference::ModelReference,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_model_trial::run(ptx, device, artifact, objects, config, reference);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, config, reference);
        anyhow::bail!("Model trial requires Linux")
    }
}

pub fn model_profile(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &DecoderConfig,
    tokens: &[u32],
    teacher_token: Option<u32>,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_model_profile::run(
        ptx,
        device,
        artifact,
        objects,
        config,
        tokens,
        teacher_token,
    );
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            ptx,
            device,
            artifact,
            objects,
            config,
            tokens,
            teacher_token,
        );
        anyhow::bail!("Model profiling requires Linux")
    }
}

pub struct SpeculationRequest<'a> {
    pub tokens: &'a [u32],
    pub output_tokens: usize,
    pub depth: usize,
    pub repetitions: usize,
}
pub fn mtp_trial(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &DecoderConfig,
    reference: &crate::packages::qwen3_8_27b::mtp::Reference,
    request: &SpeculationRequest<'_>,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_mtp_trial::run(
        ptx, device, artifact, objects, config, reference, request,
    );
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, config, reference, request);
        anyhow::bail!("MTP trial requires Linux")
    }
}

pub struct ModelBenchRequest<'a> {
    pub tokens: &'a [u32],
    pub output_tokens: usize,
    pub repetitions: usize,
}
pub fn model_benchmark(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &DecoderConfig,
    request: &ModelBenchRequest<'_>,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_model_bench::run(ptx, device, artifact, objects, config, request);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, config, request);
        anyhow::bail!("Model benchmark requires Linux")
    }
}

pub struct ResidentAttentionShape {
    pub hidden: usize,
    pub intermediate: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_width: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
}
pub struct ResidentAttentionConfig {
    pub shape: ResidentAttentionShape,
    pub prefix: String,
    pub state_prefix: String,
    pub table_name: String,
    pub vocabulary: usize,
    pub capacity: usize,
    pub state_layout: crate::engine::layout::Layout,
}
pub struct ResidentAttentionCase {
    pub tokens: Vec<u32>,
    pub reference: crate::qwen_attention_layer_reference::Layer,
}
pub fn resident_attention_check(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &ResidentAttentionConfig,
    cases: &[ResidentAttentionCase],
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_attention_trial::run(ptx, device, artifact, objects, config, cases);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, config, cases);
        anyhow::bail!("Resident attention trial requires Linux")
    }
}

pub struct GdnShape {
    pub hidden: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_width: usize,
    pub intermediate: usize,
}
pub struct ResidentGdnConfig {
    pub shape: GdnShape,
    pub prefix: String,
    pub state_prefix: String,
    pub table_name: String,
    pub vocabulary: usize,
    pub state_layout: crate::engine::layout::Layout,
}
pub struct ResidentGdnCase {
    pub tokens: Vec<u32>,
    pub reference: crate::qwen_gdn_layer_reference::Layer,
}
pub fn resident_gdn_check(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    config: &ResidentGdnConfig,
    cases: &[ResidentGdnCase],
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::resident_gdn_trial::run(ptx, device, artifact, objects, config, cases);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, config, cases);
        anyhow::bail!("Resident GDN trial requires Linux")
    }
}

pub struct Fp8MlpCase {
    pub prefix: String,
    pub rows: usize,
    pub width: usize,
    pub channels: usize,
    pub input: Vec<u16>,
    pub reference: crate::fp8_mlp_reference::Output,
}

pub fn fp8_mlp_check(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    cases: &[Fp8MlpCase],
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::fp8_mlp_trial::run(ptx, device, artifact, objects, cases);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, cases);
        anyhow::bail!("FP8 MLP GPU trial requires Linux")
    }
}

/// First operation used to validate views into a complete resident weight arena.
pub struct ResidentEntryInput {
    pub table_name: String,
    pub norm_name: String,
    pub tokens: Vec<u32>,
    pub width: usize,
    pub epsilon: f32,
    pub reference: crate::entry_reference::EntryReference,
}

pub fn residency_check(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    state_layout: &crate::engine::layout::Layout,
    entry: &ResidentEntryInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::residency::run(ptx, device, artifact, objects, state_layout, entry);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, state_layout, entry);
        anyhow::bail!("Qwen residency GPU trial requires Linux")
    }
}

/// Validated by the scalar reference before any GPU allocation or launch.
pub struct EmbeddingNormInput {
    pub table: Vec<u8>,
    pub weight: Vec<u8>,
    pub width: usize,
    pub epsilon: f32,
    pub batches: Vec<Vec<u32>>,
}

pub struct Fp8Projection {
    pub name: String,
    pub weights: Vec<u8>,
    pub scales: Vec<u8>,
    pub channels: usize,
}

pub struct AttentionInput {
    pub output_projection: Fp8Projection,
    pub post_attention_norm: ResidualNormWeights,
    pub mlp: Nvfp4Mlp,
    pub entry: EmbeddingNormInput,
    pub projections: [Fp8Projection; 3],
    pub q_norm: Vec<u8>,
    pub k_norm: Vec<u8>,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_width: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub positions: Vec<Vec<u32>>,
}

pub fn attention_check(
    ptx: &str,
    device: i32,
    input: &AttentionInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::attention::run(ptx, device, input);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, input);
        anyhow::bail!("Qwen attention GPU trial requires Linux")
    }
}

pub struct ProjectionInput {
    pub entry: EmbeddingNormInput,
    pub projections: Vec<Fp8Projection>,
    pub bf16_projections: Vec<Bf16Projection>,
    pub convolution: Option<CausalConv4Weights>,
    pub gdn: Option<GdnWeights>,
    pub gdn_output: Option<GdnOutputWeights>,
    pub post_attention_norm: Option<ResidualNormWeights>,
    pub mlp: Option<Nvfp4Mlp>,
}

/// Logical packed weights; local FP8 scales divide by each global multiplier.
pub struct Nvfp4Projection {
    pub name: String,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub input_global: f32,
    pub weight_global: f32,
    pub channels: usize,
}

pub struct Nvfp4Mlp {
    pub gate: Nvfp4Projection,
    pub up: Nvfp4Projection,
    pub down: Nvfp4Projection,
}

pub struct ResidualNormWeights {
    pub weight: Vec<u8>,
    pub epsilon: f32,
}

pub struct GdnOutputWeights {
    pub z_projection: usize,
    pub norm: Vec<u8>,
    pub epsilon: f32,
    pub projection: Fp8Projection,
}

pub struct GdnWeights {
    pub a_projection: usize,
    pub b_projection: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
    pub a_log: Vec<u8>,
    pub dt_bias: Vec<u8>,
}

/// Fixed-width causal convolution attached to one FP8 projection output.
pub struct CausalConv4Weights {
    pub projection: usize,
    pub weights: Vec<u8>,
}

pub struct Bf16Projection {
    pub name: String,
    pub weights: Vec<u8>,
    pub channels: usize,
}

pub fn projection_check(
    ptx: &str,
    device: i32,
    input: &ProjectionInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::projections::run(ptx, device, input);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, input);
        anyhow::bail!("Qwen projection GPU trial requires Linux")
    }
}

pub fn embedding_norm_check(
    ptx: &str,
    device: i32,
    input: &EmbeddingNormInput,
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::embedding_norm::run(ptx, device, input);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, input);
        anyhow::bail!("Qwen entry GPU trial requires Linux")
    }
}

#[cfg(target_os = "linux")]
mod cuda;
#[cfg(any(target_os = "linux", test))]
mod fixtures;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(
    feature = "validation",
    allow(
        dead_code,
        reason = "Dense fixture materialization is used by the separate validation binary"
    )
)]
mod gemm_fixtures;
#[cfg(any(target_os = "linux", test))]
mod memory_fixtures;
#[cfg(any(target_os = "linux", test))]
mod nvfp4_layout;
#[cfg(any(target_os = "linux", test))]
mod ordinary_fixtures;
#[cfg(any(target_os = "linux", test))]
mod rms_norm_fixtures;

/// Snapshot the exact selected CUDA device and its current free memory. This
/// creates and destroys a context, but loads no model or device kernel.
pub fn device_probe(device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::device_probe(device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        anyhow::bail!("CUDA device admission trials require Linux")
    }
}

/// Run representative RMSNorm and tiled GEMM fixtures with resident GPU timing.
/// These are kernel workloads, not model prefill/decode measurements.
pub fn workload_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::workloads::run(ptx, device, true);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA workload trials require Linux")
    }
}

/// Check each representative workload once, for bounded sanitizer execution.
pub fn workload_check(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::workloads::run(ptx, device, false);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA workload trials require Linux")
    }
}

/// Qualify shared-memory copies/loads, ordinary MMA, and register budgeting.
pub fn instruction_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::instructions::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA instruction trials require Linux")
    }
}

/// JIT and numerically check the Rust NVFP4 single-warp probe on SM120.
/// Results are instruction evidence, not model throughput.
pub fn nvfp4_probe(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("CUDA instruction trials require Linux")
    }
}
#[path = "../kernels/nvptx/silu.rs"]
pub mod silu;

#[path = "../kernels/nvptx/exponential.rs"]
pub mod exponential;

/// Check experimental projection kernels against independent synthetic references.
pub fn feature_projection_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::feature_projection_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Feature projection qualification requires Linux")
    }
}

/// Qualify experimental online attention on independent synthetic fixtures.
pub fn feature_attention_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::feature_attention_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Attention qualification requires Linux")
    }
}

/// Check fixed-address CUDA graph capture and replay.
pub fn feature_graph_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::feature_graph_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Graph qualification requires Linux")
    }
}

/// Check compact recurrence records against independent state prefixes.
pub fn feature_gdn_replay_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::feature_gdn_replay_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("GDN replay qualification requires Linux")
    }
}

/// Synthetic exact projection evidence; does not qualify model performance.
pub fn fp8_exact_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::fp8_exact_trial::standalone(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Exact FP8 qualification requires Linux")
    }
}

/// Check fused exact gate/up against independent and separate GPU controls.
pub fn feature_fusion_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::feature_fusion_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Fusion qualification requires Linux")
    }
}

/// Model-owned shape/metadata for the bounded workspace experiment.
pub struct MlpWorkspaceCase {
    pub prefix: String,
    pub width: usize,
    pub channels: usize,
    pub fp8: bool,
}
/// Compare identical resident MLP kernels under allocation/completion schedules.
pub fn mlp_workspace_trial(
    ptx: &str,
    device: i32,
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    objects: &[crate::artifact::schema::Object],
    cases: &[MlpWorkspaceCase],
) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::mlp_workspace_trial::run(ptx, device, artifact, objects, cases);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device, artifact, objects, cases);
        anyhow::bail!("MLP workspace trial requires Linux")
    }
}

/// Check GPU greedy selection against independent CPU ordering and rejection.
pub fn greedy_trial(ptx: &str, device: i32) -> anyhow::Result<serde_json::Value> {
    #[cfg(target_os = "linux")]
    return cuda::greedy_trial::run(ptx, device);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("GPU greedy qualification requires Linux")
    }
}
