//! Verified resident weight addresses and scalars, bound once at construction.
//!
//! FP8 and NVFP4 projections reuse the legacy projection constructors (and their
//! metadata checks and global-scale arithmetic) through `workspace_binding`.
//! Other tensors are bound with the same names, dtypes, shapes and extents the
//! legacy host operations check.

use super::super::{
    mlp_workspace_projection::{Arithmetic as BindingArithmetic, Binding},
    resident_embedding, resident_fp8, resident_gdn_core, resident_nvfp4,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::schema::DType,
    kernels::{DecoderBlockKind, DecoderConfig, DecoderMlpKind},
};
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub(super) enum Arithmetic {
    Fp8,
    Nvfp4 { input_scale: f32, factor: f32 },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProjectionWeights {
    pub(super) weight: u64,
    pub(super) scale: u64,
    pub(super) width: usize,
    pub(super) channels: usize,
    pub(super) arithmetic: Arithmetic,
}

impl ProjectionWeights {
    fn from_binding(binding: &Binding<'_, '_>) -> Self {
        Self {
            weight: binding.weights[0],
            scale: binding.weights[1],
            width: binding.width,
            channels: binding.channels,
            arithmetic: match binding.arithmetic {
                BindingArithmetic::Fp8 => Arithmetic::Fp8,
                BindingArithmetic::Nvfp4 {
                    input_scale,
                    factor,
                } => Arithmetic::Nvfp4 {
                    input_scale,
                    factor,
                },
            },
        }
    }

    fn fp8(
        owner: &ResidentWeights<'_>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let projection = resident_fp8::Projection::new(owner, prefix, width, channels)?;
        Ok(Self::from_binding(&projection.workspace_binding()?))
    }

    fn nvfp4(
        owner: &ResidentWeights<'_>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let projection = resident_nvfp4::Projection::new(owner, prefix, width, channels)?;
        Ok(Self::from_binding(&projection.workspace_binding()))
    }
}

pub(super) struct GdnWeights {
    pub(super) norm: u64,
    pub(super) post_norm: u64,
    pub(super) qkv: ProjectionWeights,
    pub(super) z: ProjectionWeights,
    pub(super) a: u64,
    pub(super) b: u64,
    pub(super) conv: u64,
    pub(super) a_log: u64,
    pub(super) dt_bias: u64,
    pub(super) f32_params: bool,
    pub(super) gated_norm: u64,
    pub(super) out: ProjectionWeights,
    pub(super) history: String,
    pub(super) recurrent: String,
}

pub(super) struct AttentionWeights {
    pub(super) norm: u64,
    pub(super) post_norm: u64,
    pub(super) q: ProjectionWeights,
    pub(super) k: ProjectionWeights,
    pub(super) v: ProjectionWeights,
    pub(super) out: ProjectionWeights,
    pub(super) q_norm: u64,
    pub(super) k_norm: u64,
    pub(super) key_state: String,
    pub(super) value_state: String,
}

pub(super) struct MlpWeights {
    pub(super) gate: ProjectionWeights,
    pub(super) up: ProjectionWeights,
    pub(super) down: ProjectionWeights,
    pub(super) nvfp4: bool,
}

pub(super) enum Block {
    Gdn(Box<GdnWeights>),
    Attention(Box<AttentionWeights>),
}

pub(super) struct LayerWeights {
    pub(super) block: Block,
    pub(super) mlp: MlpWeights,
}

pub(super) struct ModelWeights {
    pub(super) embedding_table: u64,
    pub(super) embedding_scale: Option<u64>,
    pub(super) first_norm: u64,
    pub(super) final_norm: u64,
    pub(super) head: ProjectionWeights,
    pub(super) layers: Vec<LayerWeights>,
}

fn bf16_vector(owner: &ResidentWeights<'_>, name: &str, width: usize) -> Result<u64> {
    owner.tensor(
        name,
        DType::Bf16,
        &[u64::try_from(width)?],
        u64::try_from(width * 2)?,
    )
}

impl ModelWeights {
    pub(super) fn representation_report(&self) -> serde_json::Value {
        let gdn_parameters: Vec<_> = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| match &layer.block {
                Block::Gdn(w) => Some(serde_json::json!({
                    "layer": index,
                    "a_log_dt_bias": if w.f32_params { "f32" } else { "bf16" },
                    "beta": "bf16",
                })),
                Block::Attention(_) => None,
            })
            .collect();
        serde_json::json!({
            "dispatch": "verified logical tensor dtype, independent of source container",
            "embedding": if self.embedding_scale.is_some() { "fp8_e4m3 + bf16 row scale" } else { "bf16" },
            "embedding_norm_input": "bf16",
            "gdn_parameters": gdn_parameters,
            "full_ninfer_arithmetic_parity": false,
        })
    }

    pub(super) fn bind(owner: &ResidentWeights<'_>, config: &DecoderConfig) -> Result<Self> {
        validate_shapes(config)?;
        let hidden = config.hidden;
        let vocabulary = config.vocabulary;
        let mut layers = Vec::with_capacity(config.layers.len());
        for layer in &config.layers {
            let nvfp4 = matches!(layer.mlp, DecoderMlpKind::Nvfp4);
            let block = match layer.block {
                DecoderBlockKind::Gdn => Block::Gdn(Box::new(bind_gdn(
                    owner,
                    config,
                    &layer.prefix,
                    &layer.state_prefix,
                )?)),
                DecoderBlockKind::Attention => Block::Attention(Box::new(bind_attention(
                    owner,
                    config,
                    &layer.prefix,
                    &layer.state_prefix,
                )?)),
            };
            let mlp = bind_mlp(owner, config, &format!("{}.mlp", layer.prefix), nvfp4)?;
            layers.push(LayerWeights { block, mlp });
        }
        let (embedding_table, embedding_scale) =
            resident_embedding::bind_table(owner, &config.embedding_table, vocabulary, hidden)?;
        Ok(Self {
            embedding_table,
            embedding_scale,
            first_norm: bf16_vector(owner, &config.first_norm, hidden)?,
            final_norm: bf16_vector(owner, &config.final_norm, hidden)?,
            head: ProjectionWeights::fp8(owner, &config.head_prefix, hidden, vocabulary)?,
            layers,
        })
    }
}

fn validate_shapes(config: &DecoderConfig) -> Result<()> {
    let gdn = &config.gdn_shape;
    let attention = &config.attention_shape;
    ensure!(
        !config.layers.is_empty() && config.layers.len() <= 256,
        "invalid decoder layer count"
    );
    ensure!(
        (1..=32768).contains(&config.hidden) && (1..=262_144).contains(&config.vocabulary),
        "invalid hidden width or vocabulary"
    );
    ensure!(
        (1..=256).contains(&gdn.key_heads)
            && (1..=256).contains(&gdn.value_heads)
            && gdn.value_heads.is_multiple_of(gdn.key_heads)
            && (1..=256).contains(&gdn.head_width)
            && (2 * gdn.key_heads + gdn.value_heads) * gdn.head_width <= 32_768,
        "invalid GDN dimensions"
    );
    ensure!(
        (1..=128).contains(&attention.query_heads)
            && (1..=128).contains(&attention.kv_heads)
            && attention.query_heads.is_multiple_of(attention.kv_heads)
            && (2..=256).contains(&attention.head_width)
            && (2..=attention.head_width).contains(&attention.rotary_dim)
            && attention.rotary_dim.is_multiple_of(2),
        "invalid attention dimensions"
    );
    Ok(())
}

fn bind_gdn(
    owner: &ResidentWeights<'_>,
    config: &DecoderConfig,
    prefix: &str,
    state_prefix: &str,
) -> Result<GdnWeights> {
    let shape = &config.gdn_shape;
    let hidden = config.hidden;
    let channels = (2 * shape.key_heads + shape.value_heads) * shape.head_width;
    let inner = shape.value_heads * shape.head_width;
    let heads = shape.value_heads;
    let attention = format!("{prefix}.linear_attn");
    let heads_u64 = u64::try_from(heads)?;
    let bf16_linear = |name: &str| -> Result<u64> {
        owner.tensor(
            &format!("{attention}.{name}.weight"),
            DType::Bf16,
            &[heads_u64, u64::try_from(hidden)?],
            u64::try_from(heads * hidden * 2)?,
        )
    };
    let (a_log, dt_bias, f32_params) =
        resident_gdn_core::bind_parameters(owner, &attention, heads)?;
    Ok(GdnWeights {
        norm: bf16_vector(owner, &format!("{prefix}.input_layernorm.weight"), hidden)?,
        post_norm: bf16_vector(
            owner,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?,
        qkv: ProjectionWeights::fp8(owner, &format!("{attention}.in_proj_qkv"), hidden, channels)?,
        z: ProjectionWeights::fp8(owner, &format!("{attention}.in_proj_z"), hidden, inner)?,
        a: bf16_linear("in_proj_a")?,
        b: bf16_linear("in_proj_b")?,
        conv: owner.tensor(
            &format!("{attention}.conv1d.weight"),
            DType::Bf16,
            &[u64::try_from(channels)?, 1, 4],
            u64::try_from(channels * 8)?,
        )?,
        a_log,
        dt_bias,
        f32_params,
        gated_norm: bf16_vector(owner, &format!("{attention}.norm.weight"), shape.head_width)?,
        out: ProjectionWeights::fp8(owner, &format!("{attention}.out_proj"), inner, hidden)?,
        history: format!("{state_prefix}.gdn.history"),
        recurrent: format!("{state_prefix}.gdn.recurrent"),
    })
}

fn bind_attention(
    owner: &ResidentWeights<'_>,
    config: &DecoderConfig,
    prefix: &str,
    state_prefix: &str,
) -> Result<AttentionWeights> {
    let shape = &config.attention_shape;
    let hidden = config.hidden;
    let q_width = shape.query_heads * shape.head_width;
    let kv_width = shape.kv_heads * shape.head_width;
    let attention = format!("{prefix}.self_attn");
    let fp8 = |name: &str, width: usize, channels: usize| {
        ProjectionWeights::fp8(owner, &format!("{attention}.{name}"), width, channels)
    };
    Ok(AttentionWeights {
        norm: bf16_vector(owner, &format!("{prefix}.input_layernorm.weight"), hidden)?,
        post_norm: bf16_vector(
            owner,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?,
        q: fp8("q_proj", hidden, q_width * 2)?,
        k: fp8("k_proj", hidden, kv_width)?,
        v: fp8("v_proj", hidden, kv_width)?,
        out: fp8("o_proj", q_width, hidden)?,
        q_norm: bf16_vector(
            owner,
            &format!("{attention}.q_norm.weight"),
            shape.head_width,
        )?,
        k_norm: bf16_vector(
            owner,
            &format!("{attention}.k_norm.weight"),
            shape.head_width,
        )?,
        key_state: format!("{state_prefix}.attention.k"),
        value_state: format!("{state_prefix}.attention.v"),
    })
}

fn bind_mlp(
    owner: &ResidentWeights<'_>,
    config: &DecoderConfig,
    prefix: &str,
    nvfp4: bool,
) -> Result<MlpWeights> {
    let hidden = config.hidden;
    let channels = config.gdn_shape.intermediate;
    ensure!(
        channels == config.attention_shape.intermediate,
        "GDN and attention MLP widths differ"
    );
    let bind = |name: &str, width: usize, out: usize| {
        let name = format!("{prefix}.{name}");
        if nvfp4 {
            ProjectionWeights::nvfp4(owner, &name, width, out)
        } else {
            ProjectionWeights::fp8(owner, &name, width, out)
        }
    };
    Ok(MlpWeights {
        gate: bind("gate_proj", hidden, channels)?,
        up: bind("up_proj", hidden, channels)?,
        down: bind("down_proj", channels, hidden)?,
        nvfp4,
    })
}
