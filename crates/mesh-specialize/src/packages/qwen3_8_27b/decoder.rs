//! Fixed text decoder configuration projected from the compiled schedule.
use super::schedule::{AttentionKind, MlpKind, Schedule};
use crate::kernels::{
    DecoderBlockKind, DecoderConfig, DecoderLayer, DecoderMlpKind, GdnShape, ResidentAttentionShape,
};
use anyhow::Result;

pub fn config(capacity: usize) -> Result<DecoderConfig> {
    let schedule = Schedule::new(capacity)?;
    Ok(DecoderConfig {
        layers: schedule
            .layers
            .iter()
            .map(|layer| DecoderLayer {
                prefix: format!("tensors/model.language_model.layers.{}", layer.index),
                state_prefix: format!("layers.{:02}", layer.index),
                block: match layer.attention {
                    AttentionKind::Gdn => DecoderBlockKind::Gdn,
                    AttentionKind::Full => DecoderBlockKind::Attention,
                },
                mlp: match layer.mlp {
                    MlpKind::Nvfp4 => DecoderMlpKind::Nvfp4,
                    MlpKind::Fp8 => DecoderMlpKind::Fp8,
                },
            })
            .collect(),
        gdn_shape: GdnShape {
            hidden: 5120,
            intermediate: 17408,
            key_heads: 16,
            value_heads: 48,
            head_width: 128,
        },
        attention_shape: ResidentAttentionShape {
            hidden: 5120,
            intermediate: 17408,
            query_heads: 24,
            kv_heads: 4,
            head_width: 256,
            rotary_dim: 64,
            rope_theta: 1e7,
        },
        embedding_table: "tensors/model.language_model.embed_tokens.weight".into(),
        first_norm: "tensors/model.language_model.layers.0.input_layernorm.weight".into(),
        final_norm: "tensors/model.language_model.norm.weight".into(),
        head_prefix: "tensors/lm_head".into(),
        hidden: 5120,
        vocabulary: 248320,
        capacity,
        state_layout: schedule.states,
    })
}
