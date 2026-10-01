mod activation;
mod attention;
mod attention_ops;
mod checks;
mod execution;
mod selection;
#[cfg(test)]
mod tests;

use super::{
    driver::{Buffer, Context, Module},
    resident_attention_prepare::Preparation,
    resident_embedding::Embedding,
    resident_mtp,
    resident_native_mtp::ResidentNativeMtp,
    resident_norm::Norm,
    resident_weights::ResidentWeights,
};
use crate::{engine::layout::Layout, engine::rope::TextRope, kernels::DecoderConfig};
use anyhow::{Result, ensure};

pub(super) type Session<'ctx> = resident_mtp::Session<'ctx>;

const HIDDEN: usize = 5_120;
const BF16_BYTES: usize = 2;
const MAX_ROWS: usize = 5;
const MAX_CONTEXT: usize = 262_144;
const INTERMEDIATE: usize = 17_408;
const QUERY_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_WIDTH: usize = 256;
const ROTARY_DIM: usize = 64;
const VOCABULARY: usize = 248_320;
const PROPOSAL_ROWS: usize = 131_072;
const EPSILON: f32 = 1e-6;

pub(super) struct Model<'w, 'ctx> {
    target_owner: &'w ResidentWeights<'ctx>,
    native: &'w ResidentNativeMtp<'ctx>,
    target_norm: Norm<'w, 'ctx>,
    embedding: Embedding<'w, 'ctx>,
    hidden_norm: Norm<'w, 'ctx>,
    input_norm: Norm<'w, 'ctx>,
    post_attention_norm: Norm<'w, 'ctx>,
    final_norm: Norm<'w, 'ctx>,
    query_prepare: Preparation<'w, 'ctx>,
    key_prepare: Preparation<'w, 'ctx>,
    rope: TextRope,
    layout: Layout,
    capacity: usize,
}

pub(super) struct Output<'ctx> {
    pub(super) hidden: Buffer<'ctx>,
    pub(super) logits: Vec<u16>,
    pub(super) token: u32,
    pub(super) proposal_row: usize,
    pub(super) past: usize,
}

impl<'w, 'ctx> Model<'w, 'ctx> {
    pub(super) fn new(
        target_owner: &'w ResidentWeights<'ctx>,
        native: &'w ResidentNativeMtp<'ctx>,
        config: &DecoderConfig,
    ) -> Result<Self> {
        checks::validate_model_config(config)?;
        ensure!(
            target_owner.belongs_to(native.context()),
            "target and native MTP weights belong to different CUDA contexts"
        );
        let views = native.views();
        let target_norm = Norm::new(target_owner, &config.final_norm, HIDDEN, EPSILON)?;
        let embedding = Embedding::from_native_mtp(
            target_owner,
            native,
            super::resident_embedding::NativeMtpEmbeddingConfig {
                table_name: &config.embedding_table,
                shape: [VOCABULARY, HIDDEN],
                epsilon: EPSILON,
            },
        )?;
        let hidden_norm =
            Norm::from_native_mtp(native.norm(&views.norms.hidden)?, HIDDEN, EPSILON)?;
        let input_norm = Norm::from_native_mtp(native.norm(&views.norms.input)?, HIDDEN, EPSILON)?;
        let post_attention_norm =
            Norm::from_native_mtp(native.norm(&views.norms.post_attention)?, HIDDEN, EPSILON)?;
        let final_norm =
            Norm::from_native_mtp(native.norm(&views.norms.final_norm)?, HIDDEN, EPSILON)?;
        let query_prepare = Preparation::from_native_mtp(
            native.norm(&views.norms.query)?,
            QUERY_HEADS,
            HEAD_WIDTH,
            ROTARY_DIM,
            true,
        )?;
        let key_prepare = Preparation::from_native_mtp(
            native.norm(&views.norms.key)?,
            KV_HEADS,
            HEAD_WIDTH,
            ROTARY_DIM,
            false,
        )?;
        selection::validate_views(views)?;
        Ok(Self {
            target_owner,
            native,
            target_norm,
            embedding,
            hidden_norm,
            input_norm,
            post_attention_norm,
            final_norm,
            query_prepare,
            key_prepare,
            rope: TextRope::new(ROTARY_DIM, config.attention_shape.rope_theta)?,
            layout: resident_mtp::mtp_state_layout(config)?,
            capacity: config.capacity,
        })
    }

    pub(super) fn target_hidden<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        raw: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        ensure!(
            self.target_owner.belongs_to(ctx) && module.belongs_to(ctx),
            "target hidden owner or module belongs to another CUDA context"
        );
        checks::validate_matrix(ctx, raw, rows, HIDDEN, "target hidden")?;
        self.target_norm.run(ctx, module, raw, rows)
    }

    pub(super) fn forward<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
        normalized_target_hidden: &Buffer<'_>,
        session: &mut resident_mtp::Session<'_>,
    ) -> Result<Output<'a>> {
        let rows =
            checks::validate_forward(self, ctx, module, tokens, normalized_target_hidden, session)?;
        if rows == 2 || rows == 3 || rows == 4 {
            return execution::forward_serial(
                self,
                ctx,
                module,
                tokens,
                normalized_target_hidden,
                session,
            );
        }
        execution::forward_rows(self, ctx, module, tokens, normalized_target_hidden, session)
    }
}
