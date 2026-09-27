//! Complete GDN decoder block using resident weight and state views.

use super::{
    driver::{Buffer, Context, Module},
    resident_bf16,
    resident_conv::Convolution,
    resident_fp8,
    resident_gdn_core::{GdnCore, Input},
    resident_mlp::{Mlp, Quantization},
    resident_norm::{Norm, residual_add},
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use anyhow::{Result, ensure};

use crate::kernels::GdnShape as Shape;

pub(super) struct Layer<'w, 'ctx> {
    norm: Norm<'w, 'ctx>,
    post_norm: Norm<'w, 'ctx>,
    qkv: resident_fp8::Projection<'w, 'ctx>,
    z: resident_fp8::Projection<'w, 'ctx>,
    a: resident_bf16::Projection<'w, 'ctx>,
    b: resident_bf16::Projection<'w, 'ctx>,
    conv: Convolution<'w, 'ctx>,
    core: GdnCore<'w, 'ctx>,
    out: resident_fp8::Projection<'w, 'ctx>,
    mlp: Mlp<'w, 'ctx>,
    history: String,
    recurrent: String,
}

impl<'w, 'ctx> Layer<'w, 'ctx> {
    pub(super) fn new(
        weights: &'w ResidentWeights<'ctx>,
        prefix: &str,
        state_prefix: &str,
        shape: &Shape,
        quantization: Quantization,
    ) -> Result<Self> {
        ensure!(
            (1..=256).contains(&shape.key_heads)
                && (1..=256).contains(&shape.value_heads)
                && (1..=256).contains(&shape.head_width),
            "invalid resident GDN head dimensions"
        );
        let hidden = shape.hidden;
        let channels = (2 * shape.key_heads + shape.value_heads) * shape.head_width;
        let inner = shape.value_heads * shape.head_width;
        let attention = format!("{prefix}.linear_attn");
        Ok(Self {
            norm: Norm::new(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                1e-6,
            )?,
            post_norm: Norm::new(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                1e-6,
            )?,
            qkv: resident_fp8::Projection::new(
                weights,
                &format!("{attention}.in_proj_qkv"),
                hidden,
                channels,
            )?,
            z: resident_fp8::Projection::new(
                weights,
                &format!("{attention}.in_proj_z"),
                hidden,
                inner,
            )?,
            a: resident_bf16::Projection::new(
                weights,
                &format!("{attention}.in_proj_a.weight"),
                hidden,
                shape.value_heads,
            )?,
            b: resident_bf16::Projection::new(
                weights,
                &format!("{attention}.in_proj_b.weight"),
                hidden,
                shape.value_heads,
            )?,
            conv: Convolution::new(weights, &format!("{attention}.conv1d.weight"), channels)?,
            core: GdnCore::new(
                weights,
                &attention,
                shape.key_heads,
                shape.value_heads,
                shape.head_width,
            )?,
            out: resident_fp8::Projection::new(
                weights,
                &format!("{attention}.out_proj"),
                inner,
                hidden,
            )?,
            mlp: Mlp::new(
                weights,
                &format!("{prefix}.mlp"),
                hidden,
                shape.intermediate,
                quantization,
            )?,
            history: format!("{state_prefix}.gdn.history"),
            recurrent: format!("{state_prefix}.gdn.recurrent"),
        })
    }

    pub(super) fn forward<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        let normalized = self.norm.run(ctx, module, hidden, rows)?;
        let qkv = self.qkv.run(ctx, module, &normalized, rows)?;
        let z = self.z.run(ctx, module, &normalized, rows)?;
        let a = self.a.run(ctx, module, &normalized, rows)?;
        let b = self.b.run(ctx, module, &normalized, rows)?;
        let conv = self
            .conv
            .run(ctx, module, &qkv.values, state, &self.history, rows)?;
        let gated = self.core.run(
            ctx,
            module,
            Input {
                qkv: &conv,
                a: &a.values,
                b: &b.values,
                z: &z.values,
            },
            state,
            &self.recurrent,
            rows,
        )?;
        let branch = self.out.run(ctx, module, &gated, rows)?;
        let post = self
            .post_norm
            .add(ctx, module, hidden, &branch.values, rows)?;
        let mlp = self.mlp.run(ctx, module, &post.normalized, rows)?;
        residual_add(ctx, module, &post.residual, &mlp.down.values)
    }
}
