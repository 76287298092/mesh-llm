//! Complete GDN decoder block using resident weight and state views.

use super::{
    driver::{Buffer, Context, Module},
    resident_bf16,
    resident_conv::Convolution,
    resident_fp8,
    resident_gdn_core::{GdnCore, Input},
    resident_mlp::Mlp,
    resident_norm::{Norm, residual_add},
    resident_projection::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use anyhow::{Result, ensure};

use crate::kernels::GdnShape as Shape;

pub(super) type StageObserver<'a> = dyn FnMut(&str, &Buffer<'_>) -> Result<()> + 'a;

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
        self.forward_observed(ctx, module, hidden, state, rows, None)
    }

    pub(super) fn forward_observed<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        rows: usize,
        observer: Option<&mut StageObserver<'_>>,
    ) -> Result<Buffer<'a>> {
        Ok(self
            .execute(ctx, module, hidden, state, (rows, false), observer)?
            .0)
    }

    pub(super) fn forward_recorded<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        rows: usize,
    ) -> Result<(Buffer<'a>, super::resident_recovery::LayerRecord<'a>)> {
        let (hidden, record) = self.execute(ctx, module, hidden, state, (rows, true), None)?;
        Ok((
            hidden,
            record.ok_or_else(|| anyhow::anyhow!("missing GDN record"))?,
        ))
    }

    fn execute<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        step: (usize, bool),
        mut observer: Option<&mut StageObserver<'_>>,
    ) -> Result<(
        Buffer<'a>,
        Option<super::resident_recovery::LayerRecord<'a>>,
    )> {
        let (rows, record) = step;
        let normalized = self.norm.run(ctx, module, hidden, rows)?;
        observe(&mut observer, "normalized", &normalized)?;
        let qkv = self.qkv.run(ctx, module, &normalized, rows)?;
        observe(&mut observer, "qkv", &qkv.values)?;
        let z = self.z.run(ctx, module, &normalized, rows)?;
        observe(&mut observer, "z", &z.values)?;
        let a = self.a.run(ctx, module, &normalized, rows)?;
        observe(&mut observer, "a", &a.values)?;
        let b = self.b.run(ctx, module, &normalized, rows)?;
        observe(&mut observer, "b", &b.values)?;
        let conv = self
            .conv
            .run(ctx, module, &qkv.values, state, &self.history, rows)?;
        observe(&mut observer, "convolution", &conv)?;
        let gated = self.core.run(
            ctx,
            module,
            Input {
                qkv: &conv,
                a: &a.values,
                b: &b.values,
                z: &z.values,
                record,
            },
            state,
            &self.recurrent,
            rows,
        )?;
        let recovery = gated.record;
        let gated = gated.output;
        observe(&mut observer, "gated", &gated)?;
        let branch = self.out.run(ctx, module, &gated, rows)?;
        observe(&mut observer, "out", &branch.values)?;
        let post = self
            .post_norm
            .add(ctx, module, hidden, &branch.values, rows)?;
        observe(&mut observer, "post_residual", &post.residual)?;
        observe(&mut observer, "post_norm", &post.normalized)?;
        let mlp = self.mlp.run(ctx, module, &post.normalized, rows)?;
        observe(&mut observer, "mlp_gate", &mlp.gate.values)?;
        observe(&mut observer, "mlp_up", &mlp.up.values)?;
        observe(&mut observer, "mlp_activation", &mlp.activation)?;
        observe(&mut observer, "mlp_down", &mlp.down.values)?;
        let output = residual_add(ctx, module, &post.residual, &mlp.down.values)?;
        observe(&mut observer, "hidden", &output)?;
        let record = recovery.map(|recurrence| super::resident_recovery::LayerRecord {
            recurrence,
            projected_qkv: qkv.values,
            history_name: self.history.clone(),
            recurrent_name: self.recurrent.clone(),
        });
        Ok((output, record))
    }
}

fn observe(
    observer: &mut Option<&mut StageObserver<'_>>,
    name: &str,
    buffer: &Buffer<'_>,
) -> Result<()> {
    if let Some(callback) = observer.as_deref_mut() {
        callback(name, buffer)?;
    }
    Ok(())
}
