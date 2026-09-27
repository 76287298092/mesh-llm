//! Complete attention decoder block over persistent weight and K/V arenas.
use super::{
    driver::{Buffer, Context, Module},
    resident_attention_core::{self, Input},
    resident_attention_gate,
    resident_attention_prepare::{Preparation, Tables},
    resident_mlp::Mlp,
    resident_norm::{Norm, residual_add},
    resident_projection::{Projection, Quantization},
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{engine::rope::TextRope, kernels::ResidentAttentionShape};
use anyhow::{Result, ensure};

pub(super) type StageObserver<'a> = dyn FnMut(&str, &Buffer<'_>) -> Result<()> + 'a;

pub(super) struct Layer<'w, 'ctx> {
    norm: Norm<'w, 'ctx>,
    post_norm: Norm<'w, 'ctx>,
    q: Projection<'w, 'ctx>,
    k: Projection<'w, 'ctx>,
    v: Projection<'w, 'ctx>,
    out: Projection<'w, 'ctx>,
    q_prepare: Preparation<'w, 'ctx>,
    k_prepare: Preparation<'w, 'ctx>,
    mlp: Mlp<'w, 'ctx>,
    rope: TextRope,
    query_heads: usize,
    kv_heads: usize,
    width: usize,
    state_prefix: String,
}
pub(super) struct Step {
    pub rows: usize,
    pub past: usize,
    pub capacity: usize,
}
impl<'w, 'ctx> Layer<'w, 'ctx> {
    pub(super) fn attach_workspace(&mut self, workspace: super::model_workspace::Shared<'ctx>) {
        self.mlp.attach_workspace(workspace);
    }
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        state_prefix: &str,
        shape: &ResidentAttentionShape,
        quantization: Quantization,
    ) -> Result<Self> {
        ensure!(
            (1..=128).contains(&shape.query_heads)
                && (1..=128).contains(&shape.kv_heads)
                && shape.query_heads.is_multiple_of(shape.kv_heads)
                && (2..=256).contains(&shape.head_width),
            "invalid resident attention dimensions"
        );
        let q_width = shape.query_heads * shape.head_width;
        let kv_width = shape.kv_heads * shape.head_width;
        let attention = format!("{prefix}.self_attn");
        let attention_quantization = match quantization {
            Quantization::Bf16 => Quantization::Bf16,
            Quantization::Fp8 | Quantization::Nvfp4 => Quantization::Fp8,
        };
        Ok(Self {
            norm: Norm::new(
                owner,
                &format!("{prefix}.input_layernorm.weight"),
                shape.hidden,
                1e-6,
            )?,
            post_norm: Norm::new(
                owner,
                &format!("{prefix}.post_attention_layernorm.weight"),
                shape.hidden,
                1e-6,
            )?,
            q: Projection::new(
                owner,
                &format!("{attention}.q_proj"),
                shape.hidden,
                q_width * 2,
                attention_quantization,
            )?,
            k: Projection::new(
                owner,
                &format!("{attention}.k_proj"),
                shape.hidden,
                kv_width,
                attention_quantization,
            )?,
            v: Projection::new(
                owner,
                &format!("{attention}.v_proj"),
                shape.hidden,
                kv_width,
                attention_quantization,
            )?,
            out: Projection::new(
                owner,
                &format!("{attention}.o_proj"),
                q_width,
                shape.hidden,
                attention_quantization,
            )?,
            q_prepare: Preparation::new(
                owner,
                &format!("{attention}.q_norm.weight"),
                shape.query_heads,
                shape.head_width,
                shape.rotary_dim,
                true,
            )?,
            k_prepare: Preparation::new(
                owner,
                &format!("{attention}.k_norm.weight"),
                shape.kv_heads,
                shape.head_width,
                shape.rotary_dim,
                false,
            )?,
            mlp: Mlp::new(
                owner,
                &format!("{prefix}.mlp"),
                shape.hidden,
                shape.intermediate,
                quantization,
            )?,
            rope: TextRope::new(shape.rotary_dim, shape.rope_theta)?,
            query_heads: shape.query_heads,
            kv_heads: shape.kv_heads,
            width: shape.head_width,
            state_prefix: state_prefix.into(),
        })
    }

    /// Caller commits the sequence cursor only after every decoder layer succeeds.
    /// A failed launch can leave K/V partially updated; discard that session on error.
    pub(super) fn forward<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        step: &Step,
    ) -> Result<Buffer<'a>> {
        self.forward_observed(ctx, module, hidden, state, step, None)
    }

    pub(super) fn forward_observed<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        state: &mut ResidentState<'_>,
        step: &Step,
        mut observer: Option<&mut StageObserver<'_>>,
    ) -> Result<Buffer<'a>> {
        ensure!(
            (1..=262144).contains(&step.capacity)
                && step
                    .past
                    .checked_add(step.rows)
                    .is_some_and(|end| end <= step.capacity),
            "resident attention context capacity exceeded"
        );
        let tables = self.rope.tables(step.past, step.rows)?;
        let cos = upload_words(ctx, &tables.cos)?;
        let sin = upload_words(ctx, &tables.sin)?;
        let normalized = self.norm.run(ctx, module, hidden, step.rows)?;
        observe(&mut observer, "normalized", &normalized)?;
        let q_linear = self.q.run(ctx, module, &normalized, step.rows)?;
        observe(&mut observer, "q_linear", &q_linear.values)?;
        let k_linear = self.k.run(ctx, module, &normalized, step.rows)?;
        observe(&mut observer, "k_linear", &k_linear.values)?;
        let v_linear = self.v.run(ctx, module, &normalized, step.rows)?;
        observe(&mut observer, "v_linear", &v_linear.values)?;
        let prepared_q = self.q_prepare.run(
            ctx,
            module,
            &q_linear.values,
            Tables {
                cos: &cos,
                sin: &sin,
            },
            step.rows,
        )?;
        observe(&mut observer, "q_prepared", &prepared_q.values)?;
        observe(&mut observer, "q_gate", &prepared_q.gate)?;
        let prepared_k = self.k_prepare.run(
            ctx,
            module,
            &k_linear.values,
            Tables {
                cos: &cos,
                sin: &sin,
            },
            step.rows,
        )?;
        observe(&mut observer, "k_prepared", &prepared_k.values)?;
        let attended = resident_attention_core::run(
            ctx,
            module,
            Input {
                q: &prepared_q.values,
                k: &prepared_k.values,
                v: &v_linear.values,
            },
            state,
            &self.state_prefix,
            &resident_attention_core::Shape {
                rows: step.rows,
                query_heads: self.query_heads,
                kv_heads: self.kv_heads,
                width: self.width,
                past: step.past,
                capacity: step.capacity,
            },
        )?;
        observe(&mut observer, "attended", &attended)?;
        let gated = resident_attention_gate::run(ctx, module, &attended, &prepared_q.gate)?;
        observe(&mut observer, "gated", &gated)?;
        let branch = self.out.run(ctx, module, &gated, step.rows)?;
        observe(&mut observer, "out", &branch.values)?;
        let post = self
            .post_norm
            .add(ctx, module, hidden, &branch.values, step.rows)?;
        observe(&mut observer, "post_residual", &post.residual)?;
        observe(&mut observer, "post_norm", &post.normalized)?;
        let output = if self.mlp.has_workspace() && observer.is_none() {
            let down = self
                .mlp
                .workspace_output(ctx, module, &post.normalized, step.rows)?;
            residual_add(ctx, module, &post.residual, &down)?
        } else {
            let mlp = self.mlp.run(ctx, module, &post.normalized, step.rows)?;
            observe(&mut observer, "mlp_gate", &mlp.gate.values)?;
            observe(&mut observer, "mlp_up", &mlp.up.values)?;
            observe(&mut observer, "mlp_activation", &mlp.activation)?;
            observe(&mut observer, "mlp_down", &mlp.down.values)?;
            residual_add(ctx, module, &post.residual, &mlp.down.values)?
        };
        observe(&mut observer, "hidden", &output)?;
        Ok(output)
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

fn upload_words<'a>(ctx: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    let bytes = words
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let result = Buffer::new(ctx, bytes.len())?;
    result.upload(&bytes)?;
    Ok(result)
}
