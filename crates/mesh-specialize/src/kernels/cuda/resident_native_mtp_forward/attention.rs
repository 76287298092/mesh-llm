use super::{HEAD_WIDTH, KV_HEADS, Model, QUERY_HEADS};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    native_mtp_q8_projection::{self, DeviceProjectionRequest, ProjectionKind},
    resident_attention_core::{self, Input, Shape},
    resident_state::ResidentState,
};
use anyhow::Result;

pub(super) struct BlockOutput<'ctx> {
    pub(super) residual: Buffer<'ctx>,
    pub(super) normalized: Buffer<'ctx>,
}

pub(super) fn run<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    hidden: &Buffer<'_>,
    state: &mut ResidentState<'_>,
    rows: usize,
    past: usize,
) -> Result<BlockOutput<'a>> {
    let normalized = model.input_norm.run(context, module, hidden, rows)?;
    let qkv = native_mtp_q8_projection::project_resident_q8_device(DeviceProjectionRequest {
        context,
        module,
        resident: model.native,
        carrier_view: &model.native.views().query_gate,
        input: &normalized,
        kind: ProjectionKind::QueryKeyValue,
        tokens: rows,
    })?;
    let projections = super::attention_ops::prepare(context, module, model, &qkv, rows, past)?;
    let attended = resident_attention_core::run(
        context,
        module,
        Input {
            q: &projections.query.values,
            k: &projections.key.values,
            v: &projections.value,
        },
        state,
        "mtp",
        &Shape {
            rows,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            width: HEAD_WIDTH,
            past,
            capacity: model.capacity,
        },
    )?;
    let gated =
        super::activation::attention_gate(context, module, &attended, &projections.query.gate)?;
    let projected =
        native_mtp_q8_projection::project_resident_q8_device(DeviceProjectionRequest {
            context,
            module,
            resident: model.native,
            carrier_view: &model.native.views().attention_output,
            input: &gated,
            kind: ProjectionKind::AttentionOutput,
            tokens: rows,
        })?;
    let post = model
        .post_attention_norm
        .add(context, module, hidden, &projected, rows)?;
    Ok(BlockOutput {
        residual: post.residual,
        normalized: post.normalized,
    })
}
