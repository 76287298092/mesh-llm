use super::super::{BF16_BYTES, INTERMEDIATE, Model};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    native_mtp_q8_projection::{self, DeviceProjectionRequest, ProjectionKind},
    resident_norm::residual_add,
};
use anyhow::{Context as _, Result, ensure};

pub(super) fn run<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    attention_output: &super::super::attention::BlockOutput<'_>,
    rows: usize,
) -> Result<Buffer<'a>> {
    let gate_up = native_mtp_q8_projection::project_resident_q8_device(DeviceProjectionRequest {
        context,
        module,
        resident: model.native,
        carrier_view: &model.native.views().mlp_gate,
        input: &attention_output.normalized,
        kind: ProjectionKind::MlpGateUp,
        tokens: rows,
    })?;
    let (gate, up) = split_gate_up(context, &gate_up, rows)?;
    let activation = super::super::activation::silu_mul(context, module, &gate, &up)?;
    let down = native_mtp_q8_projection::project_resident_q8_device(DeviceProjectionRequest {
        context,
        module,
        resident: model.native,
        carrier_view: &model.native.views().mlp_down,
        input: &activation,
        kind: ProjectionKind::MlpDown,
        tokens: rows,
    })?;
    let residual = residual_add(context, module, &attention_output.residual, &down)?;
    model.final_norm.run(context, module, &residual, rows)
}

fn split_gate_up<'a>(
    context: &'a Context,
    combined: &Buffer<'_>,
    rows: usize,
) -> Result<(Buffer<'a>, Buffer<'a>)> {
    let row_bytes = INTERMEDIATE
        .checked_mul(2)
        .context("native MLP projection row extent overflow")?;
    let expected = rows
        .checked_mul(row_bytes)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("native MLP projection extent overflow")?;
    ensure!(
        combined.belongs_to(context) && combined.len() == expected,
        "native MLP projection output extent mismatch"
    );
    let branch_bytes = rows
        .checked_mul(INTERMEDIATE)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("native MLP branch extent overflow")?;
    let gate = Buffer::new(context, branch_bytes)?;
    let up = Buffer::new(context, branch_bytes)?;
    for row in 0..rows {
        let source = row
            .checked_mul(row_bytes * BF16_BYTES)
            .context("native MLP source offset overflow")?;
        let destination = row
            .checked_mul(INTERMEDIATE * BF16_BYTES)
            .context("native MLP destination offset overflow")?;
        gate.copy_from_at(destination, combined, source, INTERMEDIATE * BF16_BYTES)?;
        up.copy_from_at(
            destination,
            combined,
            source + INTERMEDIATE * BF16_BYTES,
            INTERMEDIATE * BF16_BYTES,
        )?;
    }
    Ok((gate, up))
}
