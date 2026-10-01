use super::super::{HIDDEN, Model};
use crate::kernels::cuda::native_mtp_q8_fc_operator::launch::{
    self as fc_launch, DeviceInputRequest as FcDeviceInput,
};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    resident_state::ResidentState,
};
use anyhow::{Context as _, Result};

pub(super) fn run<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    state: &mut ResidentState<'_>,
    past: usize,
) -> Result<super::super::attention::BlockOutput<'a>> {
    let rows = tokens.len();
    let embedding = model.embedding.run(context, module, tokens)?;
    let hidden = model
        .hidden_norm
        .run(context, module, target_hidden, rows)?;
    let fc_input = concatenate(context, &embedding.normalized, &hidden, rows, HIDDEN)?;
    let fc = fc_launch::run_device_input(FcDeviceInput {
        context,
        module,
        resident: model.native,
        view: &model.native.views().fc,
        input: &fc_input,
        tokens: rows,
    })?;
    super::super::attention::run(model, context, module, &fc, state, rows, past)
}

fn concatenate<'a>(
    context: &'a Context,
    embedding: &Buffer<'_>,
    hidden: &Buffer<'_>,
    rows: usize,
    width: usize,
) -> Result<Buffer<'a>> {
    let row_bytes = width
        .checked_mul(2)
        .context("native MTP FC row extent overflow")?;
    let input_bytes = rows
        .checked_mul(row_bytes)
        .context("native MTP input extent overflow")?;
    anyhow::ensure!(
        embedding.belongs_to(context)
            && hidden.belongs_to(context)
            && embedding.len() == input_bytes
            && hidden.len() == input_bytes,
        "native MTP embedding or hidden extent mismatch"
    );
    let output_row_bytes = row_bytes
        .checked_mul(2)
        .context("native MTP FC row extent overflow")?;
    let output = Buffer::new(
        context,
        rows.checked_mul(output_row_bytes)
            .context("native MTP FC input extent overflow")?,
    )?;
    for row in 0..rows {
        let source = row
            .checked_mul(row_bytes)
            .context("native MTP input source offset overflow")?;
        let destination = row
            .checked_mul(output_row_bytes)
            .context("native MTP input destination offset overflow")?;
        output.copy_from_at(destination, embedding, source, row_bytes)?;
        output.copy_from_at(destination + row_bytes, hidden, source, row_bytes)?;
    }
    Ok(output)
}
