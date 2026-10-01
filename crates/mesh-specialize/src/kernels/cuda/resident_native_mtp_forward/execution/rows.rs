use super::super::{BF16_BYTES, HIDDEN, Model, Output, Session};
use super::run_rows;
use crate::kernels::cuda::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result};

pub(super) fn forward_rows<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    session: &mut Session<'_>,
) -> Result<Output<'a>> {
    super::super::checks::validate_matrix(
        context,
        target_hidden,
        tokens.len(),
        HIDDEN,
        "native MTP target hidden",
    )?;
    super::super::checks::validate_session_capacity(&session.cursor, tokens.len())?;
    let transaction = session.cursor.begin(tokens.len())?;
    let (hidden, logits, proposal_row, token) = run_rows(
        model,
        context,
        module,
        tokens,
        target_hidden,
        &mut session.state,
        transaction.past(),
    )?;
    let past = transaction.commit();
    Ok(Output {
        hidden,
        logits,
        token,
        proposal_row,
        past,
    })
}

pub(super) fn forward_serial<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    session: &mut Session<'_>,
) -> Result<Output<'a>> {
    let rows = tokens.len();
    super::super::checks::validate_matrix(
        context,
        target_hidden,
        rows,
        HIDDEN,
        "native MTP target hidden",
    )?;
    super::super::checks::validate_session_capacity(&session.cursor, rows)?;
    let transaction = session.cursor.begin(rows)?;
    let mut output_rows = Vec::with_capacity(rows);
    for (row, &token) in tokens.iter().enumerate() {
        let target_row = select_row(context, target_hidden, row, HIDDEN)?;
        output_rows.push(run_rows(
            model,
            context,
            module,
            &[token],
            &target_row,
            &mut session.state,
            transaction.past() + row,
        )?);
    }
    let (hidden, logits, proposal_row, token) = combine_last(context, rows, output_rows)?;
    let past = transaction.commit();
    Ok(Output {
        hidden,
        logits,
        token,
        proposal_row,
        past,
    })
}

fn combine_last<'a>(
    context: &'a Context,
    rows: usize,
    mut output_rows: Vec<(Buffer<'_>, Vec<u16>, usize, u32)>,
) -> Result<(Buffer<'a>, Vec<u16>, usize, u32)> {
    let last = output_rows
        .pop()
        .context("native MTP serial fallback produced no rows")?;
    let extent = rows
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("native MTP serial hidden extent overflow")?;
    let hidden = Buffer::new(context, extent)?;
    for (row, (row_hidden, _, _, _)) in output_rows.into_iter().enumerate() {
        let destination_offset = row
            .checked_mul(HIDDEN * BF16_BYTES)
            .context("native MTP serial output offset overflow")?;
        hidden.copy_from_at(destination_offset, &row_hidden, 0, HIDDEN * BF16_BYTES)?;
    }
    let (last_hidden, logits, proposal_row, token) = last;
    let last_offset = rows
        .checked_sub(1)
        .and_then(|last_row| last_row.checked_mul(HIDDEN * BF16_BYTES))
        .context("native MTP serial final-row offset overflow")?;
    hidden.copy_from_at(last_offset, &last_hidden, 0, HIDDEN * BF16_BYTES)?;
    Ok((hidden, logits, proposal_row, token))
}

fn select_row<'a>(
    context: &'a Context,
    input: &Buffer<'_>,
    row: usize,
    width: usize,
) -> Result<Buffer<'a>> {
    super::super::checks::validate_matrix(
        context,
        input,
        input.len() / (width * BF16_BYTES),
        width,
        "native MTP target hidden",
    )?;
    let row_bytes = width
        .checked_mul(BF16_BYTES)
        .context("native MTP target row extent overflow")?;
    let offset = row
        .checked_mul(row_bytes)
        .context("native MTP target row offset overflow")?;
    let output = Buffer::new(context, row_bytes)?;
    output.copy_from_at(0, input, offset, row_bytes)?;
    Ok(output)
}
