use super::super::{BF16_BYTES, HIDDEN, Model, PROPOSAL_ROWS};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    native_mtp_q4_operator::resident::launch::{self as q4_launch, DeviceInput as Q4DeviceInput},
};
use anyhow::{Context as _, Result, ensure};

pub(super) fn run<'a>(
    model: &Model<'_, '_>,
    context: &'a Context,
    module: &Module<'_>,
    hidden: Buffer<'a>,
) -> Result<(Buffer<'a>, Vec<u16>, usize, u32)> {
    super::super::checks::validate_proposal_parent(model)?;
    let rows = hidden.len() / (HIDDEN * BF16_BYTES);
    let proposal_input = last_row(context, &hidden, rows, HIDDEN)?;
    let logits = q4_launch::run_device_input(Q4DeviceInput {
        context,
        module,
        resident: model.native,
        view: &model.native.views().proposal_head,
        input: &proposal_input,
    })?;
    let logits_words = download_words(&logits, PROPOSAL_ROWS)?;
    let proposal_row = first_argmax(&logits_words)?;
    let target = model
        .native
        .views()
        .proposal_tokens
        .target_id(proposal_row)
        .context("native proposal row is missing from target token map")?;
    Ok((hidden, logits_words, proposal_row, target.value()))
}

fn last_row<'a>(
    context: &'a Context,
    input: &Buffer<'_>,
    rows: usize,
    width: usize,
) -> Result<Buffer<'a>> {
    ensure!(
        rows > 0 && input.belongs_to(context) && input.len() == rows * width * BF16_BYTES,
        "native MTP final hidden extent mismatch"
    );
    let row_bytes = width
        .checked_mul(BF16_BYTES)
        .context("native MTP final hidden row extent overflow")?;
    let offset = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(row_bytes))
        .context("native MTP final hidden offset overflow")?;
    let output = Buffer::new(context, row_bytes)?;
    output.copy_from_at(0, input, offset, row_bytes)?;
    Ok(output)
}

fn download_words(input: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let bytes = count
        .checked_mul(BF16_BYTES)
        .context("native proposal logits extent overflow")?;
    ensure!(
        input.len() == bytes,
        "native proposal logits extent mismatch"
    );
    let mut encoded = vec![0; bytes];
    input.download(&mut encoded)?;
    Ok(encoded
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

pub(in crate::kernels::cuda::resident_native_mtp_forward) fn first_argmax(
    logits: &[u16],
) -> Result<usize> {
    ensure!(
        logits.len() == PROPOSAL_ROWS,
        "native proposal logits row count mismatch"
    );
    let mut winner = None;
    let mut best = f32::NEG_INFINITY;
    for (row, &bits) in logits.iter().enumerate() {
        let score = f32::from_bits(u32::from(bits) << 16);
        ensure!(
            score.is_finite(),
            "native proposal logits contain nonfinite values"
        );
        if score > best {
            best = score;
            winner = Some(row);
        }
    }
    winner.context("native proposal logits are empty")
}

#[cfg(test)]
mod tests {
    use super::first_argmax;

    #[test]
    fn shortlist_argmax_keeps_the_first_tied_row() {
        let mut logits = vec![0x0000_u16; 131_072];
        logits[3] = 0x3f80;
        logits[19] = 0x3f80;

        let winner = first_argmax(&logits).expect("finite proposal logits");

        assert_eq!(winner, 3);
    }

    #[test]
    fn shortlist_argmax_rejects_nonfinite_logits() {
        let mut logits = vec![0x0000_u16; 131_072];
        logits[9] = 0x7f80;

        let result = first_argmax(&logits);

        assert!(result.is_err());
    }
}
