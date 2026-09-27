//! Resident last-row normalization and vocabulary projection for next-token logits.

use super::{
    driver::{Buffer, Context, Module},
    resident_fp8,
    resident_norm::Norm,
    resident_weights::ResidentWeights,
};
use anyhow::{Result, ensure};

pub(super) struct Head<'w, 'ctx> {
    norm: Norm<'w, 'ctx>,
    projection: resident_fp8::Projection<'w, 'ctx>,
    width: usize,
}

impl<'w, 'ctx> Head<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        norm_name: &str,
        projection_prefix: &str,
        width: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        validate_shape(width, vocabulary)?;
        let norm = Norm::new(owner, norm_name, width, 1e-6)?;
        let projection =
            resident_fp8::Projection::new(owner, projection_prefix, width, vocabulary)?;
        Ok(Self {
            norm,
            projection,
            width,
        })
    }

    /// Select the final hidden row on device, normalize it, and produce vocabulary logits.
    /// Earlier rows still pass through the decoder; only the final row predicts the next token.
    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        rows: usize,
    ) -> Result<resident_fp8::Output<'a>> {
        ensure!(
            module.belongs_to(context),
            "LM head module belongs to another context"
        );
        ensure!(
            hidden.belongs_to(context),
            "LM head hidden rows belong to another context"
        );
        let (source_offset, row_bytes) = last_row_extents(rows, self.width, hidden.len())?;
        let final_hidden = Buffer::new(context, row_bytes)?;
        final_hidden.copy_from_at(0, hidden, source_offset, row_bytes)?;
        let normalized = self.norm.run(context, module, &final_hidden, 1)?;
        self.projection.run(context, module, &normalized, 1)
    }
}

fn validate_shape(width: usize, vocabulary: usize) -> Result<()> {
    ensure!(
        (1..=32768).contains(&width),
        "LM head width is out of range"
    );
    ensure!(
        (1..=262144).contains(&vocabulary),
        "LM head vocabulary is out of range"
    );
    Ok(())
}

fn last_row_extents(rows: usize, width: usize, input_bytes: usize) -> Result<(usize, usize)> {
    ensure!(
        (1..=2048).contains(&rows),
        "LM head row count is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "LM head width is out of range"
    );
    let row_bytes = checked_product(width, 2, "LM head BF16 row")?;
    let expected_bytes = checked_product(rows, row_bytes, "LM head BF16 input")?;
    ensure!(
        input_bytes == expected_bytes,
        "LM head input extent mismatch"
    );
    let source_offset = checked_product(rows - 1, row_bytes, "LM head final row offset")?;
    Ok((source_offset, row_bytes))
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("{label} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{checked_product, last_row_extents, validate_shape};

    #[test]
    fn computes_exact_final_row_offsets() {
        assert_eq!(last_row_extents(1, 1, 2).unwrap(), (0, 2));
        assert_eq!(last_row_extents(17, 4, 136).unwrap(), (128, 8));
        assert_eq!(
            last_row_extents(2048, 32768, 134_217_728).unwrap(),
            (134_152_192, 65_536)
        );
    }

    #[test]
    fn rejects_invalid_dimensions_and_hidden_extents() {
        for (width, vocabulary) in [(0, 1), (32769, 1), (1, 0), (1, 262145)] {
            assert!(validate_shape(width, vocabulary).is_err());
        }
        assert!(last_row_extents(0, 1, 2).is_err());
        assert!(last_row_extents(2049, 1, 4098).is_err());
        assert!(last_row_extents(1, 0, 0).is_err());
        assert!(last_row_extents(1, 1, 4).is_err());
    }

    #[test]
    fn rejects_extent_arithmetic_overflow() {
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }
}
