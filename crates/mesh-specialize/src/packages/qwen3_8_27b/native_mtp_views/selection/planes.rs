use super::super::Q8MatrixView;
use super::binding::{MatrixSelection, selected_rows};
use super::plane_layout::packed_planes;
use crate::artifact::ninfer::{Directory, Object};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

pub(super) fn q8_projection(
    directory: &Directory,
    binding: &str,
    shape: [usize; 2],
) -> Result<Q8MatrixView> {
    let selected = selected_rows(directory, MatrixSelection { binding, shape })?;
    q8_view(selected.object, shape, selected.rows.collect())
}

pub(super) fn q8_view(
    object: &Object,
    shape: [usize; 2],
    source_rows: Vec<usize>,
) -> Result<Q8MatrixView> {
    ensure!(
        object.layout.as_deref() == Some("row_split_k128_v1"),
        "unsupported native MTP layout"
    );
    let (padded_k, group_size, codes, scale_bits, scale_count) = packed_planes(object)?;
    ensure!(
        object.format.as_deref() == Some("q8_g32_fp16") && group_size == 32,
        "unsupported native MTP quantization; expected q8_g32_fp16"
    );
    ensure!(
        u64::try_from(scale_count)?
            .checked_mul(2)
            .is_some_and(|expected_bytes| expected_bytes == scale_bits.bytes),
        "native Q8 FP16 scale byte count mismatch"
    );
    ensure!(
        source_rows.len() == shape[0],
        "native Q8 row map extent mismatch"
    );
    ensure!(
        source_rows.iter().copied().collect::<BTreeSet<_>>().len() == source_rows.len(),
        "native Q8 row map repeats parent rows"
    );
    let parent_rows = usize::try_from(
        *object
            .shape
            .first()
            .context("native Q8 parent N is missing")?,
    )?;
    ensure!(
        source_rows.iter().all(|&row| row < parent_rows),
        "native Q8 row map exceeds parent"
    );
    Ok(Q8MatrixView {
        object_id: object.id.clone(),
        shape,
        padded_k,
        group_size,
        codes,
        scale_bits,
        scale_count,
        source_rows,
    })
}
