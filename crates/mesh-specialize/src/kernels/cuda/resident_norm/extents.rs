use anyhow::{Context as _, Result, ensure};

pub(super) const MAX_ELEMENTS: usize = 67_108_864;

pub(super) fn matrix_extents(rows: usize, width: usize) -> Result<(usize, usize, usize)> {
    ensure!(
        (1..=2048).contains(&rows),
        "resident norm row count is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "resident norm width is out of range"
    );
    let count = rows
        .checked_mul(width)
        .context("resident norm element count overflows usize")?;
    ensure!(
        count <= MAX_ELEMENTS,
        "resident norm element count is too large"
    );
    let bf16_bytes = count
        .checked_mul(2)
        .context("resident norm BF16 extent overflows usize")?;
    let fp32_bytes = count
        .checked_mul(4)
        .context("resident norm FP32 extent overflows usize")?;
    Ok((count, bf16_bytes, fp32_bytes))
}

pub(super) fn validate_matrix_buffer(
    input_bytes: usize,
    rows: usize,
    width: usize,
) -> Result<(usize, usize, usize)> {
    let extents = matrix_extents(rows, width)?;
    ensure!(
        input_bytes == extents.1,
        "resident norm input extent mismatch"
    );
    Ok(extents)
}

pub(super) fn validate_matrix_pair(
    residual_bytes: usize,
    branch_bytes: usize,
    rows: usize,
    width: usize,
) -> Result<(usize, usize, usize)> {
    let extents = matrix_extents(rows, width)?;
    ensure!(
        residual_bytes == extents.1 && branch_bytes == extents.1,
        "resident norm input extent mismatch"
    );
    Ok(extents)
}

pub(super) fn validate_add_extents(
    left_bytes: usize,
    right_bytes: usize,
) -> Result<(usize, usize)> {
    ensure!(
        left_bytes == right_bytes && left_bytes.is_multiple_of(2),
        "residual add inputs must have the same even byte extent"
    );
    let count = left_bytes / 2;
    ensure!(
        (1..=MAX_ELEMENTS).contains(&count),
        "residual add element count is out of range"
    );
    Ok((count, left_bytes))
}

pub(super) fn row_ids(rows: usize) -> Result<Vec<u8>> {
    ensure!(
        (1..=2048).contains(&rows),
        "resident norm row count is out of range"
    );
    let byte_count = rows
        .checked_mul(4)
        .context("resident row ID byte extent overflows usize")?;
    let mut bytes = Vec::with_capacity(byte_count);
    for row in 0..rows {
        bytes.extend_from_slice(&u32::try_from(row)?.to_le_bytes());
    }
    Ok(bytes)
}
