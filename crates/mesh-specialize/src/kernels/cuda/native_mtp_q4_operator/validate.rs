use crate::packages::qwen3_8_27b::native_mtp_views::Q4MatrixView;
use anyhow::{Context, ensure};
use std::collections::BTreeSet;

const GROUP_SIZE: usize = 64;
const ROW_ALIGNMENT: usize = 128;
const PLANE_ALIGNMENT: usize = 256;
const MAX_SELECTED_ROWS: usize = 131_072;
const MAX_LOGICAL_K: usize = 32_768;

pub(super) struct ValidatedView {
    pub(super) selected_rows: u32,
    pub(super) logical_k: u32,
    pub(super) padded_k: u32,
    pub(super) scale_offset: u32,
    pub(super) source_rows: Vec<u32>,
}

pub(super) fn validate(
    object_bytes: &[u8],
    view: &Q4MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<ValidatedView> {
    let [selected_rows, logical_k] = view.shape;
    ensure!(
        (1..=MAX_SELECTED_ROWS).contains(&selected_rows)
            && (1..=MAX_LOGICAL_K).contains(&logical_k),
        "native Q4 proposal-head shape is outside its bounded domain"
    );
    ensure!(
        view.group_size == GROUP_SIZE,
        "native Q4 group width must be 64"
    );
    ensure!(
        view.source_rows.len() == selected_rows,
        "native Q4 row-map extent differs from N"
    );
    ensure!(
        input_bf16.len() == logical_k,
        "native Q4 activation width differs from K"
    );
    ensure!(
        input_bf16
            .iter()
            .all(|bits| f32::from_bits(u32::from(*bits) << 16).is_finite()),
        "native Q4 activation contains a nonfinite value"
    );
    let padded_k = logical_k
        .checked_add(ROW_ALIGNMENT - 1)
        .context("native Q4 padded K overflows")?
        / ROW_ALIGNMENT
        * ROW_ALIGNMENT;
    ensure!(
        view.padded_k == padded_k,
        "native Q4 padded K is inconsistent"
    );
    ensure!(
        view.codes.offset == 0,
        "native Q4 code plane must start at offset zero"
    );
    let code_row_bytes = padded_k / 2;
    let code_bytes =
        usize::try_from(view.codes.bytes).context("native Q4 code plane is too large")?;
    ensure!(
        code_bytes > 0 && code_bytes.is_multiple_of(code_row_bytes),
        "native Q4 code plane has incomplete rows"
    );
    let parent_rows = code_bytes / code_row_bytes;
    let groups_per_row = padded_k / GROUP_SIZE;
    let scale_count = parent_rows
        .checked_mul(groups_per_row)
        .context("native Q4 scale count overflows")?;
    ensure!(
        view.scale_count == scale_count,
        "native Q4 scale count differs from parent geometry"
    );
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("native Q4 scale plane overflows")?;
    ensure!(
        usize::try_from(view.scale_bits.bytes)? == scale_bytes,
        "native Q4 scale byte count is inconsistent"
    );
    let scale_offset = code_bytes
        .checked_add((PLANE_ALIGNMENT - code_bytes % PLANE_ALIGNMENT) % PLANE_ALIGNMENT)
        .context("native Q4 scale offset overflows")?;
    ensure!(
        usize::try_from(view.scale_bits.offset)? == scale_offset,
        "native Q4 scale plane is not 256-byte aligned"
    );
    let object_end = scale_offset
        .checked_add(scale_bytes)
        .context("native Q4 object extent overflows")?;
    ensure!(
        object_bytes.len() == object_end,
        "native Q4 object extent differs from its planes"
    );
    ensure!(
        object_bytes
            .get(code_bytes..scale_offset)
            .context("native Q4 plane padding is absent")?
            .iter()
            .all(|&byte| byte == 0),
        "native Q4 plane alignment padding must be zero"
    );
    ensure!(
        view.source_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == selected_rows,
        "native Q4 row map contains duplicates"
    );
    if u32::try_from(selected_rows).is_err()
        || u32::try_from(logical_k).is_err()
        || u32::try_from(padded_k).is_err()
        || u32::try_from(scale_offset).is_err()
    {
        anyhow::bail!("native Q4 geometry exceeds the kernel ABI");
    }
    for &parent_row in &view.source_rows {
        ensure!(
            parent_row < parent_rows,
            "native Q4 row map exceeds parent extent"
        );
        u32::try_from(parent_row).context("native Q4 parent row exceeds u32")?;
        let row_code_start = parent_row * code_row_bytes;
        let row_scale_start = scale_offset + parent_row * groups_per_row * 2;
        for group in 0..groups_per_row {
            let scale_index = row_scale_start + group * 2;
            let bits = read_scale(object_bytes, scale_index)?;
            let scale = decode_scale(bits)?;
            let group_start = group * GROUP_SIZE;
            ensure!(
                group_start < logical_k || bits == 0,
                "native Q4 padding scale must be zero"
            );
            validate_group_codes(object_bytes, row_code_start, group_start, logical_k, scale)?;
        }
    }
    Ok(ValidatedView {
        selected_rows: u32::try_from(selected_rows).context("native Q4 N exceeds u32")?,
        logical_k: u32::try_from(logical_k).context("native Q4 K exceeds u32")?,
        padded_k: u32::try_from(padded_k).context("native Q4 padded K exceeds u32")?,
        scale_offset: u32::try_from(scale_offset).context("native Q4 scale offset exceeds u32")?,
        source_rows: view
            .source_rows
            .iter()
            .map(|&row| u32::try_from(row).context("native Q4 parent row exceeds u32"))
            .collect::<anyhow::Result<Vec<_>>>()?,
    })
}

fn validate_group_codes(
    object_bytes: &[u8],
    row_code_start: usize,
    group_start: usize,
    logical_k: usize,
    scale: f32,
) -> anyhow::Result<()> {
    for lane in 0..GROUP_SIZE {
        let k = group_start + lane;
        let packed = *object_bytes
            .get(row_code_start + k / 2)
            .context("native Q4 code is absent")?;
        let nibble = if k.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        let code = i32::from(nibble & 0x07) - i32::from(nibble & 0x08);
        ensure!(
            k < logical_k || code == 0,
            "native Q4 logical-tail code must be zero"
        );
        ensure!(
            scale != 0.0 || code == 0,
            "zero native Q4 scale requires zero codes"
        );
    }
    Ok(())
}

fn read_scale(bytes: &[u8], offset: usize) -> anyhow::Result<u16> {
    let end = offset
        .checked_add(2)
        .context("native Q4 scale offset overflows")?;
    let pair = bytes
        .get(offset..end)
        .context("native Q4 scale is absent")?;
    Ok(u16::from_le_bytes(
        pair.try_into()
            .context("native Q4 scale is not two bytes")?,
    ))
}

fn decode_scale(bits: u16) -> anyhow::Result<f32> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    ensure!(
        sign == 0 && exponent != 0x1f,
        "native Q4 scale is negative or nonfinite"
    );
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}
