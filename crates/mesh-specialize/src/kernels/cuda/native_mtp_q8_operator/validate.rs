use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context, ensure};
use std::collections::BTreeSet;

pub(super) struct ValidatedView {
    pub(super) selected_rows: u32,
    pub(super) logical_k: u32,
    pub(super) padded_k: u32,
    pub(super) scale_offset: u32,
    pub(super) source_rows: Vec<u32>,
}

pub(super) fn validate(
    object_bytes: &[u8],
    view: &Q8MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<ValidatedView> {
    let [selected_rows, logical_k] = view.shape;
    ensure!(selected_rows > 0 && logical_k > 0, "native Q8 matrix shape must be nonzero");
    ensure!(view.group_size == 32, "native Q8 group width must be 32");
    ensure!(view.source_rows.len() == selected_rows, "native Q8 row map extent differs from N");
    ensure!(input_bf16.len() == logical_k, "native Q8 activation width differs from K");
    let padded_k = logical_k
        .checked_add(127)
        .context("native Q8 padded K overflows")?
        / 128
        * 128;
    ensure!(view.padded_k == padded_k, "native Q8 padded K is inconsistent");
    ensure!(view.codes.offset == 0, "native Q8 code plane must start at offset zero");
    let code_bytes = usize::try_from(view.codes.bytes).context("native Q8 code plane is too large")?;
    ensure!(code_bytes > 0 && code_bytes.is_multiple_of(padded_k), "native Q8 code plane has incomplete rows");
    let parent_rows = code_bytes / padded_k;
    let scale_count = parent_rows
        .checked_mul(padded_k / 32)
        .context("native Q8 scale count overflows")?;
    ensure!(view.scale_count == scale_count, "native Q8 scale count differs from parent geometry");
    let scale_bytes = scale_count.checked_mul(2).context("native Q8 scale plane overflows")?;
    ensure!(usize::try_from(view.scale_bits.bytes)? == scale_bytes, "native Q8 scale byte count is inconsistent");
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("native Q8 scale offset overflows")?;
    ensure!(usize::try_from(view.scale_bits.offset)? == scale_offset, "native Q8 scale plane is not 256-byte aligned");
    let object_end = scale_offset.checked_add(scale_bytes).context("native Q8 object extent overflows")?;
    ensure!(object_bytes.len() == object_end, "native Q8 object extent differs from its planes");
    ensure!(
        object_bytes.get(code_bytes..scale_offset).context("native Q8 plane padding is absent")?.iter().all(|&byte| byte == 0),
        "native Q8 plane alignment padding must be zero"
    );
    ensure!(
        view.source_rows.iter().copied().collect::<BTreeSet<_>>().len() == selected_rows,
        "native Q8 row map contains duplicates"
    );
    ensure!(
        input_bf16.iter().all(|bits| f32::from_bits(u32::from(*bits) << 16).is_finite()),
        "native Q8 activation contains a nonfinite value"
    );
    for &parent_row in &view.source_rows {
        ensure!(parent_row < parent_rows, "native Q8 row map exceeds parent extent");
        for group in 0..(padded_k / 32) {
            let scale_index = scale_offset + (parent_row * (padded_k / 32) + group) * 2;
            let bits = read_scale(object_bytes, scale_index)?;
            let scale = decode_scale(bits)?;
            let group_start = group * 32;
            ensure!(group_start < logical_k || bits == 0, "native Q8 padding scale must be zero");
            for lane in 0..32 {
                let k = group_start + lane;
                let code_index = parent_row * padded_k + k;
                let code = i8::from_ne_bytes([*object_bytes.get(code_index).context("native Q8 code is absent")?]);
                ensure!(code != i8::MIN, "native Q8 code -128 is invalid");
                ensure!(k < logical_k || code == 0, "native Q8 logical-tail code must be zero");
                ensure!(scale != 0.0 || code == 0, "zero native Q8 scale requires zero codes");
            }
        }
    }
    Ok(ValidatedView {
        selected_rows: u32::try_from(selected_rows).context("native Q8 N exceeds u32")?,
        logical_k: u32::try_from(logical_k).context("native Q8 K exceeds u32")?,
        padded_k: u32::try_from(padded_k).context("native Q8 padded K exceeds u32")?,
        scale_offset: u32::try_from(scale_offset).context("native Q8 scale offset exceeds u32")?,
        source_rows: view
            .source_rows
            .iter()
            .map(|&row| u32::try_from(row).context("native Q8 parent row exceeds u32"))
            .collect::<anyhow::Result<Vec<_>>>()?,
    })
}

fn read_scale(bytes: &[u8], offset: usize) -> anyhow::Result<u16> {
    let pair = bytes.get(offset..offset + 2).context("native Q8 scale is absent")?;
    Ok(u16::from_le_bytes(
        pair.try_into().context("native Q8 scale is not two bytes")?,
    ))
}

fn decode_scale(bits: u16) -> anyhow::Result<f32> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    ensure!(sign == 0 && exponent != 0x1f, "native Q8 scale is negative or nonfinite");
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}
