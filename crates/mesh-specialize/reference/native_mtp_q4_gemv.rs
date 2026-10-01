use crate::packages::qwen3_8_27b::native_mtp_views::Q4MatrixView;
use anyhow::{Context, ensure};
use std::collections::BTreeSet;

const GROUP_SIZE: usize = 64;
const ROW_ALIGNMENT: usize = 128;
const PLANE_ALIGNMENT: usize = 256;
const MAX_SELECTED_ROWS: usize = 131_072;
const MAX_LOGICAL_K: usize = 32_768;

pub struct Q4ProjectionReference {
    pub raw_f64: Vec<f64>,
    pub logits_bf16: Vec<u16>,
}

/// Computes selected packed-Q4 rows with an independent FP64 accumulation oracle.
///
/// # Errors
/// Returns an error when the indexed matrix, physical planes, scales, or BF16
/// activation violates the `row_split_k128_v1` Q4 contract.
pub fn run(
    object_bytes: &[u8],
    view: &Q4MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<Q4ProjectionReference> {
    let (scale_offset, parent_rows) = validate(object_bytes, view, input_bf16)?;
    let mut raw_f64 = Vec::with_capacity(view.shape[0]);
    let mut logits_bf16 = Vec::with_capacity(view.shape[0]);
    for &parent_row in &view.source_rows {
        let row_code_start = parent_row * (view.padded_k / 2);
        let row_scale_start = scale_offset + parent_row * (view.padded_k / GROUP_SIZE) * 2;
        let mut sum_f64 = 0.0_f64;
        let mut sum_f32 = 0.0_f32;
        for (k, &activation_bits) in input_bf16.iter().enumerate() {
            let packed = *object_bytes
                .get(row_code_start + k / 2)
                .context("Q4 code is absent")?;
            let nibble = if k.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let code = i8::try_from(i32::from(nibble & 0x07) - i32::from(nibble & 0x08))
                .context("Q4 nibble must decode to a signed four-bit value")?;
            let scale_index = row_scale_start + (k / GROUP_SIZE) * 2;
            let scale = decode_scale(read_scale(object_bytes, scale_index)?)?;
            let weight = scale * f32::from(code);
            let activation = f32::from_bits(u32::from(activation_bits) << 16);
            sum_f64 += f64::from(weight) * f64::from(activation);
            sum_f32 = weight.mul_add(activation, sum_f32);
        }
        raw_f64.push(sum_f64);
        logits_bf16.push(crate::entry_reference::round_bf16(sum_f32));
    }
    ensure!(parent_rows > 0, "Q4 matrix has no parent rows");
    Ok(Q4ProjectionReference {
        raw_f64,
        logits_bf16,
    })
}

fn validate(
    object_bytes: &[u8],
    view: &Q4MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<(usize, usize)> {
    let [selected_rows, logical_k] = view.shape;
    ensure!(
        (1..=MAX_SELECTED_ROWS).contains(&selected_rows)
            && (1..=MAX_LOGICAL_K).contains(&logical_k),
        "Q4 proposal head shape is outside its bounded domain"
    );
    ensure!(view.group_size == GROUP_SIZE, "Q4 group width must be 64");
    ensure!(
        view.source_rows.len() == selected_rows,
        "Q4 selected row count differs from N"
    );
    ensure!(
        input_bf16.len() == logical_k,
        "BF16 activation width differs from K"
    );
    ensure!(
        input_bf16
            .iter()
            .all(|bits| f32::from_bits(u32::from(*bits) << 16).is_finite()),
        "BF16 activation contains a nonfinite value"
    );
    let padded_k = logical_k
        .checked_add(ROW_ALIGNMENT - 1)
        .context("Q4 padded K overflows")?
        / ROW_ALIGNMENT
        * ROW_ALIGNMENT;
    ensure!(view.padded_k == padded_k, "Q4 padded K is inconsistent");
    ensure!(
        view.codes.offset == 0,
        "Q4 code plane must start at object offset zero"
    );
    let code_row_bytes = padded_k / 2;
    let code_bytes = usize::try_from(view.codes.bytes).context("Q4 code plane is too large")?;
    ensure!(
        code_bytes > 0 && code_bytes.is_multiple_of(code_row_bytes),
        "Q4 code plane does not contain complete rows"
    );
    let parent_rows = code_bytes / code_row_bytes;
    let groups_per_row = padded_k / GROUP_SIZE;
    let scale_count = parent_rows
        .checked_mul(groups_per_row)
        .context("Q4 scale count overflows")?;
    ensure!(
        view.scale_count == scale_count,
        "Q4 scale count differs from parent geometry"
    );
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("Q4 scale plane overflows")?;
    ensure!(
        usize::try_from(view.scale_bits.bytes)? == scale_bytes,
        "Q4 scale plane byte count is inconsistent"
    );
    let scale_offset = code_bytes
        .checked_add((PLANE_ALIGNMENT - code_bytes % PLANE_ALIGNMENT) % PLANE_ALIGNMENT)
        .context("Q4 aligned scale offset overflows")?;
    ensure!(
        usize::try_from(view.scale_bits.offset)? == scale_offset,
        "Q4 scale plane is not 256-byte aligned after codes"
    );
    let object_end = scale_offset
        .checked_add(scale_bytes)
        .context("Q4 object extent overflows")?;
    ensure!(
        object_bytes.len() == object_end,
        "Q4 object extent differs from its planes"
    );
    ensure!(
        object_bytes
            .get(code_bytes..scale_offset)
            .context("Q4 alignment padding is absent")?
            .iter()
            .all(|&byte| byte == 0),
        "Q4 code/scale alignment padding must be zero"
    );
    ensure!(
        view.source_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == selected_rows,
        "Q4 parent row map contains duplicates"
    );
    for &parent_row in &view.source_rows {
        ensure!(
            parent_row < parent_rows,
            "Q4 parent row exceeds the packed code plane"
        );
        let row_code_start = parent_row * code_row_bytes;
        let row_scale_start = scale_offset + parent_row * groups_per_row * 2;
        for group in 0..groups_per_row {
            let bits = read_scale(object_bytes, row_scale_start + group * 2)?;
            let scale = decode_scale(bits)?;
            let group_start = group * GROUP_SIZE;
            ensure!(
                group_start < logical_k || bits == 0,
                "Q4 padding group scale must be zero"
            );
            validate_group_codes(object_bytes, row_code_start, group_start, logical_k, scale)?;
        }
    }
    Ok((scale_offset, parent_rows))
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
            .context("Q4 code is absent")?;
        let nibble = if k.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        let code = i32::from(nibble & 0x07) - i32::from(nibble & 0x08);
        ensure!(
            k < logical_k || code == 0,
            "Q4 logical-tail code must be zero"
        );
        ensure!(
            scale != 0.0 || code == 0,
            "zero Q4 scale requires zero codes"
        );
    }
    Ok(())
}

fn read_scale(bytes: &[u8], offset: usize) -> anyhow::Result<u16> {
    let end = offset.checked_add(2).context("Q4 scale offset overflows")?;
    let pair = bytes.get(offset..end).context("FP16 Q4 scale is absent")?;
    Ok(u16::from_le_bytes(
        pair.try_into()
            .context("FP16 Q4 scale does not contain two bytes")?,
    ))
}

fn decode_scale(bits: u16) -> anyhow::Result<f32> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    ensure!(
        sign == 0 && exponent != 0x1f,
        "FP16 Q4 scale is negative or nonfinite"
    );
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView};

    #[test]
    fn q4_oracle_decodes_signed_nibbles_group_scales_and_reordered_parent_rows() {
        let view = Q4MatrixView {
            object_id: "q4-oracle".into(),
            shape: [2, 128],
            padded_k: 128,
            group_size: 64,
            codes: BytePlane {
                offset: 0,
                bytes: 192,
            },
            scale_bits: BytePlane {
                offset: 256,
                bytes: 12,
            },
            scale_count: 6,
            source_rows: vec![2, 0],
        };
        let mut object = vec![0; 268];
        object[128] = 0xf8;
        object[128 + 32] = 0x03;
        object[0] = 0x17;
        for row in 0..3 {
            let scale_offset = 256 + row * 4;
            object[scale_offset..scale_offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            object[scale_offset + 2..scale_offset + 4].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }
        object[264..266].copy_from_slice(&0x0001_u16.to_le_bytes());
        let mut input = vec![0x0000; 128];
        input[0] = 0x3f80;
        input[1] = 0xbf80;
        input[64] = 0x3f80;

        let output = run(&object, &view, &input).expect("valid indexed Q4 rows");

        assert_eq!(output.raw_f64, [3.0 - 7.0 * 2.0_f64.powi(-24), 6.0]);
        assert_eq!(output.logits_bf16, [0x4040, 0x40c0]);
    }
}
