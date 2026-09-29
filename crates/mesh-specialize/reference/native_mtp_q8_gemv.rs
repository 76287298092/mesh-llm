use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context, ensure};
use std::collections::BTreeSet;

/// Computes selected signed-Q8 rows with a direct scalar FP64 accumulation oracle.
///
/// # Errors
/// Returns an error when the matrix view, packed object, activation, or selected
/// row data violates the `row_split_k128_v1` Q8 contract.
pub fn run(object_bytes: &[u8], view: &Q8MatrixView, input_bf16: &[u16]) -> anyhow::Result<Vec<f64>> {
    let (scale_offset, parent_rows) = validate(object_bytes, view, input_bf16)?;
    let mut output = Vec::with_capacity(view.shape[0]);
    for &parent_row in &view.source_rows {
        let mut sum = 0.0_f64;
        for k in 0..view.shape[1] {
            let group = k / 32;
            let scale_index = scale_offset + (parent_row * (view.padded_k / 32) + group) * 2;
            let scale_bits = read_u16(object_bytes, scale_index)?;
            let scale = decode_scale(scale_bits)?;
            let code_index = parent_row * view.padded_k + k;
            let code = i8::from_ne_bytes([*object_bytes.get(code_index).context("Q8 code is absent")?]);
            let weight = scale * f32::from(code);
            let activation_bits = *input_bf16.get(k).context("BF16 activation is absent")?;
            let activation = f32::from_bits(u32::from(activation_bits) << 16);
            sum += f64::from(weight) * f64::from(activation);
        }
        output.push(sum);
    }
    ensure!(parent_rows > 0, "Q8 matrix has no parent rows");
    Ok(output)
}

fn validate(
    object_bytes: &[u8],
    view: &Q8MatrixView,
    input_bf16: &[u16],
) -> anyhow::Result<(usize, usize)> {
    let [selected_rows, logical_k] = view.shape;
    ensure!(selected_rows > 0 && logical_k > 0, "Q8 matrix shape must be nonzero");
    ensure!(view.group_size == 32, "Q8 group width must be 32");
    ensure!(input_bf16.len() == logical_k, "BF16 activation width differs from K");
    ensure!(view.source_rows.len() == selected_rows, "Q8 selected row count differs from N");
    let padded_k = logical_k
        .checked_add(127)
        .context("Q8 padded K overflows")?
        / 128
        * 128;
    ensure!(view.padded_k == padded_k, "Q8 padded K is inconsistent");
    ensure!(view.codes.offset == 0, "Q8 code plane must start at object offset zero");
    let code_bytes = usize::try_from(view.codes.bytes).context("Q8 code plane is too large")?;
    ensure!(code_bytes > 0 && code_bytes.is_multiple_of(padded_k), "Q8 code plane does not contain complete rows");
    let parent_rows = code_bytes / padded_k;
    let scale_count = parent_rows
        .checked_mul(padded_k / 32)
        .context("Q8 scale count overflows")?;
    ensure!(view.scale_count == scale_count, "Q8 scale count differs from parent geometry");
    let scale_bytes = scale_count.checked_mul(2).context("Q8 scale plane overflows")?;
    ensure!(usize::try_from(view.scale_bits.bytes)? == scale_bytes, "Q8 scale plane byte count is inconsistent");
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("Q8 aligned scale offset overflows")?;
    ensure!(usize::try_from(view.scale_bits.offset)? == scale_offset, "Q8 scale plane is not 256-byte aligned after codes");
    let object_end = scale_offset.checked_add(scale_bytes).context("Q8 object extent overflows")?;
    ensure!(object_bytes.len() == object_end, "Q8 object extent differs from its planes");
    ensure!(
        object_bytes.get(code_bytes..scale_offset).context("Q8 alignment padding is absent")?.iter().all(|&byte| byte == 0),
        "Q8 code/scale alignment padding must be zero"
    );
    ensure!(
        view.source_rows.iter().copied().collect::<BTreeSet<_>>().len() == selected_rows,
        "Q8 parent row map contains duplicates"
    );
    ensure!(input_bf16.iter().all(|bits| f32::from_bits(u32::from(*bits) << 16).is_finite()), "BF16 activation contains a nonfinite value");
    for &parent_row in &view.source_rows {
        ensure!(parent_row < parent_rows, "Q8 parent row exceeds the packed code plane");
        for group in 0..(padded_k / 32) {
            let scale_index = scale_offset + (parent_row * (padded_k / 32) + group) * 2;
            let scale_bits = read_u16(object_bytes, scale_index)?;
            let scale = decode_scale(scale_bits)?;
            let group_start = group * 32;
            ensure!(group_start < logical_k || scale_bits == 0, "Q8 padding group scale must be zero");
            for lane in 0..32 {
                let k = group_start + lane;
                let index = parent_row * padded_k + k;
                let code = i8::from_ne_bytes([*object_bytes.get(index).context("Q8 code is absent")?]);
                ensure!(code != i8::MIN, "Q8 code -128 is invalid");
                ensure!(k < logical_k || code == 0, "Q8 logical-tail code must be zero");
                ensure!(scale != 0.0 || code == 0, "zero Q8 scale requires zero codes");
            }
        }
    }
    Ok((scale_offset, parent_rows))
}

fn read_u16(bytes: &[u8], offset: usize) -> anyhow::Result<u16> {
    let words = bytes.get(offset..offset + 2).context("FP16 scale is absent")?;
    Ok(u16::from_le_bytes(
        words
            .try_into()
            .context("FP16 scale does not contain two bytes")?,
    ))
}

fn decode_scale(bits: u16) -> anyhow::Result<f32> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    ensure!(sign == 0 && exponent != 0x1f, "FP16 Q8 scale is negative or nonfinite");
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView};

    fn fixture() -> (Vec<u8>, Q8MatrixView, Vec<u16>) {
        let view = Q8MatrixView {
            object_id: "fixture".into(),
            shape: [2, 160],
            padded_k: 256,
            group_size: 32,
            codes: BytePlane { offset: 0, bytes: 768 },
            scale_bits: BytePlane { offset: 768, bytes: 48 },
            scale_count: 24,
            source_rows: vec![2, 0],
        };
        let mut object = vec![0; 816];
        for k in 0..160 {
            object[512 + k] = 0;
            object[k] = 0;
        }
        for (k, code) in [(0, -1_i8), (32, 2), (127, 3), (128, -2), (159, 1)] {
            object[512 + k] = code.to_ne_bytes()[0];
        }
        object[0] = 1;
        object[32] = 1;
        for group in 0..8 {
            let row_two: u16 = match group {
                0 => 0x3c00,
                1 => 0x4000,
                2 => 0x3800,
                3 => 0x3c00,
                4 => 0x3800,
                _ => 0,
            };
            let row_zero: u16 = if group < 2 { 0x3800 } else { 0 };
            object[768 + (16 + group) * 2..768 + (17 + group) * 2]
                .copy_from_slice(&row_two.to_le_bytes());
            object[768 + group * 2..768 + (group + 1) * 2]
                .copy_from_slice(&row_zero.to_le_bytes());
        }
        (object, view, vec![0x3f80; 160])
    }

    #[test]
    fn q8_gemv_respects_parent_rows_and_k128_split() {
        let (object, view, input) = fixture();
        let output = run(&object, &view, &input).expect("valid row-split Q8 matrix");
        assert_eq!(output, [5.5, 1.0]);
    }

    #[test]
    fn q8_gemv_rejects_invalid_signed_code() {
        let (mut object, view, input) = fixture();
        object[512] = 0x80;
        assert!(run(&object, &view, &input).is_err());
    }
}
