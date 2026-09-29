use super::{
    error::NativeMtpDecodeError,
    row::{PackedRow, q4_g64_fp16_decode_row, q8_g32_fp16_decode_row},
};
use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView, Q8MatrixView};

#[cfg(test)]
#[path = "view_tests.rs"]
mod tests;

type RowDecoder = for<'a> fn(PackedRow<'a>) -> Result<Vec<f32>, NativeMtpDecodeError>;

struct PackedView<'a> {
    padded_k: usize,
    group_size: usize,
    codes: &'a BytePlane,
    scales: &'a BytePlane,
    scale_count: usize,
    source_rows: &'a [usize],
    shape: [usize; 2],
    selected_row: usize,
    bits_per_code: usize,
    expected_group_size: usize,
    decode: RowDecoder,
}

pub fn decode_q8_view_row(
    object_bytes: &[u8],
    view: &Q8MatrixView,
    selected_row: usize,
) -> Result<Vec<f32>, NativeMtpDecodeError> {
    decode_view_row(
        object_bytes,
        PackedView {
            padded_k: view.padded_k,
            group_size: view.group_size,
            codes: &view.codes,
            scales: &view.scale_bits,
            scale_count: view.scale_count,
            source_rows: &view.source_rows,
            shape: view.shape,
            selected_row,
            bits_per_code: 8,
            expected_group_size: 32,
            decode: q8_g32_fp16_decode_row,
        },
    )
}

pub(super) fn decode_q4_view_row(
    object_bytes: &[u8],
    view: &Q4MatrixView,
    selected_row: usize,
) -> Result<Vec<f32>, NativeMtpDecodeError> {
    decode_view_row(
        object_bytes,
        PackedView {
            padded_k: view.padded_k,
            group_size: view.group_size,
            codes: &view.codes,
            scales: &view.scale_bits,
            scale_count: view.scale_count,
            source_rows: &view.source_rows,
            shape: view.shape,
            selected_row,
            bits_per_code: 4,
            expected_group_size: 64,
            decode: q4_g64_fp16_decode_row,
        },
    )
}

fn decode_view_row(
    object_bytes: &[u8],
    view: PackedView<'_>,
) -> Result<Vec<f32>, NativeMtpDecodeError> {
    let PackedView {
        padded_k,
        group_size,
        codes,
        scales,
        scale_count,
        source_rows,
        shape,
        selected_row,
        bits_per_code,
        expected_group_size,
        decode,
    } = view;
    let parent_row = validate_view_row(
        object_bytes,
        padded_k,
        group_size,
        codes,
        scales,
        scale_count,
        source_rows,
        shape,
        selected_row,
        bits_per_code,
        expected_group_size,
    )?;
    let (code_row, scale_row) = row_slices(
        object_bytes,
        padded_k,
        group_size,
        codes,
        scales,
        parent_row,
        bits_per_code,
    )?;
    decode(PackedRow {
        codes: code_row,
        scale_bytes: scale_row,
        logical_k: shape[1],
        padded_k,
    })
    .map_err(|error| shift_row_error(error, parent_row, padded_k, group_size))
}

#[expect(
    clippy::too_many_arguments,
    reason = "this boundary verifies all physical metadata before slicing an object"
)]
fn validate_view_row(
    object_bytes: &[u8],
    padded_k: usize,
    group_size: usize,
    codes: &BytePlane,
    scales: &BytePlane,
    scale_count: usize,
    source_rows: &[usize],
    shape: [usize; 2],
    selected_row: usize,
    bits_per_code: usize,
    expected_group_size: usize,
) -> Result<usize, NativeMtpDecodeError> {
    if group_size != expected_group_size
        || padded_k == 0
        || !padded_k.is_multiple_of(128)
        || !group_size.is_multiple_of(32)
        || shape[0] == 0
        || shape[1] == 0
        || shape[0] != source_rows.len()
        || shape[1].checked_add(127).map(|k| k / 128 * 128) != Some(padded_k)
    {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "matrix view shape, group width, or padded K is inconsistent",
        ));
    }
    let parent_row =
        *source_rows
            .get(selected_row)
            .ok_or(NativeMtpDecodeError::RowOutOfBounds {
                row: selected_row,
                rows: source_rows.len(),
            })?;
    let code_row_bytes = padded_k
        .checked_mul(bits_per_code)
        .map(|bits| bits / 8)
        .ok_or(NativeMtpDecodeError::InvalidGeometry(
            "code row extent overflows usize",
        ))?;
    let code_bytes = usize::try_from(codes.bytes).map_err(|_| {
        NativeMtpDecodeError::InvalidGeometry("code plane extent does not fit usize")
    })?;
    if code_row_bytes == 0 || !code_bytes.is_multiple_of(code_row_bytes) || codes.offset != 0 {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "code plane does not describe complete rows at object offset zero",
        ));
    }
    let parent_rows = code_bytes / code_row_bytes;
    if parent_row >= parent_rows {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "selected parent row exceeds row-split plane extent",
        ));
    }
    let expected_scale_count = parent_rows.checked_mul(padded_k / group_size).ok_or(
        NativeMtpDecodeError::InvalidGeometry("scale plane group count overflows"),
    )?;
    if scale_count != expected_scale_count {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "scale count differs from row-split physical geometry",
        ));
    }
    let scale_bytes = usize::try_from(scales.bytes).map_err(|_| {
        NativeMtpDecodeError::InvalidGeometry("scale plane extent does not fit usize")
    })?;
    let expected_scale_bytes =
        scale_count
            .checked_mul(2)
            .ok_or(NativeMtpDecodeError::InvalidGeometry(
                "scale plane extent overflows",
            ))?;
    if scale_bytes != expected_scale_bytes {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "scale plane extent differs from row-split physical geometry",
        ));
    }
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .ok_or(NativeMtpDecodeError::InvalidGeometry(
            "aligned scale plane offset overflows",
        ))?;
    if usize::try_from(scales.offset).ok() != Some(scale_offset) {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "row-split scale plane is not at the 256-byte aligned code-plane end",
        ));
    }
    let padding =
        object_bytes
            .get(code_bytes..scale_offset)
            .ok_or(NativeMtpDecodeError::PlaneExtent {
                plane: "alignment padding",
                expected: scale_offset,
                actual: object_bytes.len(),
            })?;
    if let Some((offset, &byte)) = padding.iter().enumerate().find(|(_, byte)| **byte != 0) {
        return Err(NativeMtpDecodeError::NonZeroPlanePadding {
            index: code_bytes + offset,
            byte,
        });
    }
    let object_end =
        scale_offset
            .checked_add(scale_bytes)
            .ok_or(NativeMtpDecodeError::InvalidGeometry(
                "row-split object extent overflows",
            ))?;
    if object_bytes.len() != object_end {
        return Err(NativeMtpDecodeError::PlaneExtent {
            plane: "object",
            expected: object_end,
            actual: object_bytes.len(),
        });
    }
    Ok(parent_row)
}

fn row_slices<'a>(
    object_bytes: &'a [u8],
    padded_k: usize,
    group_size: usize,
    codes: &BytePlane,
    scales: &BytePlane,
    parent_row: usize,
    bits_per_code: usize,
) -> Result<(&'a [u8], &'a [u8]), NativeMtpDecodeError> {
    let code_row_bytes = padded_k * bits_per_code / 8;
    let scale_row_bytes = padded_k / group_size * 2;
    let code_start = parent_row * code_row_bytes;
    let scale_start = usize::try_from(scales.offset)
        .ok()
        .and_then(|offset| offset.checked_add(parent_row.checked_mul(scale_row_bytes)?))
        .ok_or(NativeMtpDecodeError::InvalidGeometry(
            "scale row offset overflows",
        ))?;
    let code_end =
        code_start
            .checked_add(code_row_bytes)
            .ok_or(NativeMtpDecodeError::InvalidGeometry(
                "code row extent overflows",
            ))?;
    let scale_end =
        scale_start
            .checked_add(scale_row_bytes)
            .ok_or(NativeMtpDecodeError::InvalidGeometry(
                "scale row extent overflows",
            ))?;
    let code_row =
        object_bytes
            .get(code_start..code_end)
            .ok_or(NativeMtpDecodeError::PlaneExtent {
                plane: "code",
                expected: code_end,
                actual: object_bytes.len(),
            })?;
    let scale_row =
        object_bytes
            .get(scale_start..scale_end)
            .ok_or(NativeMtpDecodeError::PlaneExtent {
                plane: "scale",
                expected: scale_end,
                actual: object_bytes.len(),
            })?;
    let code_plane_bytes = usize::try_from(codes.bytes).map_err(|_| {
        NativeMtpDecodeError::InvalidGeometry("code plane extent does not fit usize")
    })?;
    if code_end > code_plane_bytes {
        return Err(NativeMtpDecodeError::PlaneExtent {
            plane: "code",
            expected: code_end,
            actual: code_plane_bytes,
        });
    }
    Ok((code_row, scale_row))
}

fn shift_row_error(
    error: NativeMtpDecodeError,
    parent_row: usize,
    padded_k: usize,
    group_size: usize,
) -> NativeMtpDecodeError {
    match error {
        NativeMtpDecodeError::InvalidScale { group, bits } => NativeMtpDecodeError::InvalidScale {
            group: parent_row * (padded_k / group_size) + group,
            bits,
        },
        NativeMtpDecodeError::NonZeroPaddingScale { group, bits } => {
            NativeMtpDecodeError::NonZeroPaddingScale {
                group: parent_row * (padded_k / group_size) + group,
                bits,
            }
        }
        NativeMtpDecodeError::ZeroScaleNonZeroCode { group, index } => {
            NativeMtpDecodeError::ZeroScaleNonZeroCode {
                group: parent_row * (padded_k / group_size) + group,
                index: parent_row * padded_k + index,
            }
        }
        NativeMtpDecodeError::InvalidQ8Code { index, code } => {
            NativeMtpDecodeError::InvalidQ8Code {
                index: parent_row * padded_k + index,
                code,
            }
        }
        NativeMtpDecodeError::NonZeroPaddingCode { index, code } => {
            NativeMtpDecodeError::NonZeroPaddingCode {
                index: parent_row * padded_k + index,
                code,
            }
        }
        other => other,
    }
}
