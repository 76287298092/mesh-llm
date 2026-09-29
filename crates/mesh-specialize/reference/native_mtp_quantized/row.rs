pub(super) use super::error::NativeMtpDecodeError;

pub struct PackedRow<'a> {
    pub codes: &'a [u8],
    pub scale_bytes: &'a [u8],
    pub logical_k: usize,
    pub padded_k: usize,
}

#[cfg(test)]
#[path = "row_tests.rs"]
mod tests;

#[cfg(test)]
pub fn q8_g32_fp16_decode_row(row: PackedRow<'_>) -> Result<Vec<f32>, NativeMtpDecodeError> {
    decode_row(row, 8, 32)
}

#[cfg(test)]
pub fn q4_g64_fp16_decode_row(row: PackedRow<'_>) -> Result<Vec<f32>, NativeMtpDecodeError> {
    decode_row(row, 4, 64)
}

fn decode_row(
    row: PackedRow<'_>,
    bits_per_code: usize,
    group_size: usize,
) -> Result<Vec<f32>, NativeMtpDecodeError> {
    validate_row(&row, bits_per_code, group_size)?;
    let mut decoded = Vec::with_capacity(row.logical_k);
    for group in 0..row.padded_k / group_size {
        let scale_bits =
            u16::from_le_bytes([row.scale_bytes[group * 2], row.scale_bytes[group * 2 + 1]]);
        let scale = scale_f32(scale_bits, group)?;
        if group * group_size >= row.logical_k && scale_bits != 0 {
            return Err(NativeMtpDecodeError::NonZeroPaddingScale {
                group,
                bits: scale_bits,
            });
        }
        for lane in 0..group_size {
            let index = group * group_size + lane;
            let code = match bits_per_code {
                8 => {
                    let word = row.codes[index];
                    if word == 0x80 {
                        return Err(NativeMtpDecodeError::InvalidQ8Code { index, code: word });
                    }
                    i8::from_ne_bytes([word])
                }
                4 => {
                    let packed = row.codes[index / 2];
                    let nibble = if index.is_multiple_of(2) {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    i8::try_from(if nibble & 0x08 == 0 {
                        i16::from(nibble)
                    } else {
                        i16::from(nibble) - 16
                    })
                    .map_err(|_| NativeMtpDecodeError::InvalidGeometry("invalid Q4 code"))?
                }
                _ => {
                    return Err(NativeMtpDecodeError::InvalidGeometry(
                        "unsupported grouped code width",
                    ));
                }
            };
            if index >= row.logical_k && code != 0 {
                return Err(NativeMtpDecodeError::NonZeroPaddingCode { index, code });
            }
            if scale == 0.0 && code != 0 {
                return Err(NativeMtpDecodeError::ZeroScaleNonZeroCode { group, index });
            }
            if index < row.logical_k {
                decoded.push(scale * f32::from(code));
            }
        }
    }
    Ok(decoded)
}

fn validate_row(
    row: &PackedRow<'_>,
    bits_per_code: usize,
    group_size: usize,
) -> Result<(), NativeMtpDecodeError> {
    if row.logical_k == 0 || row.logical_k > row.padded_k || !row.padded_k.is_multiple_of(128) {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "K must be positive, logical K must fit padded K, and padded K must align to 128",
        ));
    }
    let Some(expected_code_bytes) = row.padded_k.checked_mul(bits_per_code).map(|bits| bits / 8)
    else {
        return Err(NativeMtpDecodeError::InvalidGeometry(
            "code row extent overflows usize",
        ));
    };
    if row.codes.len() != expected_code_bytes {
        return Err(NativeMtpDecodeError::PlaneExtent {
            plane: "code",
            expected: expected_code_bytes,
            actual: row.codes.len(),
        });
    }
    let expected_scale_bytes =
        (row.padded_k / group_size)
            .checked_mul(2)
            .ok_or(NativeMtpDecodeError::InvalidGeometry(
                "scale row extent overflows usize",
            ))?;
    if row.scale_bytes.len() != expected_scale_bytes {
        return Err(NativeMtpDecodeError::PlaneExtent {
            plane: "scale",
            expected: expected_scale_bytes,
            actual: row.scale_bytes.len(),
        });
    }
    Ok(())
}

fn scale_f32(bits: u16, group: usize) -> Result<f32, NativeMtpDecodeError> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    if sign != 0 || exponent == 0x1f {
        return Err(NativeMtpDecodeError::InvalidScale { group, bits });
    }
    if exponent == 0 && fraction == 0 {
        return Ok(0.0);
    }
    Ok(if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits((u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13))
    })
}
