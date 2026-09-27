//! Independent logical oracle for the row-wise FP8 KV cache codec candidate.

use anyhow::{Result, ensure};

/// The candidate's fixed token-row width, in channels.
pub const ROW_WIDTH: usize = 256;
/// Maximum number of rows accepted by the host oracle.
pub const MAX_ROWS: usize = 262_144;

/// Encode status: at least one input BF16 in the row is NaN or infinity.
pub const STATUS_NONFINITE_BF16: u32 = 1;
/// Encode status: the row needs a scale above the largest finite binary16 value.
pub const STATUS_SCALE_OUT_OF_RANGE: u32 = 2;
/// Decode status: at least one row code is an E4M3FN NaN encoding.
pub const STATUS_INVALID_E4M3: u32 = 4;
/// Decode status: the row scale is not finite and strictly positive binary16.
pub const STATUS_INVALID_F16_SCALE: u32 = 8;

const SCALE_ONE_F16: u16 = 0x3c00;
const MAX_SCALE_F16: u16 = 0x7bff;
const MAX_ELEMENTS: usize = MAX_ROWS * ROW_WIDTH;

/// Row-major E4M3FN payload, one FP16 scale and one status word per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRows {
    pub codes: Vec<u8>,
    pub scales_f16: Vec<u16>,
    pub status: Vec<u32>,
}

/// Row-major BF16 output and one validation status word per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedRows {
    pub values_bf16: Vec<u16>,
    pub status: Vec<u32>,
}

/// Quantize fixed-width BF16 cache rows using one represented binary16 scale per row.
///
/// For each finite nonzero row, this selects the smallest finite positive binary16 `s` for
/// which `448 * s >= max(abs(row))`. The scale stored in `scales_f16` is also used to quantize
/// every value. This upward choice keeps the maximum inside E4M3FN's finite range despite
/// binary16 scale rounding. An all-zero row stores scale one and retains each zero's sign bit.
/// Rows with nonfinite values or scales outside binary16's finite range carry a status and have
/// deterministic zero payloads. Extent and allocation-limit failures return an error.
pub fn encode_bf16_rows(input_bf16: &[u16], rows: usize) -> Result<EncodedRows> {
    let elements = validate_rows(rows)?;
    ensure!(
        input_bf16.len() == elements,
        "FP8 KV BF16 input extent differs from rows * 256"
    );

    let mut encoded = EncodedRows {
        codes: vec![0; elements],
        scales_f16: vec![SCALE_ONE_F16; rows],
        status: vec![0; rows],
    };

    for row in 0..rows {
        let start = row * ROW_WIDTH;
        let source = &input_bf16[start..start + ROW_WIDTH];
        if source.iter().any(|&bits| !bf16_is_finite(bits)) {
            encoded.status[row] = STATUS_NONFINITE_BF16;
            continue;
        }

        let maximum = source
            .iter()
            .map(|&bits| bf16_to_f32(bits).abs())
            .fold(0.0_f32, f32::max);
        let Some(scale_bits) = choose_scale_f16(maximum) else {
            encoded.status[row] = STATUS_SCALE_OUT_OF_RANGE;
            continue;
        };
        encoded.scales_f16[row] = scale_bits;

        let represented_scale =
            f16_to_f64(scale_bits).expect("the selected scale is finite binary16") as f32;
        for (column, &bits) in source.iter().enumerate() {
            let value = bf16_to_f32(bits);
            // The candidate's declared quantization step is FP32 RNE division by the
            // stored scale, followed by logical E4M3FN nearest-even conversion.
            let scaled = value / represented_scale;
            let magnitude = f64::from(scaled.abs());
            let sign = if bits & 0x8000 == 0 { 0 } else { 0x80 };
            encoded.codes[start + column] = sign | encode_e4m3fn_rne(magnitude);
        }
    }

    Ok(encoded)
}

/// Decode finite E4M3FN rows through their represented binary16 scales to BF16.
///
/// Invalid codes and scales are accumulated as per-row status bits and yield a deterministic
/// zero BF16 row. A scale must be finite and strictly positive; positive binary16 subnormals
/// are valid. The multiplier and input E4M3 values are exactly representable in FP32, so the
/// logical FP64 product cast below agrees with the candidate's single FP32 multiplication.
pub fn decode_bf16_rows(codes: &[u8], scales_f16: &[u16], rows: usize) -> Result<DecodedRows> {
    let elements = validate_rows(rows)?;
    ensure!(
        codes.len() == elements,
        "FP8 KV code extent differs from rows * 256"
    );
    ensure!(
        scales_f16.len() == rows,
        "FP8 KV scale extent differs from rows"
    );

    let mut decoded = DecodedRows {
        values_bf16: vec![0; elements],
        status: vec![0; rows],
    };
    for (row, &scale_bits) in scales_f16.iter().enumerate().take(rows) {
        let start = row * ROW_WIDTH;
        let mut row_status = 0;
        if !is_valid_positive_f16(scale_bits) {
            row_status |= STATUS_INVALID_F16_SCALE;
        }
        if codes[start..start + ROW_WIDTH]
            .iter()
            .any(|&code| code & 0x7f == 0x7f)
        {
            row_status |= STATUS_INVALID_E4M3;
        }
        if row_status != 0 {
            decoded.status[row] = row_status;
            continue;
        }

        let scale = f16_to_f64(scale_bits).expect("validated finite positive binary16 scale");
        for (column, &code) in codes[start..start + ROW_WIDTH].iter().enumerate() {
            let represented =
                decode_e4m3fn_f64(code).expect("all row codes are finite E4M3FN after validation");
            let value = (represented * scale) as f32;
            decoded.values_bf16[start + column] = round_bf16_rne(value);
        }
    }
    Ok(decoded)
}

fn validate_rows(rows: usize) -> Result<usize> {
    ensure!((1..=MAX_ROWS).contains(&rows), "invalid FP8 KV row count");
    let elements = rows
        .checked_mul(ROW_WIDTH)
        .ok_or_else(|| anyhow::anyhow!("FP8 KV element count overflows usize"))?;
    ensure!(elements <= MAX_ELEMENTS, "FP8 KV element limit exceeded");
    Ok(elements)
}

fn choose_scale_f16(maximum: f32) -> Option<u16> {
    if maximum == 0.0 {
        return Some(SCALE_ONE_F16);
    }

    let required = f64::from(maximum) / 448.0;
    let maximum_scale = f16_to_f64(MAX_SCALE_F16)?;
    if required > maximum_scale {
        return None;
    }

    let mut low = 1_u32;
    let mut high = u32::from(MAX_SCALE_F16);
    while low < high {
        let middle = low + (high - low) / 2;
        let represented = f16_to_f64(middle as u16)
            .expect("binary search stays within positive finite binary16 values");
        if represented < required {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Some(low as u16)
}

fn bf16_is_finite(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f16_to_f64(bits: u16) -> Option<f64> {
    let sign = bits & 0x8000 != 0;
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = i32::from(bits & 0x03ff);
    if exponent == 0x1f {
        return None;
    }

    let magnitude = if exponent == 0 {
        f64::from(fraction) * 2.0_f64.powi(-24)
    } else {
        f64::from(1024 + fraction) * 2.0_f64.powi(exponent - 25)
    };
    Some(if sign { -magnitude } else { magnitude })
}

fn is_valid_positive_f16(bits: u16) -> bool {
    bits & 0x8000 == 0 && bits & 0x7fff != 0 && f16_to_f64(bits).is_some()
}

fn decode_e4m3fn_f64(code: u8) -> Option<f64> {
    let magnitude = code & 0x7f;
    if magnitude == 0x7f {
        return None;
    }
    let exponent = i32::from(magnitude >> 3);
    let fraction = i32::from(magnitude & 7);
    let positive = if exponent == 0 {
        f64::from(fraction) / 512.0
    } else {
        f64::from(8 + fraction) * 2.0_f64.powi(exponent - 10)
    };
    if code & 0x80 == 0 {
        Some(positive)
    } else {
        Some(-positive)
    }
}

fn encode_e4m3fn_rne(magnitude: f64) -> u8 {
    if magnitude >= 448.0 {
        return 126;
    }
    if magnitude <= 0.0 {
        return 0;
    }

    let mut low = 0_u32;
    let mut high = 126_u32;
    while low < high {
        let middle = low + (high - low) / 2;
        let represented = decode_e4m3fn_f64(middle as u8)
            .expect("the positive search excludes E4M3FN NaN encodings");
        if represented < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    let upper = low as u8;
    if upper == 0 {
        return 0;
    }
    let lower = upper - 1;
    let lower_value = decode_e4m3fn_f64(lower).expect("lower code is finite");
    let upper_value = decode_e4m3fn_f64(upper).expect("upper code is finite");
    let lower_distance = magnitude - lower_value;
    let upper_distance = upper_value - magnitude;
    if lower_distance < upper_distance || (lower_distance == upper_distance && lower & 1 == 0) {
        lower
    } else {
        upper
    }
}

fn round_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SCALE_F16, ROW_WIDTH, SCALE_ONE_F16, STATUS_INVALID_E4M3, STATUS_INVALID_F16_SCALE,
        STATUS_NONFINITE_BF16, STATUS_SCALE_OUT_OF_RANGE, choose_scale_f16, decode_bf16_rows,
        decode_e4m3fn_f64, encode_bf16_rows, encode_e4m3fn_rne, f16_to_f64,
    };

    #[test]
    fn signed_zero_subnormal_ties_scale_and_odd_row_tail() {
        let rows = 3;
        let mut input = vec![0_u16; rows * ROW_WIDTH];
        input[0] = 0x8000; // negative zero must retain its sign.
        let row_one = ROW_WIDTH;
        input[row_one] = 0x43e0; // 448 -> finite E4M3 maximum.
        input[row_one + 1] = 0x3b00; // 1/512 -> first E4M3 subnormal.
        input[row_one + 2] = 0xbb00; // -1/512 -> signed subnormal.
        input[row_one + 3] = 0x3a80; // 1/1024 ties to even code zero.
        input[row_one + 4] = 0x3b40; // 3/1024 ties to even code two.
        let row_two = 2 * ROW_WIDTH;
        input[row_two] = 0xc360; // -224; row scale must be 1/2.
        input[row_two + 1] = 0x42e0; // 112 at that same scale.

        let encoded = encode_bf16_rows(&input, rows).unwrap();
        assert_eq!(encoded.status, [0, 0, 0]);
        assert_eq!(encoded.scales_f16, [SCALE_ONE_F16, SCALE_ONE_F16, 0x3800]);
        assert_eq!(encoded.codes[0], 0x80);
        assert_eq!(encoded.codes[row_one], 0x7e);
        assert_eq!(encoded.codes[row_one + 1], 0x01);
        assert_eq!(encoded.codes[row_one + 2], 0x81);
        assert_eq!(encoded.codes[row_one + 3], 0x00);
        assert_eq!(encoded.codes[row_one + 4], 0x02);
        assert_eq!(encoded.codes[row_two], 0xfe);
        assert_eq!(encoded.codes[row_two + 1], 0x76);

        let decoded = decode_bf16_rows(&encoded.codes, &encoded.scales_f16, rows).unwrap();
        assert_eq!(decoded.status, [0, 0, 0]);
        assert_eq!(decoded.values_bf16[0], 0x8000);
        assert_eq!(decoded.values_bf16[row_one], 0x43e0);
        assert_eq!(decoded.values_bf16[row_two], 0xc360);
    }

    #[test]
    fn e4m3_rounding_uses_even_codes_and_saturates_finite() {
        assert_eq!(encode_e4m3fn_rne(1.0 / 1024.0), 0x00);
        assert_eq!(encode_e4m3fn_rne(3.0 / 1024.0), 0x02);
        assert_eq!(encode_e4m3fn_rne(400.0), 0x7c);
        assert_eq!(encode_e4m3fn_rne(432.0), 0x7e);
        assert_eq!(encode_e4m3fn_rne(449.0), 0x7e);
        assert_eq!(decode_e4m3fn_f64(0xfe), Some(-448.0));
        assert_eq!(decode_e4m3fn_f64(0xff), None);
    }

    #[test]
    fn represented_scale_rounds_up_and_enforces_finite_range() {
        let bits = choose_scale_f16(1.0).unwrap();
        let represented = f16_to_f64(bits).unwrap();
        assert!(represented * 448.0 >= 1.0);
        if bits > 1 {
            assert!(f16_to_f64(bits - 1).unwrap() * 448.0 < 1.0);
        }

        assert_eq!(choose_scale_f16(f32::from_bits(1)), Some(1));
        let largest_supported = f16_to_f64(MAX_SCALE_F16).unwrap() * 448.0;
        assert_eq!(
            choose_scale_f16(largest_supported as f32),
            Some(MAX_SCALE_F16)
        );
        assert_eq!(choose_scale_f16((largest_supported + 2.0) as f32), None);

        let mut unit_row = vec![0_u16; ROW_WIDTH];
        unit_row[17] = 0x3f80;
        let encoded_unit = encode_bf16_rows(&unit_row, 1).unwrap();
        assert_eq!(encoded_unit.status, [0]);
        assert_eq!(encoded_unit.scales_f16, [bits]);
        assert_eq!(encoded_unit.codes[17], 0x7e);

        let encoded = encode_bf16_rows(&vec![0x4be0; ROW_WIDTH], 1).unwrap();
        assert_eq!(encoded.status, [STATUS_SCALE_OUT_OF_RANGE]);
        assert!(encoded.codes.iter().all(|&code| code == 0));
        assert_eq!(encoded.scales_f16, [SCALE_ONE_F16]);
    }

    #[test]
    fn nonfinite_input_and_invalid_decode_rows_are_reported_and_zeroed() {
        let mut input = vec![0_u16; 2 * ROW_WIDTH];
        input[7] = 0x7fc1;
        input[ROW_WIDTH + 9] = 0xff80;
        let encoded = encode_bf16_rows(&input, 2).unwrap();
        assert_eq!(
            encoded.status,
            [STATUS_NONFINITE_BF16, STATUS_NONFINITE_BF16]
        );
        assert!(encoded.codes.iter().all(|&code| code == 0));
        assert_eq!(encoded.scales_f16, [SCALE_ONE_F16, SCALE_ONE_F16]);

        let mut codes = vec![0_u8; 2 * ROW_WIDTH];
        codes[3] = 0x7f;
        codes[ROW_WIDTH + 4] = 0xff;
        let scales = [SCALE_ONE_F16, 0x8000]; // valid scale, then negative zero.
        let decoded = decode_bf16_rows(&codes, &scales, 2).unwrap();
        assert_eq!(
            decoded.status,
            [
                STATUS_INVALID_E4M3,
                STATUS_INVALID_E4M3 | STATUS_INVALID_F16_SCALE,
            ]
        );
        assert!(decoded.values_bf16.iter().all(|&value| value == 0));
    }

    #[test]
    fn decoder_accepts_positive_subnormal_scale_and_rejects_invalid_extents() {
        let codes = vec![0x38_u8; ROW_WIDTH]; // E4M3 value 1.
        let decoded = decode_bf16_rows(&codes, &[1], 1).unwrap();
        assert_eq!(decoded.status, [0]);
        assert_eq!(decoded.values_bf16, vec![0x3380; ROW_WIDTH]);

        assert!(encode_bf16_rows(&vec![0; ROW_WIDTH - 1], 1).is_err());
        assert!(decode_bf16_rows(&codes[..ROW_WIDTH - 1], &[SCALE_ONE_F16], 1).is_err());
        assert!(decode_bf16_rows(&codes, &[], 1).is_err());
        assert!(
            decode_bf16_rows(&codes, &[0x7c00], 1).unwrap().status[0] & STATUS_INVALID_F16_SCALE
                != 0
        );
    }
}
