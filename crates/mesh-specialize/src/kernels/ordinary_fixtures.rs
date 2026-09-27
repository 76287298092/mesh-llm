//! Logical ordinary-MMA fixtures packed independently for the device probes.

use crate::reference::matmul;

const M: usize = 16;
const N: usize = 8;
const LANES: usize = 32;
const FLOAT_K: usize = 16;
const INT8_K: usize = 32;
const FLOAT16_MAGNITUDES: [u16; 8] = [
    0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
];

pub(crate) struct MmaFixture {
    pub name: String,
    pub kernel: &'static str,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub expected: Vec<f32>,
    pub integer_output: bool,
}

#[derive(Clone, Copy)]
enum FloatFormat {
    Bf16,
    Fp16,
}

pub(crate) fn fixtures() -> Result<Vec<MmaFixture>, String> {
    let mut result = Vec::with_capacity(12);
    result.extend(float_fixtures("bf16", "probe_bf16_mma", FloatFormat::Bf16)?);
    result.extend(float_fixtures("fp16", "probe_fp16_mma", FloatFormat::Fp16)?);
    result.extend(int8_fixtures()?);
    Ok(result)
}

fn float_fixtures(
    prefix: &str,
    kernel: &'static str,
    format: FloatFormat,
) -> Result<Vec<MmaFixture>, String> {
    let mut result = Vec::with_capacity(4);
    result.push(float_fixture(
        &format!("{prefix}-ones"),
        kernel,
        format,
        vec![1.0; M * FLOAT_K],
        vec![1.0; FLOAT_K * N],
    )?);
    for seed in [1393_u64, 5090, 27] {
        let mut state = seed;
        let a = small_float_values(M * FLOAT_K, &mut state);
        let b = small_float_values(FLOAT_K * N, &mut state);
        result.push(float_fixture(
            &format!("{prefix}-signed-{seed}"),
            kernel,
            format,
            a,
            b,
        )?);
    }
    Ok(result)
}

fn float_fixture(
    name: &str,
    kernel: &'static str,
    format: FloatFormat,
    logical_a: Vec<f32>,
    logical_b: Vec<f32>,
) -> Result<MmaFixture, String> {
    let expected = matmul(&logical_a, &logical_b, M, N, FLOAT_K)?;
    let a = pack_float_a(&logical_a, format)?;
    let b = pack_float_b(&logical_b, format)?;
    Ok(MmaFixture {
        name: name.into(),
        kernel,
        a,
        b,
        expected,
        integer_output: false,
    })
}

fn small_float_values(length: usize, state: &mut u64) -> Vec<f32> {
    (0..length)
        .map(|_| f32::from(next_small_integer(state)))
        .collect()
}

fn next_small_integer(state: &mut u64) -> i8 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 32) % 15) as i16 - 7) as i8
}

fn pack_float_a(logical: &[f32], format: FloatFormat) -> Result<Vec<u32>, String> {
    if logical.len() != M * FLOAT_K {
        return Err("ordinary float A matrix has an invalid shape".to_string());
    }
    let mut packed = vec![0_u32; LANES * 4];
    for lane in 0..LANES {
        let group = lane / 4;
        let thread = lane % 4;
        for element in 0..8 {
            let row = if element < 2 || (4..6).contains(&element) {
                group
            } else {
                group + 8
            };
            let col = thread * 2 + element % 2 + if element >= 4 { 8 } else { 0 };
            let value = encode_float(logical[row * FLOAT_K + col], format)?;
            packed[lane * 4 + element / 2] |= u32::from(value) << ((element % 2) * 16);
        }
    }
    Ok(packed)
}

fn pack_float_b(logical: &[f32], format: FloatFormat) -> Result<Vec<u32>, String> {
    if logical.len() != FLOAT_K * N {
        return Err("ordinary float B matrix has an invalid shape".to_string());
    }
    let mut packed = vec![0_u32; LANES * 2];
    for lane in 0..LANES {
        let group = lane / 4;
        let thread = lane % 4;
        for element in 0..4 {
            let row = thread * 2 + element % 2 + if element >= 2 { 8 } else { 0 };
            let value = encode_float(logical[row * N + group], format)?;
            packed[lane * 2 + element / 2] |= u32::from(value) << ((element % 2) * 16);
        }
    }
    Ok(packed)
}

fn encode_float(value: f32, format: FloatFormat) -> Result<u16, String> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > 7.0 {
        return Err("ordinary float fixture values must be integers from -7 through 7".to_string());
    }
    match format {
        FloatFormat::Bf16 => Ok((value.to_bits() >> 16) as u16),
        FloatFormat::Fp16 => {
            let magnitude = value.abs() as usize;
            let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
            Ok(FLOAT16_MAGNITUDES[magnitude] | sign)
        }
    }
}

fn int8_fixtures() -> Result<Vec<MmaFixture>, String> {
    let mut result = Vec::with_capacity(4);
    result.push(int8_fixture(
        "int8-ones",
        vec![1_i8; M * INT8_K],
        vec![1_i8; INT8_K * N],
    )?);
    for seed in [1393_u64, 5090, 27] {
        let (a, b) = signed_int8_inputs(seed);
        result.push(int8_fixture(&format!("int8-signed-{seed}"), a, b)?);
    }
    Ok(result)
}

fn signed_int8_inputs(seed: u64) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
    let mut a = (0..M * INT8_K)
        .map(|_| next_small_integer(&mut state).saturating_mul(2))
        .collect::<Vec<_>>();
    let mut b = (0..INT8_K * N)
        .map(|_| next_small_integer(&mut state).saturating_mul(2))
        .collect::<Vec<_>>();
    a[0] = i8::MIN;
    a[1] = i8::MAX;
    b[0] = i8::MAX;
    b[N] = i8::MIN;
    (a, b)
}

fn int8_fixture(name: &str, logical_a: Vec<i8>, logical_b: Vec<i8>) -> Result<MmaFixture, String> {
    let dense_a = logical_a
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let dense_b = logical_b
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let expected = matmul(&dense_a, &dense_b, M, N, INT8_K)?;
    let a = pack_int8_a(&logical_a)?;
    let b = pack_int8_b(&logical_b)?;
    Ok(MmaFixture {
        name: name.into(),
        kernel: "probe_int8_mma",
        a,
        b,
        expected,
        integer_output: true,
    })
}

fn pack_int8_a(logical: &[i8]) -> Result<Vec<u32>, String> {
    if logical.len() != M * INT8_K {
        return Err("ordinary INT8 A matrix has an invalid shape".to_string());
    }
    let mut packed = vec![0_u32; LANES * 4];
    for lane in 0..LANES {
        let group = lane / 4;
        let thread = lane % 4;
        for element in 0..16 {
            let row = if element < 4 || (8..12).contains(&element) {
                group
            } else {
                group + 8
            };
            let col = thread * 4 + element % 4 + if element >= 8 { 16 } else { 0 };
            let value = logical[row * INT8_K + col] as u8;
            packed[lane * 4 + element / 4] |= u32::from(value) << ((element % 4) * 8);
        }
    }
    Ok(packed)
}

fn pack_int8_b(logical: &[i8]) -> Result<Vec<u32>, String> {
    if logical.len() != INT8_K * N {
        return Err("ordinary INT8 B matrix has an invalid shape".to_string());
    }
    let mut packed = vec![0_u32; LANES * 2];
    for lane in 0..LANES {
        let group = lane / 4;
        let thread = lane % 4;
        for element in 0..8 {
            let row = thread * 4 + element % 4 + if element >= 4 { 16 } else { 0 };
            let value = logical[row * N + group] as u8;
            packed[lane * 2 + element / 4] |= u32::from(value) << ((element % 4) * 8);
        }
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::{FloatFormat, encode_float, fixtures, pack_float_a, pack_float_b};

    #[test]
    fn fixtures_have_expected_shapes_ones_and_signed_outputs() {
        let cases = fixtures().unwrap();
        assert_eq!(cases.len(), 12);
        for (index, case) in cases.iter().enumerate() {
            assert_eq!(
                case.kernel,
                ["probe_bf16_mma", "probe_fp16_mma", "probe_int8_mma"][index / 4]
            );
            assert_eq!(case.a.len(), 128, "{}", case.name);
            assert_eq!(case.b.len(), 64, "{}", case.name);
            assert_eq!(case.expected.len(), 128, "{}", case.name);
            assert!(case.expected.iter().all(|value| value.is_finite()));
            assert_eq!(case.integer_output, index >= 8, "{}", case.name);
        }
        for case in [&cases[0], &cases[4], &cases[8]] {
            let k = if case.integer_output { 32.0 } else { 16.0 };
            assert_eq!(case.expected, vec![k; 128], "{}", case.name);
        }
        for case in cases.iter().filter(|case| case.name.contains("signed")) {
            assert!(
                case.expected.iter().any(|value| *value < 0.0),
                "{}",
                case.name
            );
            assert!(
                case.expected.iter().any(|value| *value > 0.0),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn float_encodings_and_packed_words_match_hand_worked_values() {
        assert_eq!(encode_float(-7.0, FloatFormat::Bf16).unwrap(), 0xc0e0);
        assert_eq!(encode_float(1.0, FloatFormat::Fp16).unwrap(), 0x3c00);
        assert_eq!(encode_float(-7.0, FloatFormat::Fp16).unwrap(), 0xc700);

        let a = (0..16 * 16)
            .map(|index| (index % 15) as f32 - 7.0)
            .collect::<Vec<_>>();
        let b = (0..16 * 8)
            .map(|index| (index % 15) as f32 - 7.0)
            .collect::<Vec<_>>();
        let packed_a = pack_float_a(&a, FloatFormat::Bf16).unwrap();
        let packed_b = pack_float_b(&b, FloatFormat::Bf16).unwrap();
        assert_eq!(packed_a[0], 0xc0c0_c0e0);
        assert_eq!(packed_b[0], 0x3f80_c0e0);
    }

    #[test]
    fn signed_int8_cases_pack_both_extreme_signs() {
        let cases = fixtures().unwrap();
        for case in &cases[9..] {
            let bytes = case
                .a
                .iter()
                .chain(&case.b)
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            assert!(bytes.contains(&0x80), "{}", case.name);
            assert!(bytes.contains(&0x7f), "{}", case.name);
            assert_eq!(case.a[0] & 0xffff, 0x0000_7f80);
            assert_eq!(case.b[0] & 0xffff, 0x0000_807f);
        }
    }
}
