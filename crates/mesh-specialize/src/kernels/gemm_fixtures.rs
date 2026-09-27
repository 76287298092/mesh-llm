//! Host fixtures for tiled NVFP4 GEMM qualification.

use super::nvfp4_layout::pack;
use crate::reference::{decode_e2m1, decode_ue4m3};

const TILE_M: usize = 16;
const TILE_N: usize = 8;
const TILE_K: usize = 64;
const A_WORDS_PER_TILE: usize = 128;
const B_WORDS_PER_TILE: usize = 64;
const SCALE_WORDS_PER_TILE: usize = 32;
const SCALE_CODES: [u8; 4] = [0x28, 0x30, 0x38, 0x40];
const ZERO_A: [u8; TILE_M * TILE_K] = [0; TILE_M * TILE_K];
const ZERO_B: [u8; TILE_K * TILE_N] = [0; TILE_K * TILE_N];
const ZERO_SCALE_A: [u8; TILE_M * 4] = [0; TILE_M * 4];
const ZERO_SCALE_B: [u8; 4 * TILE_N] = [0; 4 * TILE_N];
#[cfg(any(test, feature = "validation"))]
const MAX_DENSE_INPUT_ELEMENTS: usize = 32 * 1024 * 1024;

pub(super) struct Fixture {
    pub name: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub m_tiles: u32,
    pub n_tiles: u32,
    pub k_tiles: u32,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub scale_a: Vec<u32>,
    pub scale_b: Vec<u32>,
    pub expected: Vec<f32>,
}

#[cfg(any(test, feature = "validation"))]
pub(super) fn dense_inputs(m: usize, n: usize, k: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (a_len, b_len) = dense_input_lengths(m, n, k)?;
    let scales = decoded_scale_codes()?;
    let mut a = reserve_dense_values(a_len, "A")?;
    let mut b = reserve_dense_values(b_len, "B")?;

    for row in 0..m {
        for index in 0..k {
            let group = index / 16;
            let value = decode_e2m1(a_code(row, index))? * scales[(row + group) % 4];
            a.push(value);
        }
    }
    for index in 0..k {
        for column in 0..n {
            let group = index / 16;
            let value = decode_e2m1(b_code(index, column))? * scales[(group + column) % 4];
            b.push(value);
        }
    }
    Ok((a, b))
}

#[cfg(any(test, feature = "validation"))]
fn decoded_scale_codes() -> Result<[f32; 4], String> {
    Ok([
        decode_ue4m3(SCALE_CODES[0])?,
        decode_ue4m3(SCALE_CODES[1])?,
        decode_ue4m3(SCALE_CODES[2])?,
        decode_ue4m3(SCALE_CODES[3])?,
    ])
}

#[cfg(any(test, feature = "validation"))]
fn dense_input_lengths(m: usize, n: usize, k: usize) -> Result<(usize, usize), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("dense GEMM dimensions must be positive".to_string());
    }
    let a_len = checked_product(m, k, "dense A input")?;
    let b_len = checked_product(k, n, "dense B input")?;
    if a_len > MAX_DENSE_INPUT_ELEMENTS || b_len > MAX_DENSE_INPUT_ELEMENTS {
        return Err("dense GEMM inputs exceed the 32 Mi-element limit".to_string());
    }
    Ok((a_len, b_len))
}

#[cfg(any(test, feature = "validation"))]
fn reserve_dense_values(size: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| format!("dense {label} input allocation failed"))?;
    Ok(values)
}

#[cfg(target_os = "linux")]
pub(super) fn fixtures() -> Result<Vec<Fixture>, String> {
    [
        ("m1-n8-k64", 1, 8, 64),
        ("m17-n13-k71", 17, 13, 71),
        ("m32-n24-k192", 32, 24, 192),
        ("m1-n5120-k5120", 1, 5120, 5120),
        ("m128-n5120-k5120", 128, 5120, 5120),
    ]
    .into_iter()
    .map(|(name, m, n, k)| make_fixture(name, m, n, k))
    .collect()
}

fn make_fixture(name: &str, m: usize, n: usize, k: usize) -> Result<Fixture, String> {
    let m_tiles = tile_count(m, TILE_M, "M")?;
    let n_tiles = tile_count(n, TILE_N, "N")?;
    let k_tiles = tile_count(k, TILE_K, "K")?;
    let mut a = reserve_words(tile_words(m_tiles, k_tiles, A_WORDS_PER_TILE, "A")?, "A")?;
    let mut b = reserve_words(tile_words(n_tiles, k_tiles, B_WORDS_PER_TILE, "B")?, "B")?;
    let mut scale_a = reserve_words(
        tile_words(m_tiles, k_tiles, SCALE_WORDS_PER_TILE, "A scales")?,
        "A scales",
    )?;
    let mut scale_b = reserve_words(
        tile_words(n_tiles, k_tiles, SCALE_WORDS_PER_TILE, "B scales")?,
        "B scales",
    )?;

    append_a_tiles(m, k, m_tiles, k_tiles, &mut a, &mut scale_a)?;
    append_b_tiles(n, k, n_tiles, k_tiles, &mut b, &mut scale_b)?;

    Ok(Fixture {
        name: name.to_string(),
        m,
        n,
        k,
        m_tiles: u32::try_from(m_tiles).map_err(|_| "M tile count exceeds u32".to_string())?,
        n_tiles: u32::try_from(n_tiles).map_err(|_| "N tile count exceeds u32".to_string())?,
        k_tiles: u32::try_from(k_tiles).map_err(|_| "K tile count exceeds u32".to_string())?,
        a,
        b,
        scale_a,
        scale_b,
        expected: expected_by_period(m, n, k)?,
    })
}

fn append_a_tiles(
    m: usize,
    k: usize,
    m_tiles: usize,
    k_tiles: usize,
    packed_a: &mut Vec<u32>,
    packed_scale_a: &mut Vec<u32>,
) -> Result<(), String> {
    for m_tile in 0..m_tiles {
        for k_tile in 0..k_tiles {
            let (codes, scales) = make_a_tile(m, k, m_tile, k_tile);
            let packed = pack(&codes, &ZERO_B, &scales, &ZERO_SCALE_B, 0, 0)?;
            packed_a.extend(packed.a);
            packed_scale_a.extend(packed.scale_a);
        }
    }
    Ok(())
}

fn append_b_tiles(
    n: usize,
    k: usize,
    n_tiles: usize,
    k_tiles: usize,
    packed_b: &mut Vec<u32>,
    packed_scale_b: &mut Vec<u32>,
) -> Result<(), String> {
    for n_tile in 0..n_tiles {
        for k_tile in 0..k_tiles {
            let (codes, scales) = make_b_tile(n, k, n_tile, k_tile);
            let packed = pack(&ZERO_A, &codes, &ZERO_SCALE_A, &scales, 0, 0)?;
            packed_b.extend(packed.b);
            packed_scale_b.extend(packed.scale_b);
        }
    }
    Ok(())
}

fn make_a_tile(m: usize, k: usize, m_tile: usize, k_tile: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0; TILE_M * TILE_K];
    for row in 0..TILE_M {
        let global_row = m_tile * TILE_M + row;
        for column in 0..TILE_K {
            let global_k = k_tile * TILE_K + column;
            if global_row < m && global_k < k {
                codes[row * TILE_K + column] = a_code(global_row, global_k);
            }
        }
    }

    let mut scales = vec![0; TILE_M * 4];
    for row in 0..TILE_M {
        let global_row = m_tile * TILE_M + row;
        for group in 0..4 {
            let global_group = k_tile * 4 + group;
            scales[row * 4 + group] = SCALE_CODES[(global_row + global_group) % 4];
        }
    }
    (codes, scales)
}

fn make_b_tile(n: usize, k: usize, n_tile: usize, k_tile: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0; TILE_K * TILE_N];
    for row in 0..TILE_K {
        let global_k = k_tile * TILE_K + row;
        for column in 0..TILE_N {
            let global_column = n_tile * TILE_N + column;
            if global_k < k && global_column < n {
                codes[row * TILE_N + column] = b_code(global_k, global_column);
            }
        }
    }

    let mut scales = vec![0; 4 * TILE_N];
    for group in 0..4 {
        let global_group = k_tile * 4 + group;
        for column in 0..TILE_N {
            let global_column = n_tile * TILE_N + column;
            scales[group * TILE_N + column] = SCALE_CODES[(global_group + global_column) % 4];
        }
    }
    (codes, scales)
}

fn a_code(row: usize, k: usize) -> u8 {
    let sign = if row.is_multiple_of(3) { 0x08 } else { 0 };
    sign | (k % 8) as u8
}

fn b_code(k: usize, column: usize) -> u8 {
    let sign = if column.is_multiple_of(5) { 0x08 } else { 0 };
    sign | ((k * 5 + 3) % 8) as u8
}

fn expected_by_period(m: usize, n: usize, k: usize) -> Result<Vec<f32>, String> {
    let row_classes = m.min(12);
    let column_classes = n.min(20);
    let class_count = checked_product(row_classes, column_classes, "expected class")?;
    let mut classes = Vec::new();
    classes
        .try_reserve_exact(class_count)
        .map_err(|_| "expected class allocation failed".to_string())?;
    for row in 0..row_classes {
        for column in 0..column_classes {
            classes.push(class_dot(row, column, k)?);
        }
    }

    let output_count = checked_product(m, n, "expected output")?;
    let mut expected = Vec::new();
    expected
        .try_reserve_exact(output_count)
        .map_err(|_| "expected output allocation failed".to_string())?;
    for row in 0..m {
        for column in 0..n {
            expected.push(classes[(row % 12) * column_classes + column % 20]);
        }
    }
    Ok(expected)
}

fn class_dot(row: usize, column: usize, k: usize) -> Result<f32, String> {
    let mut sum = 0.0_f64;
    for index in 0..k {
        let group = index / 16;
        let left = f64::from(decode_e2m1(a_code(row, index))?)
            * f64::from(decode_ue4m3(SCALE_CODES[(row + group) % 4])?);
        let right = f64::from(decode_e2m1(b_code(index, column))?)
            * f64::from(decode_ue4m3(SCALE_CODES[(group + column) % 4])?);
        sum += left * right;
    }
    let result = sum as f32;
    if !result.is_finite() {
        return Err("expected output is not finite".to_string());
    }
    Ok(result)
}

fn tile_count(size: usize, tile_size: usize, label: &str) -> Result<usize, String> {
    if size == 0 {
        return Err(format!("{label} dimension must be nonzero"));
    }
    size.checked_add(tile_size - 1)
        .map(|rounded| rounded / tile_size)
        .ok_or_else(|| format!("{label} tile count overflowed"))
}

fn tile_words(
    tiles: usize,
    k_tiles: usize,
    words_per_tile: usize,
    label: &str,
) -> Result<usize, String> {
    let tile_count = checked_product(tiles, k_tiles, label)?;
    checked_product(tile_count, words_per_tile, label)
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} size overflowed"))
}

fn reserve_words(size: usize, label: &str) -> Result<Vec<u32>, String> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(size)
        .map_err(|_| format!("{label} allocation failed"))?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_DENSE_INPUT_ELEMENTS, SCALE_CODES, TILE_K, TILE_M, TILE_N, dense_inputs, make_a_tile,
        make_b_tile, make_fixture,
    };
    use crate::reference::{decode_e2m1, decode_ue4m3, matmul};

    #[test]
    fn small_fixture_expected_values_match_independent_dense_reference() {
        for (m, n, k) in [(1, 8, 64), (17, 13, 71), (32, 24, 192)] {
            let fixture = make_fixture("oracle", m, n, k).unwrap();
            assert_eq!(fixture.expected, naive_expected(m, n, k));
        }
    }

    #[test]
    fn fixture_schema_and_packed_word_counts_are_exact() {
        let fixture = make_fixture("shape-17x13x71", 17, 13, 71).unwrap();
        assert_eq!(fixture.name, "shape-17x13x71");
        assert_eq!((fixture.m, fixture.n, fixture.k), (17, 13, 71));
        assert_eq!(
            (fixture.m_tiles, fixture.n_tiles, fixture.k_tiles),
            (2, 2, 2)
        );
        assert_eq!(fixture.a.len(), 2 * 2 * 128);
        assert_eq!(fixture.b.len(), 2 * 2 * 64);
        assert_eq!(fixture.scale_a.len(), 2 * 2 * 32);
        assert_eq!(fixture.scale_b.len(), 2 * 2 * 32);
        assert_eq!(fixture.expected.len(), fixture.m * fixture.n);
    }

    #[test]
    fn edge_tiles_zero_pad_data_and_keep_positive_scale_codes() {
        let (a, scale_a) = make_a_tile(17, 71, 1, 1);
        for row in 0..TILE_M {
            for column in 0..TILE_K {
                if row != 0 || column >= 7 {
                    assert_eq!(a[row * TILE_K + column], 0);
                }
            }
        }
        assert!(scale_a.iter().all(|scale| SCALE_CODES.contains(scale)));

        let (b, scale_b) = make_b_tile(13, 71, 1, 1);
        for row in 0..TILE_K {
            for column in 0..TILE_N {
                if row >= 7 || column >= 5 {
                    assert_eq!(b[row * TILE_N + column], 0);
                }
            }
        }
        assert!(scale_b.iter().all(|scale| SCALE_CODES.contains(scale)));
    }

    #[test]
    fn dense_inputs_match_independent_logical_values_and_fixture_product() {
        let (m, n, k) = (17, 13, 71);
        let (a, b) = dense_inputs(m, n, k).unwrap();

        let mut expected_a = Vec::with_capacity(m * k);
        for row in 0..m {
            for index in 0..k {
                let sign = if row % 3 == 0 { 8 } else { 0 };
                let code = sign | (index % 8) as u8;
                let scale = SCALE_CODES[(row + index / 16) % 4];
                expected_a.push(decode_e2m1(code).unwrap() * decode_ue4m3(scale).unwrap());
            }
        }
        let mut expected_b = Vec::with_capacity(k * n);
        for index in 0..k {
            for column in 0..n {
                let sign = if column % 5 == 0 { 8 } else { 0 };
                let code = sign | ((index * 5 + 3) % 8) as u8;
                let scale = SCALE_CODES[(index / 16 + column) % 4];
                expected_b.push(decode_e2m1(code).unwrap() * decode_ue4m3(scale).unwrap());
            }
        }

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
        let product = matmul(&a, &b, m, n, k).unwrap();
        let fixture = make_fixture("dense-input-check", m, n, k).unwrap();
        assert_eq!(product, fixture.expected);
        assert_eq!(product, naive_expected(m, n, k));
    }

    #[test]
    fn dense_inputs_reject_invalid_dimensions_and_sizes() {
        assert!(dense_inputs(0, 13, 71).is_err());
        assert!(dense_inputs(usize::MAX, 1, 2).is_err());
        assert!(dense_inputs(1, usize::MAX, 2).is_err());
        assert!(dense_inputs(MAX_DENSE_INPUT_ELEMENTS + 1, 1, 1).is_err());
        assert!(dense_inputs(1, MAX_DENSE_INPUT_ELEMENTS + 1, 1).is_err());
    }

    fn naive_expected(m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut dense_a = Vec::with_capacity(m * k);
        for row in 0..m {
            for index in 0..k {
                let sign = if row % 3 == 0 { 8 } else { 0 };
                let code = sign | (index % 8) as u8;
                let scale = SCALE_CODES[(row + index / 16) % 4];
                dense_a.push(decode_e2m1(code).unwrap() * decode_ue4m3(scale).unwrap());
            }
        }
        let mut dense_b = Vec::with_capacity(k * n);
        for index in 0..k {
            for column in 0..n {
                let sign = if column % 5 == 0 { 8 } else { 0 };
                let code = sign | ((index * 5 + 3) % 8) as u8;
                let scale = SCALE_CODES[(index / 16 + column) % 4];
                dense_b.push(decode_e2m1(code).unwrap() * decode_ue4m3(scale).unwrap());
            }
        }
        matmul(&dense_a, &dense_b, m, n, k).unwrap()
    }
}
