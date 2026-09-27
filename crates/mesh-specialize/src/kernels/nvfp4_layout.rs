const WARP_LANES: usize = 32;
const LANES_PER_GROUP: usize = 4;
const A_ROWS: usize = 16;
const A_COLUMNS: usize = 64;
const B_ROWS: usize = 64;
const B_COLUMNS: usize = 8;
const SCALE_A_ROWS: usize = 16;
const SCALE_A_COLUMNS: usize = 4;
const SCALE_B_ROWS: usize = 4;
const SCALE_B_COLUMNS: usize = 8;
const POISON_SCALE_WORD: u32 = 0x7e7e_7e7e;

pub(crate) struct PackedTile {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub scale_a: Vec<u32>,
    pub scale_b: Vec<u32>,
}

pub(crate) fn pack(
    a: &[u8],
    b: &[u8],
    scale_a: &[u8],
    scale_b: &[u8],
    selector_a: u16,
    selector_b: u16,
) -> Result<PackedTile, String> {
    validate_inputs(a, b, scale_a, scale_b, selector_a, selector_b)?;

    Ok(PackedTile {
        a: pack_a(a),
        b: pack_b(b),
        scale_a: pack_scale_a(scale_a, usize::from(selector_a)),
        scale_b: pack_scale_b(scale_b, usize::from(selector_b)),
    })
}

pub(crate) fn unpack_output(packed: &[f32]) -> Result<Vec<f32>, String> {
    if packed.len() != A_ROWS * B_COLUMNS {
        return Err("packed output must contain 128 values".to_string());
    }
    if packed.iter().any(|value| !value.is_finite()) {
        return Err("packed output contains a nonfinite value".to_string());
    }

    let mut output = vec![0.0; packed.len()];
    for (lane, lane_values) in packed.as_chunks::<4>().0.iter().enumerate() {
        let group = lane / LANES_PER_GROUP;
        let thread = lane % LANES_PER_GROUP;
        for (index, &value) in lane_values.iter().enumerate() {
            let row = group + if index >= 2 { 8 } else { 0 };
            let column = thread * 2 + index % 2;
            output[row * B_COLUMNS + column] = value;
        }
    }
    Ok(output)
}

fn validate_inputs(
    a: &[u8],
    b: &[u8],
    scale_a: &[u8],
    scale_b: &[u8],
    selector_a: u16,
    selector_b: u16,
) -> Result<(), String> {
    validate_length(a, A_ROWS * A_COLUMNS, "A")?;
    validate_length(b, B_ROWS * B_COLUMNS, "B")?;
    validate_length(scale_a, SCALE_A_ROWS * SCALE_A_COLUMNS, "A scales")?;
    validate_length(scale_b, SCALE_B_ROWS * SCALE_B_COLUMNS, "B scales")?;
    validate_nibbles(a, "A")?;
    validate_nibbles(b, "B")?;
    validate_scale_codes(scale_a, "A scales")?;
    validate_scale_codes(scale_b, "B scales")?;
    if selector_a > 1 {
        return Err("A selector must be in 0..=1".to_string());
    }
    if selector_b > 3 {
        return Err("B selector must be in 0..=3".to_string());
    }
    Ok(())
}

fn validate_length(values: &[u8], expected: usize, label: &str) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!(
            "{label} must contain exactly {expected} logical values"
        ));
    }
    Ok(())
}

fn validate_nibbles(values: &[u8], label: &str) -> Result<(), String> {
    if values.iter().any(|&value| value > 0x0f) {
        return Err(format!(
            "{label} contains a value outside the four-bit range"
        ));
    }
    Ok(())
}

fn validate_scale_codes(values: &[u8], label: &str) -> Result<(), String> {
    if values
        .iter()
        .any(|&value| value & 0x80 != 0 || value == 0x7f)
    {
        return Err(format!("{label} contains an invalid UE4M3 scale code"));
    }
    Ok(())
}

fn pack_a(values: &[u8]) -> Vec<u32> {
    let mut words = vec![0_u32; WARP_LANES * 4];
    for group in 0..8 {
        for thread in 0..LANES_PER_GROUP {
            let lane = group * LANES_PER_GROUP + thread;
            for index in 0..32 {
                let (row, column) = a_coordinate(group, thread, index);
                let word_index = lane * 4 + index / 8;
                let shift = (index % 8) * 4;
                words[word_index] |= u32::from(values[row * A_COLUMNS + column]) << shift;
            }
        }
    }
    words
}

fn a_coordinate(group: usize, thread: usize, index: usize) -> (usize, usize) {
    let row = if index < 8 || (16..24).contains(&index) {
        group
    } else {
        group + 8
    };
    let column = thread * 8 + index % 8 + if index >= 16 { 32 } else { 0 };
    (row, column)
}

fn pack_b(values: &[u8]) -> Vec<u32> {
    let mut words = vec![0_u32; WARP_LANES * 2];
    for group in 0..8 {
        for thread in 0..LANES_PER_GROUP {
            let lane = group * LANES_PER_GROUP + thread;
            for index in 0..16 {
                let row = thread * 8 + index % 8 + if index >= 8 { 32 } else { 0 };
                let column = group;
                let word_index = lane * 2 + index / 8;
                let shift = (index % 8) * 4;
                words[word_index] |= u32::from(values[row * B_COLUMNS + column]) << shift;
            }
        }
    }
    words
}

fn pack_scale_a(values: &[u8], selector: usize) -> Vec<u32> {
    let mut words = vec![POISON_SCALE_WORD; WARP_LANES];
    for group in 0..8 {
        let first_lane = group * LANES_PER_GROUP + selector * 2;
        words[first_lane] = pack_scale_row(values, group, SCALE_A_COLUMNS);
        words[first_lane + 1] = pack_scale_row(values, group + 8, SCALE_A_COLUMNS);
    }
    words
}

fn pack_scale_b(values: &[u8], selector: usize) -> Vec<u32> {
    let mut words = vec![POISON_SCALE_WORD; WARP_LANES];
    for column in 0..B_COLUMNS {
        let lane = column * LANES_PER_GROUP + selector;
        words[lane] = pack_scale_column(values, column);
    }
    words
}

fn pack_scale_row(values: &[u8], row: usize, columns: usize) -> u32 {
    let start = row * columns;
    pack_scale_bytes([
        values[start],
        values[start + 1],
        values[start + 2],
        values[start + 3],
    ])
}

fn pack_scale_column(values: &[u8], column: usize) -> u32 {
    pack_scale_bytes([
        values[column],
        values[SCALE_B_COLUMNS + column],
        values[SCALE_B_COLUMNS * 2 + column],
        values[SCALE_B_COLUMNS * 3 + column],
    ])
}

fn pack_scale_bytes(bytes: [u8; 4]) -> u32 {
    bytes
        .into_iter()
        .enumerate()
        .fold(0_u32, |word, (index, byte)| {
            word | (u32::from(byte) << (index * 8))
        })
}

#[cfg(test)]
mod tests {
    use super::{POISON_SCALE_WORD, PackedTile, pack, unpack_output};

    fn zero_inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        (vec![0; 1024], vec![0; 512], vec![0; 64], vec![0; 32])
    }

    fn scale_a_row_word(row: usize) -> u32 {
        let first = (row * 4) as u32;
        first | (first + 1) << 8 | (first + 2) << 16 | (first + 3) << 24
    }

    fn scale_b_column_word(column: usize) -> u32 {
        let column = column as u32;
        column | (column + 8) << 8 | (column + 16) << 16 | (column + 24) << 24
    }

    #[test]
    fn packs_handpicked_a_and_b_lane_words() {
        let (mut a, mut b, scale_a, scale_b) = zero_inputs();
        a[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        a[8 * 64..8 * 64 + 8].copy_from_slice(&[8, 7, 6, 5, 4, 3, 2, 1]);
        a[32..40].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        a[8 * 64 + 32..8 * 64 + 40].fill(15);

        let lane = 4 + 2;
        a[64 + 16..64 + 24].copy_from_slice(&[8, 9, 10, 11, 12, 13, 14, 15]);
        a[9 * 64 + 16..9 * 64 + 24].copy_from_slice(&[7, 6, 5, 4, 3, 2, 1, 0]);
        a[64 + 48..64 + 56].copy_from_slice(&[15, 14, 13, 12, 11, 10, 9, 8]);
        a[9 * 64 + 48..9 * 64 + 56].fill(1);

        for offset in 0..8 {
            b[offset * 8] = offset as u8 + 1;
            b[(32 + offset) * 8] = 8 - offset as u8;
            b[(16 + offset) * 8 + 3] = offset as u8 + 1;
            b[(48 + offset) * 8 + 3] = 8 - offset as u8;
        }

        let tile = pack(&a, &b, &scale_a, &scale_b, 1, 2).unwrap();
        assert_eq!(tile.a.len(), 128);
        assert_eq!(tile.a[0], 0x8765_4321);
        assert_eq!(tile.a[1], 0x1234_5678);
        assert_eq!(tile.a[2], 0x7654_3210);
        assert_eq!(tile.a[3], 0xffff_ffff);
        assert_eq!(tile.a[lane * 4], 0xfedc_ba98);
        assert_eq!(tile.a[lane * 4 + 1], 0x0123_4567);
        assert_eq!(tile.a[lane * 4 + 2], 0x89ab_cdef);
        assert_eq!(tile.a[lane * 4 + 3], 0x1111_1111);

        let b_lane = 3 * 4 + 2;
        assert_eq!(tile.b.len(), 64);
        assert_eq!(tile.b[0], 0x8765_4321);
        assert_eq!(tile.b[1], 0x1234_5678);
        assert_eq!(tile.b[b_lane * 2], 0x8765_4321);
        assert_eq!(tile.b[b_lane * 2 + 1], 0x1234_5678);
    }

    #[test]
    fn every_selector_places_selected_scales_and_leaves_poison_elsewhere() {
        let (a, b, _, _) = zero_inputs();
        let scale_a: Vec<u8> = (0..64).map(|value| value as u8).collect();
        let scale_b: Vec<u8> = (0..32).map(|value| value as u8).collect();

        for selector_a in 0_u16..=1 {
            let first_a_thread = usize::from(selector_a) * 2;
            for selector_b in 0_u16..=3 {
                let selected_b_thread = usize::from(selector_b);
                let tile = pack(&a, &b, &scale_a, &scale_b, selector_a, selector_b).unwrap();
                for lane in 0_usize..32 {
                    let group = lane / 4;
                    let thread = lane % 4;
                    let expected_a = if thread == first_a_thread {
                        scale_a_row_word(group)
                    } else if thread == first_a_thread + 1 {
                        scale_a_row_word(group + 8)
                    } else {
                        POISON_SCALE_WORD
                    };
                    let expected_b = if thread == selected_b_thread {
                        scale_b_column_word(group)
                    } else {
                        POISON_SCALE_WORD
                    };
                    assert_eq!(tile.scale_a[lane], expected_a, "A lane {lane}");
                    assert_eq!(tile.scale_b[lane], expected_b, "B lane {lane}");
                }
            }
        }
    }

    #[test]
    fn unpack_output_covers_row_major_tile_positions() {
        let packed: Vec<f32> = (0..128).map(|value| value as f32).collect();
        let output = unpack_output(&packed).unwrap();

        assert_eq!(output.len(), 128);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 4.0);
        assert_eq!(output[3], 5.0);
        assert_eq!(output[8 * 8], 2.0);
        assert_eq!(output[8 * 8 + 1], 3.0);
        assert_eq!(output[8 * 8 + 2], 6.0);
        assert_eq!(output[8 * 8 + 3], 7.0);
        assert_eq!(output[8], 16.0);
        assert_eq!(output[9 * 8], 18.0);

        let mut values = output.clone();
        values.sort_by(f32::total_cmp);
        assert_eq!(values, packed);
    }

    #[test]
    fn rejects_bad_lengths_nibbles_scales_and_selectors() {
        let (a, b, scale_a, scale_b) = zero_inputs();
        assert!(pack(&a[..a.len() - 1], &b, &scale_a, &scale_b, 0, 0).is_err());
        assert!(pack(&a, &b[..b.len() - 1], &scale_a, &scale_b, 0, 0).is_err());
        assert!(pack(&a, &b, &scale_a[..scale_a.len() - 1], &scale_b, 0, 0).is_err());
        assert!(pack(&a, &b, &scale_a, &scale_b[..scale_b.len() - 1], 0, 0).is_err());

        let mut bad_a = a.clone();
        bad_a[100] = 16;
        assert!(pack(&bad_a, &b, &scale_a, &scale_b, 0, 0).is_err());
        let mut bad_b = b.clone();
        bad_b[100] = 16;
        assert!(pack(&a, &bad_b, &scale_a, &scale_b, 0, 0).is_err());

        let mut bad_scale_a = scale_a.clone();
        bad_scale_a[0] = 0x80;
        assert!(pack(&a, &b, &bad_scale_a, &scale_b, 0, 0).is_err());
        bad_scale_a[0] = 0x7f;
        assert!(pack(&a, &b, &bad_scale_a, &scale_b, 0, 0).is_err());
        let mut bad_scale_b = scale_b.clone();
        bad_scale_b[0] = 0x80;
        assert!(pack(&a, &b, &scale_a, &bad_scale_b, 0, 0).is_err());
        bad_scale_b[0] = 0x7f;
        assert!(pack(&a, &b, &scale_a, &bad_scale_b, 0, 0).is_err());

        assert!(pack(&a, &b, &scale_a, &scale_b, 2, 0).is_err());
        assert!(pack(&a, &b, &scale_a, &scale_b, 0, 4).is_err());
    }

    #[test]
    fn unpack_output_rejects_wrong_length_and_nonfinite_values() {
        assert!(unpack_output(&[0.0; 127]).is_err());
        assert!(unpack_output(&[f32::NAN; 128]).is_err());
        assert!(unpack_output(&[f32::INFINITY; 128]).is_err());
    }

    #[test]
    fn packed_tile_fields_are_available_to_the_parent_module() {
        let tile: PackedTile = {
            let (a, b, scale_a, scale_b) = zero_inputs();
            pack(&a, &b, &scale_a, &scale_b, 0, 0).unwrap()
        };
        assert_eq!(tile.a.len(), 128);
    }
}
