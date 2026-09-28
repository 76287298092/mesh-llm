//! Host semantic model of generic PTX PRMT, checked against a logical signed table.
//! The linear FP64 oracle remains `nvfp4_linear_reference`, not this model.

/// Emulate generic (no mode suffix) PRMT, including selector-bit-3 sign replication.
fn prmt(low: u32, high: u32, control: u32) -> u32 {
    let bytes = u64::from(low) | (u64::from(high) << 32);
    let mut output = 0;
    for lane in 0..4 {
        let selector = (control >> (lane * 4)) & 15;
        let byte = ((bytes >> ((selector & 7) * 8)) & 255) as u32;
        let selected = if selector & 8 == 0 {
            byte
        } else if byte & 128 == 0 {
            0
        } else {
            255
        };
        output |= selected << (lane * 8);
    }
    output
}

/// Candidate instruction semantics, deliberately separate from the logical oracle.
pub fn modeled_pack(word: u32, first_shift: u32) -> u32 {
    let ctrl = word >> first_shift;
    let selectors = ctrl & 0x7777;
    let nonzero = (ctrl | (ctrl >> 1) | (ctrl >> 2)) & 0x1111;
    let effective_sign = ctrl & (nonzero << 3);
    let magnitudes = prmt(0x0302_0100, 0x0c08_0604, selectors);
    let sign_bytes = prmt(0x8080_8080, 0x8080_8080, effective_sign);
    let sign_ones = sign_bytes & 0x0101_0101;
    let sign_mask = sign_ones.wrapping_mul(255);
    (magnitudes ^ sign_mask).wrapping_add(sign_ones)
}

/// Direct per-code logical table, without PRMT, sign masks, or word arithmetic.
pub fn logical_pack(control: u16) -> u32 {
    const SIGNED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let bytes = std::array::from_fn(|lane| SIGNED[usize::from((control >> (lane * 4)) & 15)] as u8);
    u32::from_le_bytes(bytes)
}

/// Every possible four-nibble input, in both halfwords with hostile neighbours.
pub fn exhaustive_proof() -> bool {
    (0..=u16::MAX).all(|control| {
        let low = u32::from(control);
        let high = low << 16;
        let expected = logical_pack(control);
        [0, 0x8888_0000, 0xffff_0000, (!low & 0xffff) << 16]
            .into_iter()
            .all(|neighbour| modeled_pack(low | neighbour, 0) == expected)
            && [0, 0x8888, 0xffff, !low & 0xffff]
                .into_iter()
                .all(|neighbour| modeled_pack(high | neighbour, 16) == expected)
    })
}

pub struct Fixture {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub activation: Vec<u8>,
    pub weights: Vec<u8>,
    pub activation_scales: Vec<u8>,
    pub weight_scales: Vec<u8>,
}

impl Fixture {
    fn filled(name: String, n: usize, k: usize, activation: u8, weight: u8) -> Self {
        Self {
            name,
            n,
            k,
            activation: vec![activation; k / 2],
            weights: vec![weight; n * k / 2],
            activation_scales: vec![0x38; k / 16],
            weight_scales: vec![0x38; n * k / 16],
        }
    }
}

/// Two batches cover all controls. Sixteen basis vectors independently observe
/// every byte of all four weight pack calls; no cancellation can hide a byte error.
/// Activation-side exhaustive expansion is covered by the shared-helper host proof,
/// not claimed as exhaustive GPU activation coverage.
pub fn controls(upper: bool, position: usize) -> Fixture {
    assert!(position < 16);
    let start = if upper { 32768 } else { 0 };
    let mut f = Fixture::filled(
        format!("controls-{start}-basis-{position}"),
        32768,
        16,
        0,
        0,
    );
    f.activation[position / 2] = 2 << ((position % 2) * 4);
    for (index, row) in f.weights.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let pair = ((start + index) as u16).to_le_bytes();
        for chunk in row.as_chunks_mut::<2>().0 {
            chunk.copy_from_slice(&pair);
        }
    }
    f
}

pub fn code_pairs(activation: u8) -> Fixture {
    assert!(activation < 16);
    let mut f = Fixture::filled(
        format!("code-pairs-{activation}"),
        16,
        16,
        activation * 17,
        0,
    );
    for (code, row) in f.weights.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        row.fill(code as u8 * 17);
    }
    f
}

pub fn scale_pairs(activation: u8) -> Fixture {
    assert!(activation <= 126);
    let mut f = Fixture::filled(format!("scale-pairs-{activation}"), 127, 16, 0x9f, 0x73);
    f.activation_scales.fill(activation);
    f.weight_scales = (0..=126).collect();
    f
}

pub fn patterned(n: usize, k: usize) -> Fixture {
    let mut f = Fixture::filled(format!("mixed-n{n}-k{k}"), n, k, 0, 0);
    for (i, byte) in f.activation.iter_mut().enumerate() {
        *byte = (i.wrapping_mul(73).wrapping_add(i / 11)) as u8;
    }
    for (i, byte) in f.weights.iter_mut().enumerate() {
        *byte = (i.wrapping_mul(37).wrapping_add(i / 7).wrapping_add(0x88)) as u8;
    }
    for (i, scale) in f.activation_scales.iter_mut().enumerate() {
        *scale = ((i * 43 + 1) % 127) as u8;
    }
    for (i, scale) in f.weight_scales.iter_mut().enumerate() {
        *scale = ((i * 61 + 7) % 127) as u8;
    }
    f
}

pub fn extrema() -> Fixture {
    let mut f = Fixture::filled("max-k-max-scales-cancellation".into(), 5, 32768, 0x77, 0);
    let bytes = f.k / 2;
    f.weights[..bytes].fill(0x77);
    f.weights[bytes..2 * bytes].fill(0xff);
    f.weights[2 * bytes..(5 * bytes / 2)].fill(0x77);
    f.weights[(5 * bytes / 2)..3 * bytes].fill(0xff);
    f.weights[3 * bytes..4 * bytes].fill(0x88);
    // Last row alternates negative zero next to the largest positive value.
    f.weights[4 * bytes..].fill(0x78);
    f.activation_scales.fill(126);
    f.weight_scales.fill(126);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_four_nibble_control_matches_independent_signed_table() {
        assert!(exhaustive_proof());
    }

    #[test]
    fn negative_zero_does_not_carry_into_any_neighbour() {
        for position in 0..4 {
            for other in 0..4 {
                if position == other {
                    continue;
                }
                for code in 0..16_u16 {
                    let control = (8 << (position * 4)) | (code << (other * 4));
                    assert_eq!(modeled_pack(u32::from(control), 0), logical_pack(control));
                }
            }
        }
        assert_eq!(modeled_pack(0x8888, 0), 0);
        // Without zero-sign suppression this would be 0x01010100, not zero.
        let broken = 0xffff_ffff_u32.wrapping_add(0x0101_0101);
        assert_eq!(broken, 0x0101_0100);
    }

    #[test]
    fn generic_prmt_sign_replication_is_not_a_signed_table_lookup() {
        assert_eq!(prmt(0x0302_0100, 0x0c08_0604, 0xfedc), 0);
        assert_eq!(prmt(0x8080_8080, 0x8080_8080, 0x8080), 0xff80_ff80);
    }

    #[test]
    fn exhaustive_gpu_fixture_batches_cover_every_control_and_basis_position() {
        for upper in [false, true] {
            for position in 0..16 {
                let f = controls(upper, position);
                assert_eq!(f.weights.len(), f.n * 8);
                assert_eq!(f.activation.iter().filter(|&&b| b != 0).count(), 1);
                assert_eq!(f.activation[position / 2], 2 << ((position % 2) * 4));
                for (row, bytes) in f.weights.as_chunks::<8>().0.iter().enumerate() {
                    let code = row + if upper { 32768 } else { 0 };
                    for pair in bytes.as_chunks::<2>().0 {
                        assert_eq!(u16::from_le_bytes([pair[0], pair[1]]) as usize, code);
                    }
                }
            }
        }
    }
}
