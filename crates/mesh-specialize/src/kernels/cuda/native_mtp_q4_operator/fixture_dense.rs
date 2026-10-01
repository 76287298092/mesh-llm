use super::Fixture;
use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView};
use anyhow::{Context as _, Result};

const LOGICAL_K: usize = 2179;
const PARENT_ROWS: usize = 12;
const SOURCE_ROWS: [usize; 9] = [10, 2, 8, 0, 11, 4, 6, 1, 9];

pub(super) fn build() -> Result<Fixture> {
    let padded_k = LOGICAL_K
        .checked_add(127)
        .context("dense fixture padded K overflows")?
        / 128
        * 128;
    let code_bytes = PARENT_ROWS
        .checked_mul(padded_k / 2)
        .context("dense fixture code extent overflows")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("dense fixture scale offset overflows")?;
    let groups_per_row = padded_k / 64;
    let scale_count = PARENT_ROWS
        .checked_mul(groups_per_row)
        .context("dense fixture scale count overflows")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("dense fixture scale extent overflows")?;
    let object_bytes_len = scale_offset
        .checked_add(scale_bytes)
        .context("dense fixture object extent overflows")?;
    let mut object_bytes = vec![0; object_bytes_len];
    for parent_row in 0..PARENT_ROWS {
        for k in 0..LOGICAL_K {
            let residue = (parent_row * 11 + k * 7) % 16;
            let code = i8::try_from(residue).context("dense fixture Q4 code is out of range")? - 8;
            let nibble = code.to_ne_bytes()[0] & 0x0f;
            let packed_index = parent_row * (padded_k / 2) + k / 2;
            if k.is_multiple_of(2) {
                object_bytes[packed_index] |= nibble;
            } else {
                object_bytes[packed_index] |= nibble << 4;
            }
        }
        for group in 0..groups_per_row {
            let scale = if group * 64 < LOGICAL_K {
                match (parent_row + group) % 4 {
                    0 => 0x3c00_u16,
                    1 => 0x3800,
                    2 => 0x4000,
                    _ => 0x3a00,
                }
            } else {
                0
            };
            let offset = scale_offset + (parent_row * groups_per_row + group) * 2;
            object_bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    let input_bf16 = (0..LOGICAL_K)
        .map(|k| match k % 8 {
            0 => 0x3f80,
            1 => 0xbf80,
            2 => 0x3f00,
            3 => 0xbf00,
            4 => 0x3fc0,
            5 => 0xbfc0,
            6 => 0x3e80,
            _ => 0xbe80,
        })
        .collect();
    let scale_bytes = u64::try_from(scale_bytes).context("dense fixture scale size exceeds u64")?;
    let view = Q4MatrixView {
        object_id: "synthetic-native-mtp-q4-dense".into(),
        shape: [SOURCE_ROWS.len(), LOGICAL_K],
        padded_k,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(code_bytes)?,
        },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset)?,
            bytes: scale_bytes,
        },
        scale_count,
        source_rows: SOURCE_ROWS.to_vec(),
    };
    Ok(Fixture {
        name: "dense-n9-k2179",
        view,
        object_bytes,
        input_bf16,
        proposal_tokens: vec![91_337, 17, 5, 23, 41, 59, 71, 83, 97],
    })
}
