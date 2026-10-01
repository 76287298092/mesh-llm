use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Fixture {
    ExactSquares,
    DenseSigned,
}

pub(super) fn words(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect()
}

pub(super) fn input(fixture: Fixture, rows: usize, width: usize) -> Result<Vec<u16>> {
    ensure!(matches!(width, 256 | 5120) && (1..=120).contains(&rows), "fixture extent invalid");
    let count = rows.checked_mul(width).ok_or_else(|| anyhow::anyhow!("fixture overflow"))?;
    ensure!(count <= 30_720, "fixture exceeds readback bound");
    (0..count).map(|index| {
        let row = index / width;
        let column = index % width;
        let sign = if (row + column).is_multiple_of(2) { 0 } else { 0x8000 };
        let magnitude = match fixture {
            Fixture::ExactSquares => [0x3f00, 0x3f80, 0x4000, 0x4080, 0x4100][row % 5],
            Fixture::DenseSigned => {
                let fraction = u16::try_from((column * 37 + row * 19) % 128)?;
                let exponent = u16::try_from((column / 7 + row * 3) % 9)?;
                0x3b80 + exponent * 128 + fraction
            }
        };
        Ok(sign | magnitude)
    }).collect()
}
