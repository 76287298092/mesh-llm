//! Compact BF16 rotary tables for contiguous text positions.
use anyhow::{Context, Result, ensure};

pub struct TextRope {
    inverse_frequencies: Vec<f32>,
}
pub struct Tables {
    pub cos: Vec<u16>,
    pub sin: Vec<u16>,
}

impl TextRope {
    pub fn new(rotary_dim: usize, theta: f32) -> Result<Self> {
        ensure!(
            (2..=1024).contains(&rotary_dim) && rotary_dim.is_multiple_of(2),
            "invalid text rotary dimension"
        );
        ensure!(
            theta.is_finite() && theta > 1.0,
            "invalid text rotary theta"
        );
        let inverse_frequencies = (0..rotary_dim / 2)
            .map(|frequency| {
                let denominator =
                    f64::from(theta).powf((2 * frequency) as f64 / rotary_dim as f64) as f32;
                1.0_f32 / denominator
            })
            .collect::<Vec<_>>();
        ensure!(
            inverse_frequencies
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
            "invalid rotary inverse frequency"
        );
        Ok(Self {
            inverse_frequencies,
        })
    }

    /// FP32 position/frequency product, FP64 trig rounded to FP32, then BF16 RNE.
    /// This explicit CPU profile matches the previously qualified text tables;
    /// it makes no claim about other frameworks' trigonometric implementations.
    pub fn tables(&self, past: usize, rows: usize) -> Result<Tables> {
        ensure!((1..=2048).contains(&rows), "invalid rotary row count");
        let end = past.checked_add(rows).context("rotary position overflow")?;
        ensure!(
            end <= 262_144,
            "rotary positions exceed supported text range"
        );
        let count = rows * self.inverse_frequencies.len();
        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for position in past..end {
            for &inverse in &self.inverse_frequencies {
                let angle = position as f32 * inverse;
                let angle = f64::from(angle);
                cos.push(bf16(angle.cos() as f32));
                sin.push(bf16(angle.sin() as f32));
            }
        }
        Ok(Tables { cos, sin })
    }
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_and_unit_position_have_known_values() {
        let rope = TextRope::new(8, 256.0).unwrap();
        assert_eq!(rope.inverse_frequencies, [1.0, 0.25, 0.0625, 0.015625]);
        let tables = rope.tables(0, 2).unwrap();
        assert_eq!(&tables.cos[..4], &[0x3f80; 4]);
        assert_eq!(&tables.sin[..4], &[0; 4]);
        assert_eq!(tables.cos[4], 0x3f0a);
        assert_eq!(tables.sin[4], 0x3f57);
    }
    #[test]
    fn partitioned_positions_match_and_bounds_reject_overflow() {
        let rope = TextRope::new(64, 1e7).unwrap();
        let whole = rope.tables(131070, 3).unwrap();
        let first = rope.tables(131070, 1).unwrap();
        let next = rope.tables(131071, 2).unwrap();
        assert_eq!(whole.cos, [first.cos, next.cos].concat());
        assert_eq!(whole.sin, [first.sin, next.sin].concat());
        assert!(rope.tables(262143, 1).is_ok());
        assert!(rope.tables(262143, 2).is_err());
        assert!(rope.tables(usize::MAX, 1).is_err());
        assert!(rope.tables(0, 0).is_err());
        assert!(rope.tables(0, 2049).is_err());
        assert!(TextRope::new(3, 1e7).is_err());
        assert!(TextRope::new(64, f32::NAN).is_err());
    }
    #[test]
    fn agrees_with_independent_reference_at_high_positions() {
        let rope = TextRope::new(64, 1e7).unwrap();
        for position in [0, 1, 17, 131071, 262143] {
            let actual = rope.tables(position, 1).unwrap();
            let expected =
                crate::attention_prepare_reference::text_rope_tables(&[position as u32], 64, 1e7)
                    .unwrap();
            assert_eq!(actual.cos, expected.0);
            assert_eq!(actual.sin, expected.1);
        }
    }
}
