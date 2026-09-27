//! Independent host oracle for BF16 greedy vocabulary selection.

use core::fmt;

pub const MAX_VOCABULARY: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub token: u32,
    pub bits: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyError {
    InvalidExtent { len: usize },
    NonFinite { first_index: u32, bits: u16 },
}

impl fmt::Display for GreedyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent { len } => {
                write!(
                    f,
                    "invalid BF16 logit extent {len}; expected 1..={MAX_VOCABULARY}"
                )
            }
            Self::NonFinite { first_index, bits } => write!(
                f,
                "nonfinite BF16 logit at index {first_index} (bits 0x{bits:04x})"
            ),
        }
    }
}

impl std::error::Error for GreedyError {}

/// Return the first maximum under direct BF16-to-f32 finite/comparison semantics.
///
/// Equal values retain the lowest input index, including `-0.0` versus `+0.0`.
/// The first NaN or infinity rejects selection and reports its original index.
pub fn greedy(logits: &[u16]) -> Result<Selection, GreedyError> {
    if logits.is_empty() || logits.len() > MAX_VOCABULARY {
        return Err(GreedyError::InvalidExtent { len: logits.len() });
    }

    let first_bits = logits[0];
    let mut best_value = f32::from_bits(u32::from(first_bits) << 16);
    if !best_value.is_finite() {
        return Err(GreedyError::NonFinite {
            first_index: 0,
            bits: first_bits,
        });
    }
    let mut selected = Selection {
        token: 0,
        bits: first_bits,
    };
    for (index, &bits) in logits.iter().enumerate().skip(1) {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(GreedyError::NonFinite {
                first_index: index as u32,
                bits,
            });
        }
        if value > best_value {
            selected = Selection {
                token: index as u32,
                bits,
            };
            best_value = value;
        }
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::{GreedyError, MAX_VOCABULARY, Selection, greedy};

    #[test]
    fn exhausts_every_single_bf16_representation() {
        for bits in 0..=u16::MAX {
            let value = f32::from_bits(u32::from(bits) << 16);
            let expected = if value.is_finite() {
                Ok(Selection { token: 0, bits })
            } else {
                Err(GreedyError::NonFinite {
                    first_index: 0,
                    bits,
                })
            };
            assert_eq!(greedy(&[bits]), expected, "BF16 bits 0x{bits:04x}");
        }
    }

    #[test]
    fn preserves_first_ties_including_signed_zero() {
        assert_eq!(
            greedy(&[0x8000, 0x0000]),
            Ok(Selection {
                token: 0,
                bits: 0x8000,
            })
        );
        assert_eq!(
            greedy(&[0x0000, 0x8000]),
            Ok(Selection {
                token: 0,
                bits: 0x0000,
            })
        );
        assert_eq!(
            greedy(&[0xbf80, 0xc000, 0xbf00]),
            Ok(Selection {
                token: 2,
                bits: 0xbf00,
            })
        );
    }

    #[test]
    fn handles_cross_tile_ties_and_vocabulary_tail() {
        let mut logits = vec![0xc000; 3 * 1024 + 5];
        logits[1023] = 0x4080;
        logits[1024] = 0x4080;
        let tail = logits.len() - 1;
        logits[tail] = 0x4080;
        assert_eq!(
            greedy(&logits),
            Ok(Selection {
                token: 1023,
                bits: 0x4080,
            })
        );

        logits[tail] = 0x4081;
        assert_eq!(
            greedy(&logits),
            Ok(Selection {
                token: (logits.len() - 1) as u32,
                bits: 0x4081,
            })
        );
    }

    #[test]
    fn supports_maximum_vocabulary() {
        let mut logits = vec![0x0000; MAX_VOCABULARY];
        logits[MAX_VOCABULARY - 1] = 0x7f7f;
        assert_eq!(
            greedy(&logits),
            Ok(Selection {
                token: (MAX_VOCABULARY - 1) as u32,
                bits: 0x7f7f,
            })
        );
    }

    #[test]
    fn reports_the_lowest_of_multiple_nonfinite_indices() {
        let mut logits = vec![0x3f80; 2 * 1024 + 7];
        logits[2049] = 0x7f80;
        logits[1023] = 0xffc1;
        logits[255] = 0x7f81;
        assert_eq!(
            greedy(&logits),
            Err(GreedyError::NonFinite {
                first_index: 255,
                bits: 0x7f81,
            })
        );
        assert_eq!(
            greedy(&[0x7fc0, 0xff80, 0x7f80]),
            Err(GreedyError::NonFinite {
                first_index: 0,
                bits: 0x7fc0,
            })
        );
    }

    #[test]
    fn rejects_invalid_extents() {
        assert_eq!(greedy(&[]), Err(GreedyError::InvalidExtent { len: 0 }));
        assert_eq!(
            greedy(&vec![0; MAX_VOCABULARY + 1]),
            Err(GreedyError::InvalidExtent {
                len: MAX_VOCABULARY + 1,
            })
        );
    }
}
