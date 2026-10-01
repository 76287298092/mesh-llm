use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q8MatrixView};
use anyhow::{Context as _, Result};

pub(super) const ROWS: usize = 5120;
pub(super) const K: usize = 10240;
pub(super) const GROUPS: usize = K / 32;
const SCALES: [u16; 4] = [0x3400, 0x3800, 0x3c00, 0x4000];
const FACTORS: [f32; 5] = [1.0, -1.0, 2.0, -2.0, 0.5];

#[derive(Clone, Copy)]
pub(super) enum Candidate {
    C4,
    C8,
}

impl Candidate {
    pub(super) const fn entry(self) -> &'static str {
        match self {
            Self::C4 => "native_mtp_q8_sliced_k_fc_c4",
            Self::C8 => "native_mtp_q8_sliced_k_fc_c8",
        }
    }
    pub(super) const fn tokens(self) -> usize {
        match self {
            Self::C4 => 1,
            Self::C8 => 5,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Dense,
    LastK,
    Cancellation,
}

impl Kind {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Dense => "signed-dense-dyadic",
            Self::LastK => "last-k-all-rows",
            Self::Cancellation => "dense-pair-cancellation-all-rows",
        }
    }
}

pub(super) struct Fixture {
    pub(super) candidate: Candidate,
    pub(super) kind: Kind,
    pub(super) object: Vec<u8>,
    pub(super) input: Vec<u16>,
    pub(super) view: Q8MatrixView,
    pub(super) code_bytes: usize,
}

impl Fixture {
    pub(super) fn new(candidate: Candidate, kind: Kind) -> Result<Self> {
        let code_bytes = ROWS.checked_mul(K).context("FC code extent overflow")?;
        let scale_count = ROWS
            .checked_mul(GROUPS)
            .context("FC scale count overflow")?;
        let scale_bytes = scale_count
            .checked_mul(2)
            .context("FC scale extent overflow")?;
        let extent = code_bytes
            .checked_add(scale_bytes)
            .context("FC object extent overflow")?;
        let mut object = filled(extent, 0_u8)?;
        populate_planes(&mut object, code_bytes, kind)?;
        let count = candidate
            .tokens()
            .checked_mul(K)
            .context("FC input extent overflow")?;
        let mut input = filled(count, 0_u16)?;
        populate_input(&mut input, kind)?;
        let source_rows = vec![
            0, 1, 7, 8, 15, 16, 17, 31, 32, 2559, 2560, 5103, 5104, 5118, 5119,
        ];
        let view = Q8MatrixView {
            object_id: "synthetic-q8-fc".into(),
            shape: [source_rows.len(), K],
            padded_k: K,
            group_size: 32,
            codes: BytePlane {
                offset: 0,
                bytes: u64::try_from(code_bytes)?,
            },
            scale_bits: BytePlane {
                offset: u64::try_from(code_bytes)?,
                bytes: u64::try_from(scale_bytes)?,
            },
            scale_count,
            source_rows,
        };
        Ok(Self {
            candidate,
            kind,
            object,
            input,
            view,
            code_bytes,
        })
    }

    pub(super) fn simple_expected(&self, token: usize, row: usize) -> Result<Option<u16>> {
        match self.kind {
            Kind::Dense => Ok(None),
            Kind::LastK | Kind::Cancellation => {
                let scales = [0.25_f32, 0.5, 1.0, 2.0];
                let value =
                    f32::from(row_code(row)?) * scales[(row + GROUPS - 1) % 4] * FACTORS[token];
                Ok(Some(round_bf16(value)))
            }
        }
    }
}

fn populate_planes(object: &mut [u8], code_bytes: usize, kind: Kind) -> Result<()> {
    for row in 0..ROWS {
        for column in 0..K {
            let code = match kind {
                Kind::Dense => match column {
                    0 => -128,
                    1 => 127,
                    _ => i8::try_from((row * 3 + column * 5 + column / 32) % 14)? - 7,
                },
                Kind::LastK => {
                    if column == K - 1 {
                        row_code(row)?
                    } else {
                        0
                    }
                }
                Kind::Cancellation => row_code(row)?,
            };
            object[row * K + column] = code.to_ne_bytes()[0];
        }
        for group in 0..GROUPS {
            let offset = code_bytes + (row * GROUPS + group) * 2;
            object[offset..offset + 2].copy_from_slice(&SCALES[(row + group) % 4].to_le_bytes());
        }
    }
    Ok(())
}

fn populate_input(input: &mut [u16], kind: Kind) -> Result<()> {
    for (token, column) in input.as_chunks_mut::<K>().0.iter_mut().enumerate() {
        for (index, word) in column.iter_mut().enumerate() {
            let base = match kind {
                Kind::Dense => {
                    if (index + index / 32) % 3 == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                Kind::LastK => {
                    if index == K - 1 {
                        1.0
                    } else {
                        0.0
                    }
                }
                Kind::Cancellation => {
                    if index == K - 1 {
                        0.0
                    } else if index % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
            *word = u16::try_from((base * FACTORS[token]).to_bits() >> 16)?;
        }
    }
    Ok(())
}

fn row_code(row: usize) -> Result<i8> {
    let magnitude = i8::try_from(row % 7 + 1)?;
    Ok(if row.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    })
}

pub(super) fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let bytes = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)).to_le_bytes();
    u16::from_le_bytes([bytes[2], bytes[3]])
}

pub(super) fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .context("FC host allocation failed")?;
    result.resize(count, value);
    Ok(result)
}
