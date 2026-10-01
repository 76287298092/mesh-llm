pub(super) use super::DensePattern;
use super::{OutputInitialization, ProjectionKind};
use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;
use anyhow::{Context as _, Result, ensure};

const FACTORS: [u16; 5] = [0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x4000];

impl DensePattern {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::AlternatingUnit => "dense-alternating-unit",
            Self::SignedMix => "dense-signed-mix",
        }
    }

    pub(super) fn input(self, k: usize, tokens: usize) -> Result<Vec<u16>> {
        let count = k
            .checked_mul(tokens)
            .context("Q8 dense input extent overflow")?;
        let mut input = Vec::new();
        input
            .try_reserve_exact(count)
            .context("Q8 dense input allocation failed")?;
        for token in 0..tokens {
            for column in 0..k {
                let word = match self {
                    Self::AlternatingUnit => {
                        let negative = ((((column / 32) >> token) & 1) ^ (column & 1)) != 0;
                        if negative { 0xbf80 } else { 0x3f80 }
                    }
                    Self::SignedMix => FACTORS[(column + token) % FACTORS.len()],
                };
                input.push(word);
            }
        }
        Ok(input)
    }

    pub(super) const fn initialization(repeat: usize) -> OutputInitialization {
        match repeat {
            0 => OutputInitialization::Poison(0x7fc1),
            _ => OutputInitialization::Poison(0xffc2),
        }
    }
}

pub(super) fn input_value(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

pub(super) fn identity_view(carrier: &Q8MatrixView, kind: ProjectionKind) -> Result<Q8MatrixView> {
    let [_, k] = kind.dimensions();
    let parent_rows = kind.parent_rows();
    let code_bytes = parent_rows
        .checked_mul(k)
        .context("identity Q8 codes extent overflow")?;
    let scale_count = parent_rows
        .checked_mul(k / 32)
        .context("identity Q8 scale count overflow")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("identity Q8 scale extent overflow")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("identity Q8 scale offset overflow")?;
    ensure!(
        carrier.codes.offset == 0 && carrier.codes.bytes == u64::try_from(code_bytes)?,
        "identity Q8 view needs the complete parent code plane"
    );
    ensure!(
        carrier.scale_bits.offset == u64::try_from(scale_offset)?
            && carrier.scale_bits.bytes == u64::try_from(scale_bytes)?
            && carrier.scale_count == scale_count,
        "identity Q8 view needs the full parent scale plane"
    );
    Ok(Q8MatrixView {
        object_id: carrier.object_id.clone(),
        shape: [parent_rows, k],
        padded_k: k,
        group_size: 32,
        codes: carrier.codes.clone(),
        scale_bits: carrier.scale_bits.clone(),
        scale_count,
        source_rows: (0..parent_rows).collect(),
    })
}
