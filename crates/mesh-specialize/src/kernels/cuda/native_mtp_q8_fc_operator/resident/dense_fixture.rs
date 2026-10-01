#[cfg(test)]
mod tests;

use super::super::{FACTOR_BITS, FC_GROUP, FC_K};
use anyhow::{Context as _, Result};

const GROUPS: usize = FC_K / FC_GROUP;

// Both fixtures use BF16-exact half-integers with |activation| <= 2. A G32
// dot's absolute sum is at most 32 * 128 * 2 = 8192, exact in FP32.
#[derive(Clone, Copy)]
pub(super) enum DensePattern {
    AlternatingUnit,
    SignedDyadic,
}

impl DensePattern {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::AlternatingUnit => "dense-alternating-unit",
            Self::SignedDyadic => "dense-signed-dyadic",
        }
    }
}

pub(super) fn input(pattern: DensePattern, tokens: usize) -> Result<Vec<u16>> {
    let count = tokens
        .checked_mul(FC_K)
        .context("dense FC input extent overflow")?;
    let mut input = Vec::new();
    input
        .try_reserve_exact(count)
        .context("dense FC input allocation failed")?;
    for token in 0..tokens {
        for group in 0..GROUPS {
            for lane in 0..FC_GROUP {
                let word = match pattern {
                    DensePattern::AlternatingUnit => {
                        let negative = ((group >> token) & 1) ^ (lane & 1) != 0;
                        if negative { 0xbf80 } else { 0x3f80 }
                    }
                    DensePattern::SignedDyadic => {
                        let k = group * FC_GROUP + lane;
                        FACTOR_BITS[(k + token) % FACTOR_BITS.len()]
                    }
                };
                input.push(word);
            }
        }
    }
    Ok(input)
}

pub(super) fn columns_are_pairwise_distinct(words: &[u16], rows: usize) -> bool {
    if rows == 0 || !words.len().is_multiple_of(rows) {
        return false;
    }
    for (index, column) in words.chunks_exact(rows).enumerate() {
        if words
            .chunks_exact(rows)
            .skip(index + 1)
            .any(|other| other == column)
        {
            return false;
        }
    }
    true
}
