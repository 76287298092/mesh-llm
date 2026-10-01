use anyhow::{Context as _, Result};

pub(super) struct DenseInput {
    pub(super) name: &'static str,
    pub(super) words: Vec<u16>,
}

#[derive(Clone, Copy)]
enum Pattern {
    Alternating,
    Dyadic,
}

impl Pattern {
    const fn name(self) -> &'static str {
        match self {
            Self::Alternating => "alternating-plus-minus-one",
            Self::Dyadic => "dense-signed-dyadic",
        }
    }

    const fn word(self, lane: usize) -> u16 {
        match self {
            Self::Alternating => match lane % 2 {
                0 => 0x3f80,
                _ => 0xbf80,
            },
            Self::Dyadic => match lane % 5 {
                0 => 0x3f00,
                1 => 0xbf00,
                2 => 0x3f80,
                3 => 0xbf80,
                _ => 0x4000,
            },
        }
    }
}

pub(super) fn dense_cases() -> Result<[DenseInput; 2]> {
    Ok([build(Pattern::Alternating)?, build(Pattern::Dyadic)?])
}

fn build(pattern: Pattern) -> Result<DenseInput> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(5_120)
        .context("cannot reserve native Q4 dense activation")?;
    words.extend((0..5_120).map(|lane| pattern.word(lane)));
    Ok(DenseInput {
        name: pattern.name(),
        words,
    })
}

#[cfg(test)]
mod tests {
    use super::dense_cases;

    #[test]
    fn activation_cases_are_dense_and_cover_distinct_signed_values() {
        let [alternating, dyadic] = dense_cases().expect("dense Q4 activations");

        assert_eq!(alternating.words.len(), 5_120);
        assert_eq!(dyadic.words.len(), 5_120);
        assert!(alternating.words.iter().all(|word| *word != 0));
        assert!(dyadic.words.iter().all(|word| *word != 0));
        assert_eq!(&alternating.words[..4], [0x3f80, 0xbf80, 0x3f80, 0xbf80]);
        assert_eq!(&dyadic.words[..5], [0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x4000]);
    }
}
