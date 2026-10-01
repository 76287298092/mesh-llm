use super::super::resident_model::Session;
pub(in crate::kernels::cuda) use super::state::{StateComparison, compare_states};
use anyhow::{Context as _, Result};
use serde::Serialize;
#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct WordComparison {
    pub left_words: usize,
    pub right_words: usize,
    pub compared_words: usize,
    pub differing_words: usize,
    pub expected_words: Option<usize>,
    pub passed: bool,
    pub first_mismatch_index: Option<usize>,
    pub first_left_word: Option<u16>,
    pub first_right_word: Option<u16>,
}

impl Default for WordComparison {
    fn default() -> Self {
        Self::new()
    }
}

impl WordComparison {
    pub(in crate::kernels::cuda) const fn new() -> Self {
        Self {
            left_words: 0,
            right_words: 0,
            compared_words: 0,
            differing_words: 0,
            expected_words: None,
            passed: false,
            first_mismatch_index: None,
            first_left_word: None,
            first_right_word: None,
        }
    }

    pub(in crate::kernels::cuda) fn add(
        &mut self,
        left: &[u16],
        right: &[u16],
        expected_words: usize,
    ) -> Result<()> {
        let index_base = self.compared_words;
        self.left_words = self
            .left_words
            .checked_add(left.len())
            .context("left word count overflows usize")?;
        self.right_words = self
            .right_words
            .checked_add(right.len())
            .context("right word count overflows usize")?;
        self.compared_words = self
            .compared_words
            .checked_add(left.len().min(right.len()))
            .context("compared word count overflows usize")?;
        for (index, (left_word, right_word)) in left.iter().zip(right).enumerate() {
            if left_word != right_word {
                self.differing_words = self
                    .differing_words
                    .checked_add(1)
                    .context("differing word count overflows usize")?;
                self.first_mismatch_index.get_or_insert(
                    index_base
                        .checked_add(index)
                        .context("word mismatch index overflows usize")?,
                );
                if self.first_left_word.is_none() {
                    self.first_left_word = Some(*left_word);
                    self.first_right_word = Some(*right_word);
                }
            }
        }
        let shared = left.len().min(right.len());
        let unmatched = left.len().abs_diff(right.len());
        self.differing_words = self
            .differing_words
            .checked_add(unmatched)
            .context("differing word count overflows usize")?;
        if unmatched > 0 && self.first_mismatch_index.is_none() {
            self.first_mismatch_index = Some(
                index_base
                    .checked_add(shared)
                    .context("word mismatch index overflows usize")?,
            );
        }
        self.expected_words = Some(expected_words);
        self.passed = self.left_words == expected_words
            && self.right_words == expected_words
            && self.differing_words == 0;
        if !self.passed && self.first_mismatch_index.is_none() {
            self.first_mismatch_index = Some(self.left_words.min(self.right_words));
        }
        Ok(())
    }

    pub(in crate::kernels::cuda) fn passed(&self) -> bool {
        self.passed
    }
}

pub(in crate::kernels::cuda) fn outputs_equal(
    layers: &[LayerComparison],
    hidden: &WordComparison,
    logits: &WordComparison,
    state: Option<&StateComparison>,
    cursor: Option<&CursorComparison>,
) -> bool {
    hidden.passed()
        && logits.passed()
        && layers.iter().all(|layer| layer.words.passed())
        && state.is_some_and(|comparison| comparison.equal)
        && cursor.is_some_and(|comparison| comparison.equal)
}

pub(in crate::kernels::cuda) fn validate_layer_outputs(
    layers: &[LayerComparison],
    expected_layers: usize,
) -> bool {
    layers.len() == expected_layers
        && layers
            .iter()
            .enumerate()
            .all(|(index, layer)| layer.layer == index && layer.words.passed())
}

pub(in crate::kernels::cuda) fn first_differing_layer(layers: &[LayerComparison]) -> Option<usize> {
    layers
        .iter()
        .find(|layer| !layer.words.passed())
        .map(|layer| layer.layer)
}

#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct LayerComparison {
    pub layer: usize,
    pub words: WordComparison,
}

#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct CursorSnapshot {
    pub past: usize,
    pub capacity: usize,
    pub poisoned: bool,
}

#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct CursorComparison {
    pub left: CursorSnapshot,
    pub right: CursorSnapshot,
    pub equal: bool,
}

pub(in crate::kernels::cuda) fn compare_cursors(
    sessions: (&Session<'_>, &Session<'_>),
    expected: CursorSnapshot,
) -> CursorComparison {
    compare_cursor_snapshots(snapshot(sessions.0), snapshot(sessions.1), expected)
}

pub(in crate::kernels::cuda) fn snapshot(session: &Session<'_>) -> CursorSnapshot {
    CursorSnapshot {
        past: session.cursor.past(),
        capacity: session.cursor.capacity(),
        poisoned: session.cursor.is_poisoned(),
    }
}

pub(in crate::kernels::cuda) fn compare_cursor_snapshots(
    left: CursorSnapshot,
    right: CursorSnapshot,
    expected: CursorSnapshot,
) -> CursorComparison {
    let equal = expected.past <= expected.capacity
        && left.past == expected.past
        && right.past == expected.past
        && left.capacity == expected.capacity
        && right.capacity == expected.capacity
        && !left.poisoned
        && !right.poisoned;
    CursorComparison { left, right, equal }
}

pub(in crate::kernels::cuda) fn reports_all_five_cases(cases: &[bool]) -> bool {
    cases.len() == 5 && cases.iter().all(|&passed| passed)
}
#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
