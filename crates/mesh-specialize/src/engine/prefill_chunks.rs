//! Pure, bounded prefill partition and fixed-output cursor planning.

use anyhow::{Context, Result, ensure};
use serde::Serialize;

pub const MAX_PROMPT_TOKENS: usize = 32_768;
pub const MAX_CHUNK_ROWS: usize = 512;

/// Half-open prompt row range. Past is always `start`, never reset per chunk.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

impl Chunk {
    pub fn rows(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Plan {
    pub prompt_tokens: usize,
    pub chunk_size: usize,
    pub max_rows: usize,
    pub output_tokens: usize,
    pub capacity: usize,
    pub chunks: Vec<Chunk>,
}

impl Plan {
    /// First output comes from final prefill; remaining outputs each consume one row.
    pub fn new(prompt_tokens: usize, chunk_size: usize, output_tokens: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_PROMPT_TOKENS).contains(&prompt_tokens),
            "prompt length must be 1..=32768"
        );
        ensure!(
            (1..=MAX_CHUNK_ROWS).contains(&chunk_size),
            "chunk size must be 1..=512"
        );
        ensure!(
            (2..=512).contains(&output_tokens),
            "output tokens must be 2..=512"
        );
        let capacity = checked_end(prompt_tokens, output_tokens - 1, usize::MAX)?;
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < prompt_tokens {
            let rows = chunk_size.min(prompt_tokens - start);
            let end = checked_end(start, rows, prompt_tokens)?;
            chunks.push(Chunk { start, end });
            start = end;
        }
        Ok(Self {
            prompt_tokens,
            chunk_size,
            max_rows: chunk_size.min(prompt_tokens),
            output_tokens,
            capacity,
            chunks,
        })
    }

    pub fn validate_capacity(&self, capacity: usize) -> Result<()> {
        ensure!(
            self.capacity == capacity,
            "configured capacity must equal prompt + outputs - 1"
        );
        Ok(())
    }
}

/// Check a nonempty cursor advance without wrapping, before indexing or executing.
pub fn checked_end(past: usize, rows: usize, capacity: usize) -> Result<usize> {
    ensure!(rows > 0, "cursor advance must contain at least one row");
    let end = past
        .checked_add(rows)
        .context("cursor advance overflows usize")?;
    ensure!(end <= capacity, "cursor advance exceeds capacity");
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::{Plan, checked_end};

    #[test]
    fn requested_partition_matrix_covers_every_row_once() {
        for length in [1, 511, 512, 513, 8192] {
            for chunk_size in [1, 128, 512] {
                let plan = Plan::new(length, chunk_size, 2).unwrap();
                assert_eq!(plan.capacity, length + 1);
                assert_eq!(plan.max_rows, length.min(chunk_size));
                assert_eq!(plan.chunks.len(), length.div_ceil(chunk_size));
                let mut past = 0;
                for chunk in &plan.chunks {
                    assert_eq!(chunk.start, past);
                    assert!((1..=chunk_size).contains(&chunk.rows()));
                    past = checked_end(past, chunk.rows(), plan.capacity).unwrap();
                    assert_eq!(chunk.end, past);
                }
                assert_eq!(past, length);
                assert_eq!(checked_end(past, 1, plan.capacity).unwrap(), plan.capacity);
            }
        }
    }

    #[test]
    fn bounds_and_capacity_are_not_silently_relaxed() {
        for chunk in [0, 513, usize::MAX] {
            assert!(Plan::new(512, chunk, 2).is_err());
        }
        for length in [0, 32_769, usize::MAX] {
            assert!(Plan::new(length, 128, 2).is_err());
        }
        for outputs in [0, 1, 513, usize::MAX] {
            assert!(Plan::new(1, 1, outputs).is_err());
        }
        let plan = Plan::new(32_768, 512, 512).unwrap();
        assert_eq!(plan.capacity, 33_279);
        assert!(plan.validate_capacity(33_279).is_ok());
        assert!(plan.validate_capacity(33_278).is_err());
        assert!(plan.validate_capacity(33_280).is_err());
    }

    #[test]
    fn cursor_checks_overflow_and_exhaustion() {
        assert!(checked_end(usize::MAX, 1, usize::MAX).is_err());
        assert!(checked_end(1, usize::MAX, usize::MAX).is_err());
        assert!(checked_end(0, 0, 1).is_err());
        assert!(checked_end(512, 2, 513).is_err());
        assert_eq!(
            checked_end(usize::MAX - 1, 1, usize::MAX).unwrap(),
            usize::MAX
        );
    }
}
