//! Streaming extraction of selected BF16 embedding rows from a verified artifact.

use anyhow::{Context, Result, ensure};
use std::io::{self, Write};

use crate::artifact::schema::{DType, ObjectKind};

/// Copy requested rows from one resident `[vocab, width]` BF16 tensor.
///
/// The artifact is streamed in full so `VerifiedArtifact` can recheck its object
/// digest. Only the selected rows are retained, in token order, including repeats.
pub fn embedding_rows(
    artifact: &mut crate::artifact::reader::VerifiedArtifact,
    name: &str,
    width: usize,
    tokens: &[u32],
) -> Result<Vec<u8>> {
    ensure!(
        (1..=32768).contains(&width),
        "embedding width is out of range"
    );
    ensure!(
        (1..=2048).contains(&tokens.len()),
        "embedding token count is out of range"
    );
    let width_u64 = u64::try_from(width).context("embedding width does not fit u64")?;
    let (vocab, object_length) = {
        let objects = &artifact.directory().objects;
        let object_index = objects
            .binary_search_by(|object| object.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("embedding tensor not found: {name}"))?;
        let object = &objects[object_index];
        ensure!(
            object.kind == ObjectKind::Tensor && object.dtype == DType::Bf16,
            "embedding object must be a BF16 tensor"
        );
        ensure!(
            object.shape.len() == 2,
            "embedding tensor must have rank two"
        );
        ensure!(
            object.shape[1] == width_u64,
            "embedding tensor width does not match the requested width"
        );
        let vocab = object.shape[0];
        ensure!(vocab > 0, "embedding vocabulary is empty");
        let expected_length = vocab
            .checked_mul(width_u64)
            .and_then(|elements| elements.checked_mul(2))
            .context("embedding tensor byte length overflows u64")?;
        ensure!(
            object.length == expected_length,
            "embedding tensor length does not match its BF16 shape"
        );
        (vocab, object.length)
    };
    ensure!(
        tokens.iter().all(|&token| u64::from(token) < vocab),
        "embedding token index is out of range"
    );

    let mut collector = RowCollector::new(object_length, width, tokens)?;
    let copied_length = artifact.copy_object(name, &mut collector)?;
    ensure!(
        copied_length == object_length,
        "embedding object stream length mismatch"
    );
    collector.finish()
}

#[derive(Clone, Copy)]
struct RowCopy {
    source_start: u64,
    source_end: u64,
    destination_start: usize,
}

struct RowCollector {
    object_length: u64,
    stream_offset: u64,
    copies: Vec<RowCopy>,
    next_copy: usize,
    output: Vec<u8>,
}

impl RowCollector {
    fn new(object_length: u64, width: usize, tokens: &[u32]) -> Result<Self> {
        let row_bytes = width
            .checked_mul(2)
            .context("embedding row byte length overflows usize")?;
        let row_bytes_u64 =
            u64::try_from(row_bytes).context("embedding row byte length does not fit u64")?;
        let output_length = tokens
            .len()
            .checked_mul(row_bytes)
            .context("selected embedding rows overflow usize")?;
        let mut copies = Vec::with_capacity(tokens.len());
        for (index, &token) in tokens.iter().enumerate() {
            let source_start = u64::from(token)
                .checked_mul(row_bytes_u64)
                .context("embedding source row offset overflows u64")?;
            let source_end = source_start
                .checked_add(row_bytes_u64)
                .context("embedding source row end overflows u64")?;
            ensure!(
                source_end <= object_length,
                "selected embedding row exceeds the object"
            );
            let destination_start = index
                .checked_mul(row_bytes)
                .context("embedding destination row offset overflows usize")?;
            copies.push(RowCopy {
                source_start,
                source_end,
                destination_start,
            });
        }
        copies.sort_unstable_by_key(|copy| copy.source_start);
        Ok(Self {
            object_length,
            stream_offset: 0,
            copies,
            next_copy: 0,
            output: vec![0; output_length],
        })
    }

    fn capture(&mut self, bytes: &[u8]) -> Result<()> {
        let chunk_length = u64::try_from(bytes.len()).context("artifact chunk does not fit u64")?;
        let chunk_end = self
            .stream_offset
            .checked_add(chunk_length)
            .context("embedding object stream offset overflows u64")?;
        ensure!(
            chunk_end <= self.object_length,
            "embedding object stream exceeds its declared length"
        );

        let mut copy_index = self.next_copy;
        while let Some(copy) = self.copies.get(copy_index).copied() {
            if copy.source_start >= chunk_end {
                break;
            }
            let start = copy.source_start.max(self.stream_offset);
            let end = copy.source_end.min(chunk_end);
            if start < end {
                let source_start = usize::try_from(start - self.stream_offset)
                    .context("embedding chunk source offset does not fit usize")?;
                let destination_start = copy
                    .destination_start
                    .checked_add(
                        usize::try_from(start - copy.source_start)
                            .context("embedding destination offset does not fit usize")?,
                    )
                    .context("embedding destination range overflows usize")?;
                let length = usize::try_from(end - start)
                    .context("embedding copy length does not fit usize")?;
                let source_end = source_start
                    .checked_add(length)
                    .context("embedding source range overflows usize")?;
                let destination_end = destination_start
                    .checked_add(length)
                    .context("embedding destination range overflows usize")?;
                ensure!(
                    source_end <= bytes.len(),
                    "embedding source chunk range is invalid"
                );
                ensure!(
                    destination_end <= self.output.len(),
                    "embedding destination range is invalid"
                );
                self.output[destination_start..destination_end]
                    .copy_from_slice(&bytes[source_start..source_end]);
            }
            copy_index += 1;
        }
        while self
            .copies
            .get(self.next_copy)
            .is_some_and(|copy| copy.source_end <= chunk_end)
        {
            self.next_copy += 1;
        }
        self.stream_offset = chunk_end;
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>> {
        ensure!(
            self.stream_offset == self.object_length,
            "embedding object stream ended before its declared length"
        );
        ensure!(
            self.next_copy == self.copies.len(),
            "embedding object stream did not include every selected row"
        );
        Ok(self.output)
    }
}

impl Write for RowCollector {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.capture(bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RowCollector;
    use std::io::Write;

    #[test]
    fn copies_out_of_order_and_repeated_rows_across_stream_chunks() {
        let source = (0_u8..28).collect::<Vec<_>>();
        let source_length = u64::try_from(source.len()).unwrap();
        let mut collector = RowCollector::new(source_length, 2, &[5, 1, 5, 3]).unwrap();
        for chunk in source.chunks(3) {
            collector.write_all(chunk).unwrap();
        }
        let expected = [
            &source[20..24],
            &source[4..8],
            &source[20..24],
            &source[12..16],
        ]
        .concat();
        assert_eq!(collector.finish().unwrap(), expected);
    }

    #[test]
    fn repeated_rows_each_receive_every_piece_when_a_row_spans_chunks() {
        let source = (0_u8..12).collect::<Vec<_>>();
        let mut collector = RowCollector::new(12, 2, &[1, 1]).unwrap();
        for chunk in source.chunks(6) {
            collector.write_all(chunk).unwrap();
        }
        let expected = [&source[4..8], &source[4..8]].concat();
        assert_eq!(collector.finish().unwrap(), expected);
    }

    #[test]
    fn rejects_stream_overrun_and_unfinished_object() {
        let mut collector = RowCollector::new(4, 1, &[0]).unwrap();
        assert!(collector.write_all(&[1, 2, 3, 4, 5]).is_err());
        assert!(collector.finish().is_err());
    }

    #[test]
    fn rejects_stream_offset_overflow() {
        let mut collector = RowCollector::new(u64::MAX, 1, &[0]).unwrap();
        collector.stream_offset = u64::MAX;
        assert!(collector.write(&[1]).is_err());
    }

    #[test]
    fn requires_every_declared_object_byte_before_finishing() {
        let mut collector = RowCollector::new(8, 1, &[3]).unwrap();
        collector.write_all(&[0; 6]).unwrap();
        assert!(collector.finish().is_err());
    }
}
