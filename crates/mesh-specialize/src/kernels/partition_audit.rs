//! Diagnostic row fingerprints, independent of how rows are submitted to the GPU.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(super) struct Rows {
    row_bytes: usize,
    expected_rows: usize,
    layers: Vec<Vec<String>>,
}
impl Rows {
    pub(super) fn new(layers: usize, rows: usize, row_bytes: usize) -> Result<Self> {
        ensure!(
            (1..=256).contains(&layers) && (1..=512).contains(&rows),
            "invalid partition audit geometry"
        );
        ensure!(
            row_bytes > 0 && row_bytes <= 65536 && row_bytes.is_multiple_of(2),
            "invalid BF16 audit row width"
        );
        Ok(Self {
            row_bytes,
            expected_rows: rows,
            layers: vec![Vec::new(); layers],
        })
    }
    pub(super) fn record(&mut self, layer: usize, bytes: &[u8]) -> Result<()> {
        ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(self.row_bytes),
            "partition audit requires complete rows"
        );
        let rows = bytes.len() / self.row_bytes;
        ensure!(
            layer < self.layers.len() && rows <= self.expected_rows - self.layers[layer].len(),
            "partition audit exceeds planned extent"
        );
        self.layers[layer].extend(
            bytes
                .chunks_exact(self.row_bytes)
                .map(|row| hex::encode(Sha256::digest(row))),
        );
        Ok(())
    }
    pub(super) fn compare(&self, other: &Self) -> Result<Value> {
        ensure!(
            self.row_bytes == other.row_bytes
                && self.expected_rows == other.expected_rows
                && self.layers.len() == other.layers.len(),
            "partition audit geometry mismatch"
        );
        let mut differences = Vec::new();
        for (layer, (a, b)) in self.layers.iter().zip(&other.layers).enumerate() {
            ensure!(
                a.len() == self.expected_rows && b.len() == self.expected_rows,
                "partition audit is incomplete"
            );
            let rows = a
                .iter()
                .zip(b)
                .enumerate()
                .filter(|(_, (left, right))| left != right)
                .map(|(row, _)| row)
                .collect::<Vec<_>>();
            if let Some(&first) = rows.first() {
                differences.push(json!({"layer":layer,"differing_rows":rows,"first_row":first,"whole_sha256":a[first],"token_sha256":b[first]}));
            }
        }
        Ok(
            json!({"diagnostic_only":true,"all_row_hashes_equal":differences.is_empty(),"layers":self.layers.len(),"rows":self.expected_rows,"row_bytes":self.row_bytes,"differences":differences,"scope":"BF16 layer outputs, SHA256 per row; first layer with any differing row precedes later-layer drift"}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_independent_and_localizes_earlier_rows() {
        let mut whole = Rows::new(2, 3, 2).unwrap();
        let mut token = Rows::new(2, 3, 2).unwrap();
        for layer in 0..2 {
            whole.record(layer, &[1, 0, 2, 0, 3, 0]).unwrap();
        }
        for value in [1, 2, 3] {
            for layer in 0..2 {
                token.record(layer, &[value, 0]).unwrap();
            }
        }
        assert_eq!(whole.compare(&token).unwrap()["all_row_hashes_equal"], true);
        token.layers[0][1] = "different".into();
        let result = whole.compare(&token).unwrap();
        assert_eq!(result["differences"][0]["layer"], 0);
        assert_eq!(result["differences"][0]["differing_rows"], json!([1]));
    }
    #[test]
    fn rejects_incomplete_and_overlapping_capture() {
        let mut a = Rows::new(1, 1, 2).unwrap();
        let b = Rows::new(1, 1, 2).unwrap();
        assert!(a.compare(&b).is_err());
        assert!(a.record(0, &[1]).is_err());
        assert!(a.record(1, &[1, 0]).is_err());
        a.record(0, &[1, 0]).unwrap();
        assert!(a.record(0, &[2, 0]).is_err());
    }
}
