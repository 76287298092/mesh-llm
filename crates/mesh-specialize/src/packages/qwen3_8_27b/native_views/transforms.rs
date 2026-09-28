//! Bit-preserving layout operations with independently indexed inverse checks.
use super::Transform;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

pub(super) fn forward(kind: &Transform, source: &[u8]) -> Result<Vec<u8>> {
    match kind {
        Transform::Copy => Ok(source.to_vec()),
        Transform::Rows { row_bytes, order } => {
            ensure!(
                *row_bytes > 0 && source.len().is_multiple_of(*row_bytes),
                "invalid row transform extent"
            );
            ensure!(
                order.iter().copied().collect::<BTreeSet<_>>().len() == order.len(),
                "duplicate row in native permutation"
            );
            let bytes = row_bytes
                .checked_mul(order.len())
                .context("row transform overflow")?;
            let mut out = Vec::with_capacity(bytes);
            for &row in order {
                let start = row
                    .checked_mul(*row_bytes)
                    .context("row address overflow")?;
                let end = start
                    .checked_add(*row_bytes)
                    .context("row extent overflow")?;
                out.extend_from_slice(
                    source
                        .get(start..end)
                        .context("row outside native parent")?,
                );
            }
            Ok(out)
        }
        Transform::Conv { channels } => {
            ensure!(
                channels.checked_mul(8) == Some(source.len()),
                "invalid convolution extent"
            );
            let mut out = vec![0; source.len()];
            for c in 0..*channels {
                for tap in 0..4 {
                    let from = (tap * channels + c) * 2;
                    let to = (c * 4 + tap) * 2;
                    out[to..to + 2].copy_from_slice(&source[from..from + 2]);
                }
            }
            Ok(out)
        }
        Transform::NvScales { rows, width } => {
            ensure!(
                *rows > 0 && rows.is_multiple_of(128) && *width > 0 && width.is_multiple_of(64),
                "invalid scale tile geometry"
            );
            let groups = width / 16;
            ensure!(
                rows.checked_mul(groups) == Some(source.len()),
                "invalid scale transform extent"
            );
            let mut out = vec![0; source.len()];
            for row in 0..*rows {
                for g in 0..groups {
                    let index = ((row / 128) * (width / 64) + g / 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) / 32) * 4
                        + g % 4;
                    out[row * groups + g] = source[index];
                }
            }
            Ok(out)
        }
    }
}

pub(super) fn verify_inverse(kind: &Transform, source: &[u8], output: &[u8]) -> Result<()> {
    match kind {
        Transform::Copy => ensure!(source == output, "native copied bytes changed"),
        Transform::Rows { row_bytes, order } => {
            ensure!(
                order.len().checked_mul(*row_bytes) == Some(output.len()),
                "inverse row size mismatch"
            );
            for (i, &row) in order.iter().enumerate() {
                ensure!(
                    source.get(row * row_bytes..(row + 1) * row_bytes)
                        == Some(&output[i * row_bytes..(i + 1) * row_bytes]),
                    "inverse row permutation mismatch"
                );
            }
            // Unselected parent rows remain in the source artifact and in other views.
        }
        Transform::Conv { channels } => {
            ensure!(source.len() == output.len(), "inverse conv extent mismatch");
            for index in 0..source.len() / 2 {
                let tap = index / channels;
                let channel = index % channels;
                let index_out = (channel * 4 + tap) * 2;
                ensure!(
                    source[index * 2..index * 2 + 2] == output[index_out..index_out + 2],
                    "inverse convolution mismatch"
                );
            }
        }
        Transform::NvScales { rows: _, width } => {
            ensure!(
                source.len() == output.len(),
                "inverse scale extent mismatch"
            );
            for (j, &byte) in source.iter().enumerate() {
                let tile = j / 512;
                let lane = j % 512;
                let row = 128 * (tile / (width / 64)) + lane / 16 + 32 * ((lane % 16) / 4);
                let group = 4 * (tile % (width / 64)) + lane % 4;
                ensure!(
                    output[row * (width / 16) + group] == byte,
                    "inverse NVFP4 scale permutation mismatch"
                );
            }
        }
    }
    Ok(())
}
