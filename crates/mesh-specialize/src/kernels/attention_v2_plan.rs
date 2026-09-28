//! Launch planning for the unqualified D256, 24:4 BF16 split-attention candidate.
use anyhow::{Result, ensure};

pub const QUERY_HEADS: usize = 24;
pub const KV_HEADS: usize = 4;
pub const WIDTH: usize = 256;
pub const MAX_SPLITS: usize = 85;
pub const PARTIAL_FLOATS: usize = WIDTH + 2;
pub const SCALE: f32 = 0.0625;

/// Do not silently substitute generic attention for an unsupported small-M shape.
pub fn validate_geometry(query_heads: usize, kv_heads: usize, width: usize) -> Result<()> {
    ensure!(
        (query_heads, kv_heads, width) == (QUERY_HEADS, KV_HEADS, WIDTH),
        "split-decode attention requires 24 query heads, 4 KV heads, and width 256"
    );
    Ok(())
}

/// Persistent stream allocation, independent of the row count of a later step.
pub fn persistent_workspace_bytes(max_rows: usize, capacity: usize) -> Result<usize> {
    ensure!(max_rows > 0, "split attention max rows must be positive");
    Ok(max_rows.min(8) * QUERY_HEADS * split_count(capacity)? * PARTIAL_FLOATS * 4)
}

/// Aim for 340 partial CTAs (two per 170 SMs), without splitting below ~64 keys.
/// This is a scheduling hypothesis, not a measured throughput claim.
pub fn split_count(length: usize) -> Result<usize> {
    ensure!((1..=262_144).contains(&length), "invalid attention length");
    Ok(length.div_ceil(64).min(MAX_SPLITS))
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub rows: usize,
    pub past: usize,
    pub capacity: usize,
    /// Workspace stride and launch upper bound; can exceed the active split count.
    pub split_slots: usize,
    pub workspace_bytes: usize,
    pub partial_grid: [u32; 3],
    pub reduce_grid: [u32; 3],
}

impl Plan {
    /// `max_length` is the replay interval's largest initialized length, not capacity.
    /// Device-position launches recompute active splits, but keep these addresses/grids.
    pub fn new(rows: usize, past: usize, capacity: usize, max_length: usize) -> Result<Self> {
        ensure!((1..=8).contains(&rows), "split attention supports M=1..8");
        ensure!((1..=262_144).contains(&capacity), "invalid KV capacity");
        let length = past
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("length overflow"))?;
        ensure!(
            length <= max_length && max_length <= capacity,
            "invalid replay interval"
        );
        let split_slots = split_count(max_length)?;
        let workspace_bytes = rows * QUERY_HEADS * split_slots * PARTIAL_FLOATS * 4;
        Ok(Self {
            rows,
            past,
            capacity,
            split_slots,
            workspace_bytes,
            partial_grid: [KV_HEADS as u32, split_slots as u32, 1],
            reduce_grid: [(rows * QUERY_HEADS) as u32, 1, 1],
        })
    }

    pub fn partial_kernel(self) -> &'static str {
        if self.rows == 1 {
            "attention_split_decode_bf16"
        } else {
            "attention_split_bf16"
        }
    }

    pub fn dimensions(self) -> [u32; 7] {
        [
            self.rows as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            WIDTH as u32,
            self.past as u32,
            self.capacity as u32,
            self.split_slots as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dispatch_rejects_generic_geometry_and_selects_decode_or_verify() {
        assert!(validate_geometry(24, 4, 256).is_ok());
        for (q, k, d) in [(12, 4, 256), (24, 8, 256), (24, 4, 128)] {
            assert!(validate_geometry(q, k, d).is_err());
        }
        for rows in 1..=8 {
            let plan = Plan::new(rows, 17, 8192, 8192).unwrap();
            assert_eq!(
                plan.partial_kernel(),
                if rows == 1 {
                    "attention_split_decode_bf16"
                } else {
                    "attention_split_bf16"
                }
            );
            assert!(plan.workspace_bytes <= persistent_workspace_bytes(512, 8192).unwrap());
        }
        assert_eq!(
            persistent_workspace_bytes(1, 8192).unwrap(),
            24 * 85 * 258 * 4
        );
        assert_eq!(
            persistent_workspace_bytes(512, 8192).unwrap(),
            8 * 24 * 85 * 258 * 4
        );
        assert_eq!(persistent_workspace_bytes(5, 63).unwrap(), 5 * 24 * 258 * 4);
        assert!(persistent_workspace_bytes(0, 63).is_err());
        assert!(persistent_workspace_bytes(1, 0).is_err());
    }

    #[test]
    fn split_boundaries_and_sm_target() {
        for (length, expected) in [
            (1, 1),
            (64, 1),
            (65, 2),
            (513, 9),
            (5440, 85),
            (8192, 85),
            (131072, 85),
        ] {
            assert_eq!(split_count(length).unwrap(), expected);
        }
        assert!(split_count(0).is_err());
        assert!(split_count(262145).is_err());
    }

    #[test]
    fn replay_stride_and_workspace() {
        let p = Plan::new(5, 8190, 9000, 8500).unwrap();
        assert_eq!(p.partial_grid, [4, 85, 1]);
        assert_eq!(p.reduce_grid, [120, 1, 1]);
        assert_eq!(p.workspace_bytes, 5 * 24 * 85 * 258 * 4);
        assert_eq!(p.dimensions(), [5, 24, 4, 256, 8190, 9000, 85]);
    }

    #[test]
    fn tiled_split_ranges_cover_visible_prefix_once() {
        for length in [1_usize, 2, 63, 64, 65, 513, 8195, 131072] {
            let splits = split_count(length).unwrap();
            let tiles = length.div_ceil(16);
            let mut cursor = 0;
            for split in 0..splits {
                let begin = tiles * split / splits * 16;
                let end = (tiles * (split + 1) / splits * 16).min(length);
                assert_eq!(begin, cursor);
                assert!(end > begin);
                cursor = end;
            }
            assert_eq!(cursor, length);
        }
    }

    #[test]
    fn rejects_bad_extents_and_intervals() {
        for (m, p, c, l) in [
            (0, 0, 1, 1),
            (9, 0, 9, 9),
            (1, usize::MAX, 1, 1),
            (5, 9, 13, 13),
            (1, 0, 4, 5),
            (1, 2, 4, 2),
        ] {
            assert!(Plan::new(m, p, c, l).is_err());
        }
    }
}
