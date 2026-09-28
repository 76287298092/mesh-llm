//! Fixed-geometry admission for the unqualified exact-order M=1 attention schedule.
use anyhow::{Result, ensure};

pub const KERNEL: &str = "causal_attention_warp_fp64";
pub const GRID: [u32; 3] = [24 * 4, 1, 1];
pub const BLOCK: [u32; 3] = [32, 1, 1];

/// The unchanged control ABI's six dimensions, including a by-value position.
/// Graph execution is intentionally unsupported until a position variant is qualified.
pub fn validate(dimensions: [u32; 6]) -> Result<()> {
    let [rows, query_heads, kv_heads, width, past, capacity] = dimensions;
    ensure!(
        [rows, query_heads, kv_heads, width] == [1, 24, 4, 256],
        "warp-fp64 attention requires M=1, 24 query heads, 4 KV heads, D=256"
    );
    ensure!(
        past < capacity && capacity <= 262_144,
        "invalid warp-fp64 KV prefix/capacity"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_geometry_and_prefix_are_required() {
        assert_eq!(GRID, [96, 1, 1]);
        assert_eq!(BLOCK, [32, 1, 1]);
        for past in [0, 1, 32, 127, 512, 8191, 32767, 131071] {
            assert!(validate([1, 24, 4, 256, past, past + 1]).is_ok());
        }
        for invalid in [
            [0, 24, 4, 256, 0, 1],
            [5, 24, 4, 256, 0, 5],
            [1, 12, 4, 256, 0, 1],
            [1, 24, 8, 256, 0, 1],
            [1, 24, 4, 128, 0, 1],
            [1, 24, 4, 256, 1, 1],
            [1, 24, 4, 256, 0, 262145],
            [1, 24, 4, 256, u32::MAX, 1],
        ] {
            assert!(validate(invalid).is_err(), "accepted {invalid:?}");
        }
    }
}
