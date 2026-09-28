//! Linear-capacity workspace and fixed M=1 launch admission for staged FP64 attention.
use anyhow::{Context as _, Result, ensure};

pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 4;
pub const WIDTH: usize = 256;
pub const SCALE: f32 = 0.0625;
pub const KERNELS: [&str; 3] = [
    "attention_staged_scores_fp64",
    "attention_staged_coefficients_fp64",
    "attention_staged_values_fp64",
];
pub const BLOCKS: [[u32; 3]; 3] = [[32, 1, 1], [32, 1, 1], [64, 1, 1]];
pub const SHORT_TRIAL_PASTS: [usize; 5] = [0, 1, 32, 105, 127];
pub const FULL_TRIAL_PASTS: [usize; 7] = [0, 1, 32, 105, 127, 511, 8190];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub length: usize,
    pub capacity: usize,
    pub head_elements: usize,
    pub workspace_bytes: usize,
    pub grids: [[u32; 3]; 3],
}
impl Plan {
    pub fn new(dimensions: [usize; 6]) -> Result<Self> {
        let [rows, qh, kh, width, past, capacity] = dimensions;
        ensure!(
            [rows, qh, kh, width] == [1, HEADS, KV_HEADS, WIDTH],
            "staged-fp64 requires M=1, 24 query heads, 4 KV heads and D256"
        );
        ensure!(
            (1..=262_144).contains(&capacity),
            "invalid staged attention capacity"
        );
        let length = past
            .checked_add(1)
            .context("staged attention length overflow")?;
        ensure!(
            length <= capacity,
            "staged attention prefix exceeds capacity"
        );
        let head_elements = HEADS
            .checked_mul(capacity)
            .context("staged score extent overflow")?;
        let workspace_bytes = head_elements
            .checked_mul(3)
            .and_then(|n| n.checked_add(HEADS))
            .and_then(|n| n.checked_mul(8))
            .context("staged workspace extent overflow")?;
        Ok(Self {
            length,
            capacity,
            head_elements,
            workspace_bytes,
            // Keys are x, heads y: y never exceeds CUDA's 65535 grid limit.
            grids: [[u32::try_from(length)?, 24, 1], [24, 1, 1], [24, 4, 1]],
        })
    }
    pub fn dimensions(self) -> [u32; 2] {
        [self.length as u32, self.capacity as u32]
    }
    /// Byte offsets of scores, alpha, beta and normalizer in a single FP64 allocation.
    pub fn offsets(self) -> [usize; 4] {
        [
            0,
            self.head_elements * 8,
            self.head_elements * 16,
            self.head_elements * 24,
        ]
    }
    /// Head-strided initialized prefix; tails remain untouched and must never be read.
    pub fn initialized(self, element: usize) -> bool {
        if element < 3 * self.head_elements {
            element % self.capacity < self.length
        } else {
            element < 3 * self.head_elements + HEADS
        }
    }
}

/// Half-open disjoint ranges, checked without dereferencing device pointers.
/// Order: Q, K cache, V cache, BF16 output, FP32 output, workspace.
pub fn validate_addresses(plan: Plan, pointers: [u64; 6]) -> Result<()> {
    let bytes = [
        HEADS * WIDTH * 2,
        plan.capacity * KV_HEADS * WIDTH * 2,
        plan.capacity * KV_HEADS * WIDTH * 2,
        HEADS * WIDTH * 2,
        HEADS * WIDTH * 4,
        plan.workspace_bytes,
    ];
    let alignments = [2, 2, 2, 2, 4, 8];
    let mut ends = [0_u64; 6];
    for i in 0..6 {
        ensure!(
            pointers[i] != 0 && pointers[i].is_multiple_of(alignments[i]),
            "staged attention pointer alignment/null failure at {i}"
        );
        ends[i] = pointers[i]
            .checked_add(u64::try_from(bytes[i])?)
            .context("staged attention address overflow")?;
    }
    for i in 0..6 {
        for j in i + 1..6 {
            ensure!(
                ends[i] <= pointers[j] || ends[j] <= pointers[i],
                "staged attention ranges {i} and {j} overlap"
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct TrialOptions {
    pub long_cases: bool,
    pub timing: bool,
}
impl TrialOptions {
    /// Sanitizer-safe defaults: no timings and no prefixes longer than128 tokens.
    pub fn parse(scope: Option<&str>, timing: Option<&str>) -> Result<Self> {
        let long_cases = match scope {
            None | Some("short") => false,
            Some("full") => true,
            _ => anyhow::bail!("MESH_SPECIALIZE_STAGED_TRIAL_SCOPE must be short or full"),
        };
        let timing = match timing {
            None | Some("off") => false,
            Some("on") => true,
            _ => anyhow::bail!("MESH_SPECIALIZE_STAGED_TRIAL_TIMING must be off or on"),
        };
        Ok(Self { long_cases, timing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_and_grid_are_linear_and_capacity_strided() {
        let p = Plan::new([1, 24, 4, 256, 8191, 131072]).unwrap();
        assert_eq!(p.grids, [[8192, 24, 1], [24, 1, 1], [24, 4, 1]]);
        assert_eq!(BLOCKS, [[32, 1, 1], [32, 1, 1], [64, 1, 1]]);
        assert_eq!(p.workspace_bytes, 72 * 1024 * 1024 + 24 * 8);
        assert_eq!(
            p.offsets(),
            [0, 24 * 1024 * 1024, 48 * 1024 * 1024, 72 * 1024 * 1024]
        );
        let max = Plan::new([1, 24, 4, 256, 262143, 262144]).unwrap();
        assert_eq!(max.grids[0], [262144, 24, 1]);
    }
    #[test]
    fn rejects_other_rows_shapes_capacity_and_overflows() {
        for d in [
            [5, 24, 4, 256, 0, 5],
            [1, 12, 4, 256, 0, 1],
            [1, 24, 8, 256, 0, 1],
            [1, 24, 4, 128, 0, 1],
            [1, 24, 4, 256, 1, 1],
            [1, 24, 4, 256, 0, 0],
            [1, 24, 4, 256, 0, 262145],
            [1, 24, 4, 256, usize::MAX, 131072],
        ] {
            assert!(Plan::new(d).is_err());
        }
    }
    #[test]
    fn initialized_prefix_is_separate_in_every_head_and_array() {
        let p = Plan::new([1, 24, 4, 256, 1, 9]).unwrap();
        for array in 0..3 {
            for head in 0..24 {
                for key in 0..9 {
                    assert_eq!(
                        p.initialized(array * p.head_elements + head * 9 + key),
                        key < 2
                    );
                }
            }
        }
        assert!(p.initialized(3 * p.head_elements + 23));
        assert!(!p.initialized(3 * p.head_elements + 24));
    }
    #[test]
    fn addresses_must_be_aligned_bounded_and_disjoint() {
        let p = Plan::new([1, 24, 4, 256, 0, 1]).unwrap();
        let good = [0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 0x60000];
        assert!(validate_addresses(p, good).is_ok());
        for i in 0..6 {
            let mut bad = good;
            bad[i] += 1;
            assert!(validate_addresses(p, bad).is_err());
        }
        let mut overlap = good;
        overlap[5] = good[1];
        assert!(validate_addresses(p, overlap).is_err());
        let mut overflow = good;
        overflow[5] = u64::MAX - 7;
        assert!(validate_addresses(p, overflow).is_err());
    }
    #[test]
    fn source_keeps_exact_tree_serial_recurrences_and_capacity_planes() {
        let source = include_str!("../../kernels/nvptx/attention_staged_fp64.rs");
        let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
        for required in [
            "warp_dot(local_tree(products))",
            "lane!=0",
            "for key in 0..length",
            "normalizer=add_rn(multiply_rn(normalizer,alpha),beta)",
            "accumulator=add_rn(multiply_rn(accumulator,alpha),multiply_rn(beta,value))",
            "3*plane+head as usize",
            "let plane=24*capacity as usize",
        ] {
            let required: String = required.chars().filter(|c| !c.is_whitespace()).collect();
            assert!(compact.contains(&required), "missing {required}");
        }
        for forbidden in ["bar.sync", ".shared", "fma.", "ex2."] {
            assert!(!source.contains(forbidden));
        }
    }

    #[test]
    fn sanitizer_defaults_disable_expensive_trial_work() {
        let options = TrialOptions::parse(None, None).unwrap();
        assert!(!options.long_cases && !options.timing);
        let full = TrialOptions::parse(Some("full"), Some("on")).unwrap();
        assert!(full.long_cases && full.timing);
        assert!(TrialOptions::parse(Some("all"), None).is_err());
        assert!(TrialOptions::parse(None, Some("yes")).is_err());
    }

    #[test]
    fn trial_pasts_cover_requested_attention_lengths() {
        assert_eq!(SHORT_TRIAL_PASTS.map(|past| past + 1), [1, 2, 33, 106, 128]);
        assert_eq!(
            FULL_TRIAL_PASTS.map(|past| past + 1),
            [1, 2, 33, 106, 128, 512, 8191]
        );
    }
}
