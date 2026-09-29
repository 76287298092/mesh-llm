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
pub const PREFIX_SCAN_TILE: usize = 128;
pub const PREFIX_KERNELS: [&str; 3] = [
    "attention_staged_max_prefix_fp64",
    "attention_staged_coefficients_parallel_fp64",
    "attention_staged_normalizer_fp64",
];
pub const PREFIX_SCHEDULE_KERNELS: [&str; 5] = [
    KERNELS[0],
    PREFIX_KERNELS[0],
    PREFIX_KERNELS[1],
    PREFIX_KERNELS[2],
    KERNELS[2],
];
pub const PREFIX_BLOCKS: [[u32; 3]; 4] = [[32, 1, 1], [128, 1, 1], [32, 1, 1], [64, 1, 1]];
pub const SHORT_TRIAL_PASTS: [usize; 5] = [0, 1, 32, 105, 127];
pub const FULL_TRIAL_PASTS: [usize; 7] = [0, 1, 32, 105, 127, 511, 8190];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientSchedule {
    SerialV1,
    PrefixParallelV2,
}
impl CoefficientSchedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::SerialV1 => "serial-v1",
            Self::PrefixParallelV2 => "prefix-parallel-v2",
        }
    }

    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("serial-v1") => Ok(Self::SerialV1),
            Some("prefix-parallel-v2") => Ok(Self::PrefixParallelV2),
            _ => anyhow::bail!(
                "MESH_SPECIALIZE_STAGED_FP64_SCHEDULE must be serial-v1 or prefix-parallel-v2"
            ),
        }
    }

    pub fn current() -> Result<Self> {
        match std::env::var("MESH_SPECIALIZE_STAGED_FP64_SCHEDULE") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => anyhow::bail!("invalid MESH_SPECIALIZE_STAGED_FP64_SCHEDULE: {error}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub length: usize,
    pub capacity: usize,
    pub head_elements: usize,
    pub workspace_bytes: usize,
    pub grids: [[u32; 3]; 3],
    pub coefficient_schedule: CoefficientSchedule,
}
impl Plan {
    pub fn new(dimensions: [usize; 6]) -> Result<Self> {
        Self::new_with_schedule(dimensions, CoefficientSchedule::SerialV1)
    }

    pub fn new_with_schedule(
        dimensions: [usize; 6],
        coefficient_schedule: CoefficientSchedule,
    ) -> Result<Self> {
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
        let base_workspace_bytes = head_elements
            .checked_mul(3)
            .and_then(|n| n.checked_add(HEADS))
            .and_then(|n| n.checked_mul(8))
            .context("staged workspace extent overflow")?;
        let prefix_elements = match coefficient_schedule {
            CoefficientSchedule::SerialV1 => 0,
            CoefficientSchedule::PrefixParallelV2 => head_elements,
        };
        let prefix_bytes = prefix_elements
            .checked_mul(8)
            .context("staged prefix byte extent overflow")?;
        let workspace_bytes = base_workspace_bytes
            .checked_add(prefix_bytes)
            .context("staged workspace extent overflow")?;
        Ok(Self {
            length,
            capacity,
            head_elements,
            workspace_bytes,
            // Keys are x, heads y: y never exceeds CUDA's 65535 grid limit.
            grids: [[u32::try_from(length)?, 24, 1], [24, 1, 1], [24, 4, 1]],
            coefficient_schedule,
        })
    }

    pub fn prefix_grids(self) -> Result<[[u32; 3]; 5]> {
        Ok([
            self.grids[0],
            [24, 1, 1],
            [
                u32::try_from(self.length.div_ceil(PREFIX_SCAN_TILE))?,
                24,
                1,
            ],
            [24, 1, 1],
            self.grids[2],
        ])
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
    pub fn prefix_offset(self) -> usize {
        3 * self.head_elements + HEADS
    }
    /// Head-strided initialized prefix; tails remain untouched and must never be read.
    pub fn initialized(self, element: usize) -> bool {
        if element < 3 * self.head_elements {
            element % self.capacity < self.length
        } else if element < 3 * self.head_elements + HEADS {
            true
        } else if self.coefficient_schedule == CoefficientSchedule::PrefixParallelV2 {
            let prefix = self.prefix_offset();
            (prefix..prefix + self.head_elements).contains(&element)
                && (element - prefix) % self.capacity < self.length
        } else {
            false
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
#[path = "attention_staged_plan_tests.rs"]
mod tests;
