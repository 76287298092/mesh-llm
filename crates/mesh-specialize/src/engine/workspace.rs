//! Capacity-bounded scratch planning built on the canonical engine layout.

use super::layout::{Layout as RegionLayout, Region};
use anyhow::{Context, Result, anyhow, ensure};

pub const FP8_CODES_REGION: &str = "fp8_codes";
pub const ROW_SCALES_REGION: &str = "row_scales";
pub const BF16_OUTPUT_REGION: &str = "bf16_output";
pub const FP32_DIAGNOSTICS_REGION: &str = "fp32_diagnostics";

/// An immutable, capacity-bounded workspace layout with stable named offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceLayout {
    placement: RegionLayout,
    high_water_bytes: usize,
    capacity_bytes: usize,
}

impl WorkspaceLayout {
    /// Place named regions using `engine::layout`'s checked 256-byte alignment.
    pub fn new(
        capacity_bytes: usize,
        entries: impl IntoIterator<Item = (String, usize)>,
    ) -> Result<Self> {
        ensure!(
            capacity_bytes > 0,
            "workspace capacity must be greater than zero"
        );
        let placement_entries = entries
            .into_iter()
            .map(|(name, bytes)| {
                let bytes = u64::try_from(bytes)
                    .context("workspace region size does not fit the layout range")?;
                Ok((name, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let placement = RegionLayout::new(placement_entries)?;
        let high_water_bytes =
            usize::try_from(placement.bytes).context("workspace layout size does not fit usize")?;
        ensure!(
            high_water_bytes <= capacity_bytes,
            "workspace layout high-water mark is {high_water_bytes} bytes, beyond capacity {capacity_bytes}"
        );
        Ok(Self {
            placement,
            high_water_bytes,
            capacity_bytes,
        })
    }

    /// Plan one FP8 projection chain from caller-supplied `m`, `k`, and `n` geometry.
    pub fn fp8_projection_chain(
        rows: usize,
        input_width: usize,
        output_width: usize,
        capacity_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            rows > 0,
            "FP8 projection workspace rows must be greater than zero"
        );
        ensure!(
            input_width > 0,
            "FP8 projection workspace input width must be greater than zero"
        );
        ensure!(
            output_width > 0,
            "FP8 projection workspace output width must be greater than zero"
        );

        let code_bytes = checked_product(rows, input_width, "FP8 code workspace")?;
        let row_scale_bytes = checked_product(rows, 4, "FP8 row-scale workspace")?;
        let output_elements = checked_product(rows, output_width, "FP8 output workspace")?;
        let bf16_output_bytes = checked_product(output_elements, 2, "BF16 output workspace")?;
        let fp32_diagnostic_bytes =
            checked_product(output_elements, 4, "FP32 diagnostic workspace")?;

        Self::new(
            capacity_bytes,
            [
                (FP8_CODES_REGION.to_owned(), code_bytes),
                (ROW_SCALES_REGION.to_owned(), row_scale_bytes),
                (BF16_OUTPUT_REGION.to_owned(), bf16_output_bytes),
                (FP32_DIAGNOSTICS_REGION.to_owned(), fp32_diagnostic_bytes),
            ],
        )
    }

    /// Shared MLP layout covers FP8 and NVFP4 scratch without per-layer arenas.
    pub fn mlp_chain(shape: [usize; 3]) -> Result<Self> {
        let [rows, width, channels] = shape;
        ensure!(
            (1..=512).contains(&rows)
                && (1..=32768).contains(&width)
                && (1..=32768).contains(&channels),
            "invalid model MLP workspace shape"
        );
        let mut entries = Vec::new();
        for (name, k, n) in [
            ("gate", width, channels),
            ("up", width, channels),
            ("down", channels, width),
        ] {
            let input = checked_product(rows, k, "MLP input")?;
            let output = checked_product(rows, n, "MLP output")?;
            for (suffix, bytes) in [
                ("codes", input),
                ("scales", (rows * 4).max(input / 16)),
                ("effective", 4.max(input / 16 * 4)),
                ("values", output * 2),
                ("raw", output * 4),
            ] {
                entries.push((format!("{name}.{suffix}"), bytes));
            }
        }
        for (suffix, bytes) in [
            ("values", rows * channels * 2),
            ("silu", rows * channels * 4),
            ("activated", rows * channels * 2),
            ("raw", rows * channels * 4),
        ] {
            entries.push((format!("activation.{suffix}"), bytes));
        }
        Self::new(512 * 1024 * 1024, entries)
    }

    /// Find a planned region using the canonical layout's sorted name index.
    pub fn region(&self, name: &str) -> Result<&Region> {
        self.placement
            .region(name)
            .map_err(|error| anyhow!("workspace region lookup failed: {error}"))
    }

    pub fn regions(&self) -> &[Region] {
        &self.placement.regions
    }

    /// Aligned arena size and high-water mark used by the CUDA owner.
    pub fn high_water_bytes(&self) -> usize {
        self.high_water_bytes
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} byte count overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{
        BF16_OUTPUT_REGION, FP8_CODES_REGION, FP32_DIAGNOSTICS_REGION, ROW_SCALES_REGION,
        WorkspaceLayout,
    };

    fn entries(items: &[(&str, usize)]) -> Vec<(String, usize)> {
        items
            .iter()
            .map(|(name, bytes)| ((*name).to_owned(), *bytes))
            .collect()
    }

    #[test]
    fn reuses_checked_layout_alignment_and_disjoint_placement() {
        let layout =
            WorkspaceLayout::new(1_024, entries(&[("gamma", 7), ("alpha", 3), ("beta", 4)]))
                .unwrap();

        assert_eq!(layout.region("alpha").unwrap().offset, 0);
        assert_eq!(layout.region("beta").unwrap().offset, 256);
        assert_eq!(layout.region("gamma").unwrap().offset, 512);
        assert_eq!(layout.high_water_bytes(), 768);
        assert!(
            layout
                .regions()
                .iter()
                .all(|region| region.offset % 256 == 0)
        );
        assert!(
            layout
                .regions()
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].length <= pair[1].offset)
        );
    }

    #[test]
    fn canonical_names_make_offsets_independent_of_input_order() {
        let forward =
            WorkspaceLayout::new(1_024, entries(&[("codes", 12), ("output", 24)])).unwrap();
        let reverse =
            WorkspaceLayout::new(1_024, entries(&[("output", 24), ("codes", 12)])).unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(forward.region("codes").unwrap().offset, 0);
        assert_eq!(forward.region("output").unwrap().offset, 256);
    }

    #[test]
    fn fp8_projection_plan_uses_checked_caller_geometry() {
        let layout = WorkspaceLayout::fp8_projection_chain(2, 7, 3, 1_024).unwrap();

        assert_eq!(layout.region(FP8_CODES_REGION).unwrap().length, 14);
        assert_eq!(layout.region(ROW_SCALES_REGION).unwrap().length, 8);
        assert_eq!(layout.region(BF16_OUTPUT_REGION).unwrap().length, 12);
        assert_eq!(layout.region(FP32_DIAGNOSTICS_REGION).unwrap().length, 24);
        assert!(
            layout
                .regions()
                .iter()
                .all(|region| region.offset % 256 == 0)
        );
    }

    #[test]
    fn rejects_zero_capacity_sizes_and_regions_beyond_capacity() {
        assert!(WorkspaceLayout::new(0, entries(&[("a", 1)])).is_err());
        assert!(WorkspaceLayout::new(1_024, Vec::<(String, usize)>::new()).is_err());
        assert!(WorkspaceLayout::new(1_024, entries(&[("zero", 0)])).is_err());
        assert!(WorkspaceLayout::new(256, entries(&[("too-large", 257)])).is_err());
        assert!(
            WorkspaceLayout::new(usize::MAX, entries(&[("a", usize::MAX), ("b", 1)]),).is_err()
        );
        assert!(WorkspaceLayout::fp8_projection_chain(usize::MAX, 2, 1, usize::MAX).is_err());
        assert!(WorkspaceLayout::fp8_projection_chain(0, 1, 1, 1_024).is_err());
    }
}

#[cfg(test)]
mod mlp_chain_tests {
    use super::WorkspaceLayout;
    #[test]
    fn covers_both_quantizers_and_all_buffers_with_one_bounded_arena() {
        for rows in [1, 5, 128, 512] {
            let p = WorkspaceLayout::mlp_chain([rows, 5120, 17408]).unwrap();
            assert!(p.high_water_bytes() < 256 * 1024 * 1024);
            assert_eq!(p.regions().len(), 19);
            assert!(p.region("gate.codes").unwrap().length >= (rows * 5120) as u64);
            assert!(p.region("down.effective").unwrap().length >= (rows * 17408 / 16 * 4) as u64);
            assert_eq!(
                p.region("down.values").unwrap().length,
                (rows * 5120 * 2) as u64
            );
        }
        for s in [
            [0, 5120, 17408],
            [513, 5120, 17408],
            [1, 0, 17408],
            [1, 5120, usize::MAX],
        ] {
            assert!(WorkspaceLayout::mlp_chain(s).is_err());
        }
    }
}
