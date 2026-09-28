//! Process-fixed A/B scheduling. FP64 reassociation still needs model qualification.
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::sync::OnceLock;

pub const PAIRED_KERNEL: &str = "bf16_ab_decode_fp64";
pub const BASELINE_KERNEL: &str = "bf16_linear_decode";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Baseline,
    PairedFp64,
}

impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::PairedFp64 => "paired-fp64",
        }
    }

    /// Schedule admission only. Callers still validate owners, extents, alignment,
    /// non-aliasing and completion lifetimes. Unsupported shapes keep the control.
    pub fn select(self, rows: usize, channels: usize, width: usize) -> Self {
        if self == Self::PairedFp64
            && rows == 1
            && (1..=256).contains(&channels)
            && (8..=32768).contains(&width)
            && width.is_multiple_of(8)
        {
            Self::PairedFp64
        } else {
            Self::Baseline
        }
    }

    pub fn report(self, rows: usize, channels: usize, width: usize) -> Value {
        let selected = self.select(rows, channels, width);
        let paired = selected == Self::PairedFp64;
        json!({"requested":self.name(),"selected":selected.name(),
            "shape":[rows,channels,width],
            "kernel":if paired { PAIRED_KERNEL } else { BASELINE_KERNEL },
            "launches_per_gdn_layer":if paired { 1 } else { 2 },
            "scope":"shape-selected A/B only; FP64 reassociation; not a qualification claim"})
    }
}

fn parse(value: Option<&str>) -> Result<Schedule> {
    match value {
        None | Some("baseline") => Ok(Schedule::Baseline),
        Some("paired-fp64") => Ok(Schedule::PairedFp64),
        _ => bail!("MESH_SPECIALIZE_AB_SCHEDULE must be baseline or paired-fp64"),
    }
}

pub fn current() -> Result<Schedule> {
    static VALUE: OnceLock<Result<Schedule, String>> = OnceLock::new();
    match VALUE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_AB_SCHEDULE") {
        Ok(value) => parse(Some(&value)).map_err(|error| error.to_string()),
        Err(std::env::VarError::NotPresent) => parse(None).map_err(|error| error.to_string()),
        Err(error) => Err(error.to_string()),
    }) {
        Ok(value) => Ok(*value),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_explicit_schedule_parse_without_mutating_environment() {
        assert_eq!(parse(None).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("baseline")).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("paired-fp64")).unwrap(), Schedule::PairedFp64);
        for value in ["", "auto", "paired-fp32", "PAIRED-FP64", "paired-fp64 "] {
            assert!(parse(Some(value)).is_err());
        }
    }

    #[test]
    fn paired_only_for_supported_single_row_shapes() {
        for n in [1, 3, 48, 256] {
            for k in [8, 24, 1032, 5120, 32768] {
                assert_eq!(Schedule::PairedFp64.select(1, n, k), Schedule::PairedFp64);
                assert_eq!(Schedule::Baseline.select(1, n, k), Schedule::Baseline);
            }
        }
        for m in [0, 2, 4, 8, 128, 512, 2048] {
            assert_eq!(Schedule::PairedFp64.select(m, 48, 5120), Schedule::Baseline);
        }
        for (n, k) in [(0, 8), (257, 8), (48, 0), (48, 7), (48, 9), (48, 32776)] {
            assert_eq!(Schedule::PairedFp64.select(1, n, k), Schedule::Baseline);
        }
    }

    #[test]
    fn report_distinguishes_requested_from_shape_selected_schedule() {
        let prefill = Schedule::PairedFp64.report(128, 48, 5120);
        assert_eq!(prefill["requested"], "paired-fp64");
        assert_eq!(prefill["selected"], "baseline");
        assert_eq!(prefill["launches_per_gdn_layer"], 2);
        let decode = Schedule::PairedFp64.report(1, 48, 5120);
        assert_eq!(decode["selected"], "paired-fp64");
        assert_eq!(decode["shape"], json!([1, 48, 5120]));
        assert_eq!(decode["launches_per_gdn_layer"], 1);
    }
}
