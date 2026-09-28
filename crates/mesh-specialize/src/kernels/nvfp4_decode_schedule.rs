//! Process-fixed exact M=1 scheduling only. Prefill and arithmetic profiles are unchanged.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

pub const BASELINE_KERNEL: &str = "nvfp4_decode_exact";
pub const PRMT_KERNEL: &str = "nvfp4_decode_exact_prmt";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Baseline,
    Prmt,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Prmt => "prmt",
        }
    }
    /// Callers own the unchanged exact-decode shape/alignment/domain contract.
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Baseline => BASELINE_KERNEL,
            Self::Prmt => PRMT_KERNEL,
        }
    }
    pub fn apply(
        self,
        mut schedule: super::nvfp4_profile::Schedule,
        rows: usize,
    ) -> super::nvfp4_profile::Schedule {
        if rows == 1 && schedule.kernel == BASELINE_KERNEL {
            schedule.kernel = self.kernel();
        }
        schedule
    }
}
fn parse(value: Option<&str>) -> Result<Schedule> {
    match value {
        None | Some("baseline") => Ok(Schedule::Baseline),
        Some("prmt") => Ok(Schedule::Prmt),
        _ => bail!("MESH_SPECIALIZE_NVFP4_DECODE_SCHEDULE must be baseline or prmt"),
    }
}
pub fn current() -> Result<Schedule> {
    static SCHEDULE: OnceLock<Result<Schedule, String>> = OnceLock::new();
    match SCHEDULE.get_or_init(
        || match std::env::var("MESH_SPECIALIZE_NVFP4_DECODE_SCHEDULE") {
            Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|e| e.to_string()),
            Err(error) => Err(error.to_string()),
        },
    ) {
        Ok(schedule) => Ok(*schedule),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

/// Select after the existing NVFP4 profile has made its shape decision.
pub fn projection(
    rows: usize,
    columns: usize,
    width: usize,
) -> Result<super::nvfp4_profile::Schedule> {
    let old = super::nvfp4_profile::current()?.schedule(rows, columns, width);
    Ok(current()?.apply(old, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::nvfp4_profile::Profile;

    #[test]
    fn baseline_default_and_invalid_values_fail_closed() {
        assert_eq!(parse(None).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("baseline")).unwrap().kernel(), BASELINE_KERNEL);
        assert_eq!(parse(Some("prmt")).unwrap().kernel(), PRMT_KERNEL);
        for bad in ["", "auto", "PRMT", "prmt "] {
            assert!(parse(Some(bad)).is_err());
        }
    }
    #[test]
    fn only_exact_single_row_kernel_changes_and_geometry_is_identical() {
        for profile in [
            Profile::Baseline,
            Profile::TiledPrefill,
            Profile::WidePrefill,
        ] {
            for rows in [1, 2, 4, 16, 128, 512, 513] {
                let old = profile.schedule(rows, 17408, 5120);
                let selected = Schedule::Prmt.apply(profile.schedule(rows, 17408, 5120), rows);
                assert_eq!(
                    (selected.tile_rows, selected.tile_columns, selected.threads),
                    (old.tile_rows, old.tile_columns, old.threads)
                );
                assert_eq!(
                    selected.kernel,
                    if rows == 1 { PRMT_KERNEL } else { old.kernel }
                );
                assert_eq!(
                    Schedule::Baseline
                        .apply(profile.schedule(rows, 17408, 5120), rows)
                        .kernel,
                    old.kernel
                );
            }
        }
    }
}
