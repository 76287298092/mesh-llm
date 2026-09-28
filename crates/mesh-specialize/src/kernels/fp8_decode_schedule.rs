//! Process-fixed decode scheduling, independent of the FP8 arithmetic profile.
use super::fp8_profile::Profile;
use anyhow::{Result, anyhow, bail};
#[cfg(any(test, target_os = "linux"))]
use serde_json::{Value, json};
#[cfg(any(test, target_os = "linux"))]
use std::cell::Cell;
use std::sync::OnceLock;

pub const BASELINE_KERNEL: &str = "fp8_linear_exact";
pub const VECTOR16_KERNEL: &str = "fp8_linear_exact_vector16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Baseline,
    Vector16,
}

/// Why a call actually selected (or did not select) the vector schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Baseline,
    Vector16,
    OtherProfile,
    OtherRows,
    UnsupportedWidth,
    Unaligned,
}

impl Selection {
    pub fn kernel(self) -> &'static str {
        if self == Self::Vector16 {
            VECTOR16_KERNEL
        } else {
            BASELINE_KERNEL
        }
    }
    pub fn reason(self) -> &'static str {
        match self {
            Self::Baseline => "baseline requested",
            Self::Vector16 => "exact M=1, K multiple of 16, aligned A/W",
            Self::OtherProfile => "arithmetic profile is not exact",
            Self::OtherRows => "M is not 1",
            Self::UnsupportedWidth => "K is not a multiple of 16 in 16..=32768",
            Self::Unaligned => "A or W is not 16-byte aligned",
        }
    }
}

impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Vector16 => "vector16",
        }
    }

    /// This checks schedule admission only. Callers still own finite-code,
    /// allocation extent/lifetime, scale, and N bounds from the exact contract.
    pub fn select(
        self,
        profile: Profile,
        rows: usize,
        width: usize,
        addresses: [u64; 2],
    ) -> Selection {
        if self == Self::Baseline {
            return Selection::Baseline;
        }
        if profile != Profile::Exact {
            return Selection::OtherProfile;
        }
        if rows != 1 {
            return Selection::OtherRows;
        }
        if !(16..=32768).contains(&width) || !width.is_multiple_of(16) {
            return Selection::UnsupportedWidth;
        }
        if addresses.iter().any(|p| *p == 0 || !p.is_multiple_of(16)) {
            return Selection::Unaligned;
        }
        Selection::Vector16
    }
}

fn parse(value: Option<&str>) -> Result<Schedule> {
    match value {
        None | Some("baseline") => Ok(Schedule::Baseline),
        Some("vector16") => Ok(Schedule::Vector16),
        _ => bail!("MESH_SPECIALIZE_FP8_DECODE_SCHEDULE must be baseline or vector16"),
    }
}

pub fn current() -> Result<Schedule> {
    static SCHEDULE: OnceLock<Result<Schedule, String>> = OnceLock::new();
    match SCHEDULE.get_or_init(
        || match std::env::var("MESH_SPECIALIZE_FP8_DECODE_SCHEDULE") {
            Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        },
    ) {
        Ok(schedule) => Ok(*schedule),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

/// Counts host schedule decisions, including graph capture, NOT graph replays.
/// Fixed storage keeps the stream enqueue path allocation-free.
#[cfg(any(test, target_os = "linux"))]
#[derive(Default)]
pub(crate) struct Decisions {
    baseline: Cell<usize>,
    vector16: Cell<usize>,
    width_fallback: Cell<usize>,
    alignment_fallback: Cell<usize>,
    other: Cell<usize>,
}

#[cfg(any(test, target_os = "linux"))]
impl Decisions {
    pub(crate) fn record(&self, selection: Selection) {
        let count = match selection {
            Selection::Baseline => &self.baseline,
            Selection::Vector16 => &self.vector16,
            Selection::UnsupportedWidth => &self.width_fallback,
            Selection::Unaligned => &self.alignment_fallback,
            Selection::OtherProfile | Selection::OtherRows => &self.other,
        };
        count.set(count.get().saturating_add(1));
    }

    pub(crate) fn report(&self, requested: Schedule) -> Value {
        json!({
            "requested": requested.name(),
            "scope": "exact-profile M=1 resident_fp8 and stream projections; MLP workspace/prepared paths unchanged",
            "counts_are": "host selection decisions, including capture, excluding graph replays; not GPU launch counts",
            "selected_vector16": self.vector16.get(),
            "selected_baseline": self.baseline.get() + self.width_fallback.get() + self.alignment_fallback.get() + self.other.get(),
            "fallback_width": self.width_fallback.get(),
            "fallback_alignment": self.alignment_fallback.get(),
            "ineligible_profile_or_rows": self.other.get(),
            "vector16_kernel": VECTOR16_KERNEL,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_selection_and_safe_fallbacks() {
        assert_eq!(parse(None).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("baseline")).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("vector16")).unwrap(), Schedule::Vector16);
        assert!(parse(Some("auto")).is_err());
        let pick = |profile, rows, width, pointers| {
            Schedule::Vector16.select(profile, rows, width, pointers)
        };
        for k in [16, 32, 128, 5120, 6144, 17408, 32768] {
            assert_eq!(pick(Profile::Exact, 1, k, [16, 256]), Selection::Vector16);
        }
        for k in [0, 1, 15, 17, 513, 32769, 32784] {
            assert_eq!(
                pick(Profile::Exact, 1, k, [16, 256]),
                Selection::UnsupportedWidth
            );
        }
        for pointers in [[0, 256], [16, 0], [17, 256], [16, 257]] {
            assert_eq!(pick(Profile::Exact, 1, 16, pointers), Selection::Unaligned);
        }
        for m in [0, 2, 3, 4, 16, 2048] {
            assert_eq!(pick(Profile::Exact, m, 16, [16, 256]), Selection::OtherRows);
        }
        for p in [
            Profile::A16Decode,
            Profile::A16Head,
            Profile::A16HeadGemv,
            Profile::NativePrefill,
            Profile::NativePrefillAudit,
            Profile::NativePrefillShort,
            Profile::NativePrefillShortAudit,
        ] {
            assert_eq!(pick(p, 1, 16, [16, 256]), Selection::OtherProfile);
        }
        assert_eq!(
            Schedule::Baseline.select(Profile::Exact, 1, 16, [16, 256]),
            Selection::Baseline
        );
    }

    #[test]
    fn vectors_cover_each_k_once_without_overread_and_integer_bound_is_exact() {
        for k in [16, 32, 128, 496, 512, 528, 5120, 6144, 17408, 32768] {
            let mut seen = vec![0; k];
            for lane in 0..32 {
                for start in (lane * 16..k).step_by(32 * 16) {
                    assert!(start + 16 <= k);
                    for value in &mut seen[start..start + 16] {
                        *value += 1;
                    }
                }
            }
            assert!(seen.iter().all(|v| *v == 1));
        }
        let bound = 32768_i64 * 229376_i64 * 229376_i64;
        assert!(bound < (1_i64 << 51));
    }

    #[test]
    fn records_actual_selection_not_just_requested_schedule() {
        let counts = Decisions::default();
        counts.record(Selection::Vector16);
        counts.record(Selection::UnsupportedWidth);
        counts.record(Selection::Unaligned);
        let report = counts.report(Schedule::Vector16);
        assert_eq!(report["selected_vector16"], 1);
        assert_eq!(report["selected_baseline"], 2);
        assert_eq!(report["fallback_width"], 1);
        assert_eq!(report["fallback_alignment"], 1);
    }
}
