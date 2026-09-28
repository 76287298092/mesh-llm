//! Explicit attention arithmetic for internal qualification tools.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Exact,
    Online,
    OnlineAudit,
    /// FP32 split-sequence attention for M=1..8; exact FP64 attention for larger chunks.
    SplitDecode,
    /// Exact-order FP64 warp schedule for M=1; unchanged baseline for larger chunks.
    WarpFp64,
    /// Original CTA schedule for all rows, with exact-order unrolled exponential.
    UnrolledFp64,
    /// Three exact-order FP64 stages for M=1; original baseline for larger chunks.
    StagedFp64,
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "bf16-fp64-v1",
            Self::Online => "bf16-online-fp32-v1",
            Self::OnlineAudit => "bf16-online-audit-exact-output-v1",
            Self::SplitDecode => "bf16-split-decode-fp32-exact-prefill-v1",
            Self::WarpFp64 => "bf16-warp-fp64-exact-order-v1",
            Self::UnrolledFp64 => "bf16-unrolled-fp64-exact-order-v1",
            Self::StagedFp64 => "bf16-staged-fp64-exact-order-v1",
        }
    }
    /// Fallback kernel. Callers first handle split/staged multi-launch schedules.
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Online => "attention_online_bf16",
            Self::UnrolledFp64 => "causal_attention_unrolled_fp64",
            Self::Exact
            | Self::OnlineAudit
            | Self::SplitDecode
            | Self::WarpFp64
            | Self::StagedFp64 => "causal_attention_bf16",
        }
    }
    pub fn uses_warp(self, rows: usize) -> bool {
        self == Self::WarpFp64 && rows == 1
    }
    pub fn kernel_for_rows(self, rows: usize) -> &'static str {
        if self.uses_warp(rows) {
            super::attention_warp_plan::KERNEL
        } else {
            self.kernel()
        }
    }
    pub fn uses_staged(self, rows: usize) -> bool {
        self == Self::StagedFp64 && rows == 1
    }
    pub fn uses_split(self, rows: usize) -> bool {
        self == Self::SplitDecode && (1..=8).contains(&rows)
    }
    pub fn supports_stream(self) -> bool {
        matches!(
            self,
            Self::Exact
                | Self::SplitDecode
                | Self::WarpFp64
                | Self::UnrolledFp64
                | Self::StagedFp64
        )
    }
    pub fn is_audit(self) -> bool {
        self == Self::OnlineAudit
    }
}
fn parse(value: Option<&str>) -> Result<Profile> {
    match value {
        None | Some("exact") => Ok(Profile::Exact),
        Some("online") => Ok(Profile::Online),
        Some("online-audit") => Ok(Profile::OnlineAudit),
        Some("split-decode") => Ok(Profile::SplitDecode),
        Some("warp-fp64") => Ok(Profile::WarpFp64),
        Some("unrolled-fp64") => Ok(Profile::UnrolledFp64),
        Some("staged-fp64") => Ok(Profile::StagedFp64),
        _ => bail!(
            "MESH_SPECIALIZE_ATTENTION_PROFILE must be exact, online, online-audit, split-decode, warp-fp64, unrolled-fp64, or staged-fp64"
        ),
    }
}
pub fn current() -> Result<Profile> {
    static VALUE: OnceLock<Result<Profile, String>> = OnceLock::new();
    match VALUE.get_or_init(
        || match std::env::var("MESH_SPECIALIZE_ATTENTION_PROFILE") {
            Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        },
    ) {
        Ok(v) => Ok(*v),
        Err(e) => Err(anyhow!(e.clone())),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audit_returns_exact_outputs_and_unknown_profiles_fail() {
        assert_eq!(parse(None).unwrap(), Profile::Exact);
        assert_eq!(
            parse(Some("online-audit")).unwrap().kernel(),
            "causal_attention_bf16"
        );
        assert!(parse(Some("online-audit")).unwrap().is_audit());
        assert_eq!(
            parse(Some("online")).unwrap().kernel(),
            "attention_online_bf16"
        );
        assert!(parse(Some("tiled")).is_err());
    }

    #[test]
    fn split_decode_is_explicit_and_keeps_exact_prefill() {
        let profile = parse(Some("split-decode")).unwrap();
        assert_eq!(profile, Profile::SplitDecode);
        assert_eq!(profile.name(), "bf16-split-decode-fp32-exact-prefill-v1");
        assert!(!profile.is_audit());
        for rows in 1..=8 {
            assert!(profile.uses_split(rows));
        }
        for rows in [0, 9, 128, 512, 2048] {
            assert!(!profile.uses_split(rows));
        }
        assert_eq!(profile.kernel(), "causal_attention_bf16");
        for control in [Profile::Exact, Profile::Online, Profile::OnlineAudit] {
            for rows in [1, 5, 8, 512] {
                assert!(!control.uses_split(rows));
            }
        }
        assert_eq!(parse(None).unwrap(), Profile::Exact);
        assert!(parse(Some("split")).is_err());
    }

    #[test]
    fn warp_is_explicit_single_row_only_with_unchanged_fallbacks() {
        let profile = parse(Some("warp-fp64")).unwrap();
        assert_eq!(profile.name(), "bf16-warp-fp64-exact-order-v1");
        assert!(profile.uses_warp(1));
        assert_eq!(profile.kernel_for_rows(1), "causal_attention_warp_fp64");
        for rows in [0, 2, 5, 8, 128, 512, 2048] {
            assert!(!profile.uses_warp(rows));
            assert!(!profile.uses_split(rows));
            assert_eq!(profile.kernel_for_rows(rows), "causal_attention_bf16");
        }
        for control in [
            Profile::Exact,
            Profile::Online,
            Profile::OnlineAudit,
            Profile::SplitDecode,
        ] {
            assert!(!control.uses_warp(1));
            assert_eq!(control.kernel_for_rows(1), control.kernel());
        }
        assert_eq!(parse(None).unwrap(), Profile::Exact);
        assert!(!profile.is_audit());
    }

    #[test]
    fn unrolled_is_explicit_and_uses_control_schedule_for_all_rows() {
        let profile = parse(Some("unrolled-fp64")).unwrap();
        assert_eq!(profile, Profile::UnrolledFp64);
        assert_eq!(profile.name(), "bf16-unrolled-fp64-exact-order-v1");
        for rows in [1, 5, 17, 128, 512, 2048] {
            assert_eq!(
                profile.kernel_for_rows(rows),
                "causal_attention_unrolled_fp64"
            );
            assert!(!profile.uses_warp(rows));
            assert!(!profile.uses_split(rows));
        }
        assert!(!profile.is_audit());
        assert_eq!(parse(None).unwrap(), Profile::Exact);
        assert!(parse(Some("unrolled")).is_err());
    }

    #[test]
    fn staged_is_explicit_single_row_and_preserves_all_other_profiles() {
        let profile = parse(Some("staged-fp64")).unwrap();
        assert_eq!(profile.name(), "bf16-staged-fp64-exact-order-v1");
        assert!(profile.uses_staged(1));
        for rows in [0, 2, 5, 17, 128, 512, 2048] {
            assert!(!profile.uses_staged(rows));
            assert_eq!(profile.kernel_for_rows(rows), "causal_attention_bf16");
        }
        for old in [
            Profile::Exact,
            Profile::Online,
            Profile::OnlineAudit,
            Profile::SplitDecode,
            Profile::WarpFp64,
            Profile::UnrolledFp64,
        ] {
            assert!(!old.uses_staged(1));
        }
        assert!(!profile.uses_warp(1) && !profile.uses_split(1) && !profile.is_audit());
        assert_eq!(parse(None).unwrap(), Profile::Exact);
    }

    #[test]
    fn stream_admits_exact_split_decode_warp_unrolled_or_staged() {
        assert!(Profile::StagedFp64.supports_stream());
        assert!(Profile::UnrolledFp64.supports_stream());
        assert!(Profile::WarpFp64.supports_stream());
        assert!(Profile::Exact.supports_stream());
        assert!(Profile::SplitDecode.supports_stream());
        assert!(!Profile::Online.supports_stream());
        assert!(!Profile::OnlineAudit.supports_stream());
    }
}
