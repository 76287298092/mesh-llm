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
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "bf16-fp64-v1",
            Self::Online => "bf16-online-fp32-v1",
            Self::OnlineAudit => "bf16-online-audit-exact-output-v1",
            Self::SplitDecode => "bf16-split-decode-fp32-exact-prefill-v1",
        }
    }
    /// Single-kernel path. Callers must first handle `uses_split(rows)` separately.
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Online => "attention_online_bf16",
            Self::Exact | Self::OnlineAudit | Self::SplitDecode => "causal_attention_bf16",
        }
    }
    pub fn uses_split(self, rows: usize) -> bool {
        self == Self::SplitDecode && (1..=8).contains(&rows)
    }
    pub fn supports_stream(self) -> bool {
        matches!(self, Self::Exact | Self::SplitDecode)
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
        _ => bail!(
            "MESH_SPECIALIZE_ATTENTION_PROFILE must be exact, online, online-audit, or split-decode"
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
    fn stream_admits_only_exact_or_split_decode() {
        assert!(Profile::Exact.supports_stream());
        assert!(Profile::SplitDecode.supports_stream());
        assert!(!Profile::Online.supports_stream());
        assert!(!Profile::OnlineAudit.supports_stream());
    }
}
