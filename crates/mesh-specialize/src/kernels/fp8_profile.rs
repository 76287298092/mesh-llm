//! Process-fixed arithmetic selection for internal ablation tools.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Exact,
    A16Decode,
    NativePrefill,
    NativePrefillAudit,
    NativePrefillShort,
    NativePrefillShortAudit,
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact-a8-v1",
            Self::A16Decode => "fp8-a16-decode-v1",
            Self::NativePrefill => "native-fp8-prefill-v1",
            Self::NativePrefillAudit => "native-fp8-prefill-v1-audit",
            Self::NativePrefillShort => "native-fp8-prefill-k64-v1",
            Self::NativePrefillShortAudit => "native-fp8-prefill-k64-v1-audit",
        }
    }
    pub fn is_audit(self) -> bool {
        matches!(
            self,
            Self::NativePrefillAudit | Self::NativePrefillShortAudit
        )
    }
    pub fn native_kernel(self) -> Option<&'static str> {
        match self {
            Self::Exact | Self::A16Decode => None,
            Self::NativePrefill | Self::NativePrefillAudit => Some("fp8_prefill_native"),
            Self::NativePrefillShort | Self::NativePrefillShortAudit => {
                Some("fp8_prefill_native_short")
            }
        }
    }
}
fn parse(value: Option<&str>) -> Result<Profile> {
    match value {
        None | Some("exact") => Ok(Profile::Exact),
        Some("a16-decode") => Ok(Profile::A16Decode),
        Some("native-prefill") => Ok(Profile::NativePrefill),
        Some("native-prefill-audit") => Ok(Profile::NativePrefillAudit),
        Some("native-prefill-short") => Ok(Profile::NativePrefillShort),
        Some("native-prefill-short-audit") => Ok(Profile::NativePrefillShortAudit),
        _ => bail!(
            "MESH_SPECIALIZE_FP8_PROFILE must be exact, a16-decode, native-prefill, native-prefill-audit, native-prefill-short, or native-prefill-short-audit"
        ),
    }
}
/// Read once per process. This is an internal benchmark profile, not a serving setting.
pub fn current() -> Result<Profile> {
    static PROFILE: OnceLock<Result<Profile, String>> = OnceLock::new();
    match PROFILE.get_or_init(|| {
        let value = std::env::var("MESH_SPECIALIZE_FP8_PROFILE");
        match value {
            Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        }
    }) {
        Ok(profile) => Ok(*profile),
        Err(error) => Err(anyhow!(error.clone())),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_is_explicit_and_rejects_unknown_profiles() {
        assert_eq!(parse(None).unwrap(), Profile::Exact);
        assert_eq!(parse(Some("exact")).unwrap(), Profile::Exact);
        assert_eq!(
            parse(Some("native-prefill")).unwrap(),
            Profile::NativePrefill
        );
        assert_eq!(
            parse(Some("native-prefill-short")).unwrap().native_kernel(),
            Some("fp8_prefill_native_short")
        );
        assert!(
            parse(Some("native-prefill-short-audit"))
                .unwrap()
                .is_audit()
        );
        assert_eq!(parse(Some("a16-decode")).unwrap(), Profile::A16Decode);
        assert!(Profile::A16Decode.native_kernel().is_none());
        assert!(!parse(None).unwrap().is_audit());
        assert!(parse(Some("native")).is_err());
    }
}
