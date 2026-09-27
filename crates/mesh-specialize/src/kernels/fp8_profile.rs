//! Process-fixed arithmetic selection for internal ablation tools.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Exact,
    NativePrefill,
    NativePrefillAudit,
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact-a8-v1",
            Self::NativePrefill => "native-fp8-prefill-v1",
            Self::NativePrefillAudit => "native-fp8-prefill-v1-audit",
        }
    }
}
fn parse(value: Option<&str>) -> Result<Profile> {
    match value {
        None | Some("exact") => Ok(Profile::Exact),
        Some("native-prefill") => Ok(Profile::NativePrefill),
        Some("native-prefill-audit") => Ok(Profile::NativePrefillAudit),
        _ => bail!(
            "MESH_SPECIALIZE_FP8_PROFILE must be exact, native-prefill, or native-prefill-audit"
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
        assert!(parse(Some("native")).is_err());
    }
}
