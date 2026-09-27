//! Explicit attention arithmetic for internal qualification tools.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Exact,
    Online,
    OnlineAudit,
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "bf16-fp64-v1",
            Self::Online => "bf16-online-fp32-v1",
            Self::OnlineAudit => "bf16-online-audit-exact-output-v1",
        }
    }
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Online => "attention_online_bf16",
            Self::Exact | Self::OnlineAudit => "causal_attention_bf16",
        }
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
        _ => bail!("MESH_SPECIALIZE_ATTENTION_PROFILE must be exact, online, or online-audit"),
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
}
