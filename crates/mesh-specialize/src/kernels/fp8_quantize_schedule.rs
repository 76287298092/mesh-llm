use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Baseline,
    ReuseInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedInputGroup {
    GdnQkvZ,
    AttentionQkv,
    Fp8MlpGateUp,
}

impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ReuseInput => "reuse-input",
        }
    }

    pub fn reuses_inputs(self) -> bool {
        self == Self::ReuseInput
    }

    pub fn shared_source(self, group: SharedInputGroup) -> Option<&'static str> {
        if !self.reuses_inputs() {
            return None;
        }
        Some(match group {
            SharedInputGroup::GdnQkvZ => "gdn.qkv",
            SharedInputGroup::AttentionQkv => "attn.q",
            SharedInputGroup::Fp8MlpGateUp => "gate",
        })
    }
}

fn parse(value: Option<&str>) -> Result<Schedule> {
    match value {
        None | Some("baseline") => Ok(Schedule::Baseline),
        Some("reuse-input") => Ok(Schedule::ReuseInput),
        _ => bail!("MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE must be baseline or reuse-input"),
    }
}

pub fn current() -> Result<Schedule> {
    static SCHEDULE: OnceLock<Result<Schedule, String>> = OnceLock::new();
    match SCHEDULE.get_or_init(
        || match std::env::var("MESH_SPECIALIZE_FP8_QUANTIZE_SCHEDULE") {
            Ok(value) => parse(Some(&value)).map_err(|error| error.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        },
    ) {
        Ok(schedule) => Ok(*schedule),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::{Schedule, SharedInputGroup, parse};

    #[test]
    fn selects_only_explicit_quantization_schedules() {
        assert_eq!(parse(None).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("baseline")).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("reuse-input")).unwrap(), Schedule::ReuseInput);
        assert!(parse(Some("auto")).is_err());
        assert!(!Schedule::Baseline.reuses_inputs());
        assert!(Schedule::ReuseInput.reuses_inputs());
    }

    #[test]
    fn maps_shared_projection_groups_to_the_first_codes_and_scales() {
        let groups = [
            (
                SharedInputGroup::GdnQkvZ,
                ("gdn.qkv.codes", "gdn.qkv.scales"),
            ),
            (
                SharedInputGroup::AttentionQkv,
                ("attn.q.codes", "attn.q.scales"),
            ),
            (
                SharedInputGroup::Fp8MlpGateUp,
                ("gate.codes", "gate.scales"),
            ),
        ];
        for (group, (codes, scales)) in groups {
            assert_eq!(Schedule::Baseline.shared_source(group), None);
            let source = Schedule::ReuseInput.shared_source(group);
            assert_eq!(
                source.map(|projection| format!("{projection}.codes")),
                Some(codes.to_owned())
            );
            assert_eq!(
                source.map(|projection| format!("{projection}.scales")),
                Some(scales.to_owned())
            );
        }
    }
}
