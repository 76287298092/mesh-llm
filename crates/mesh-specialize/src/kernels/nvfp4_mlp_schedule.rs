use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Baseline,
    A16SwiGlu,
}

impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::A16SwiGlu => "a16-swiglu",
        }
    }

    pub fn fuses_decode_row(self, decode: bool, rows: usize, past: usize) -> bool {
        decode && self == Self::A16SwiGlu && rows == 1 && past > 0
    }
}

fn parse(value: Option<&str>) -> Result<Schedule> {
    match value {
        None | Some("baseline") => Ok(Schedule::Baseline),
        Some("a16-swiglu") => Ok(Schedule::A16SwiGlu),
        _ => bail!("MESH_SPECIALIZE_NVFP4_MLP_SCHEDULE must be baseline or a16-swiglu"),
    }
}

pub fn current() -> Result<Schedule> {
    static SCHEDULE: OnceLock<Result<Schedule, String>> = OnceLock::new();
    match SCHEDULE.get_or_init(
        || match std::env::var("MESH_SPECIALIZE_NVFP4_MLP_SCHEDULE") {
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
    use super::{Schedule, parse};

    #[test]
    fn explicit_profile_only_fuses_single_row_decode() {
        assert_eq!(parse(None).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("baseline")).unwrap(), Schedule::Baseline);
        assert_eq!(parse(Some("a16-swiglu")).unwrap(), Schedule::A16SwiGlu);
        assert!(parse(Some("auto")).is_err());
        assert!(!Schedule::Baseline.fuses_decode_row(true, 1, 1));
        assert!(!Schedule::A16SwiGlu.fuses_decode_row(false, 1, 1));
        assert!(!Schedule::A16SwiGlu.fuses_decode_row(true, 1, 0));
        assert!(Schedule::A16SwiGlu.fuses_decode_row(true, 1, 1));
        for rows in [2, 4, 17, 512] {
            assert!(!Schedule::A16SwiGlu.fuses_decode_row(true, rows, 1));
        }
    }
}
