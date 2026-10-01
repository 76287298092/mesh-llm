use crate::{artifact::model_source::ModelArtifact, kernels::TargetBatchLoadRequest};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SelectedRows(usize);

impl TryFrom<usize> for SelectedRows {
    type Error = anyhow::Error;

    fn try_from(rows: usize) -> Result<Self> {
        ensure!(
            (1..=5).contains(&rows),
            "target batch rows must be in 1..=5"
        );
        Ok(Self(rows))
    }
}

impl SelectedRows {
    pub const fn get(self) -> usize {
        self.0
    }

    pub fn cases(selection: Option<Self>) -> std::ops::RangeInclusive<usize> {
        match selection {
            Some(rows) => rows.get()..=rows.get(),
            None => 1..=5,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub prefix: Vec<u32>,
    pub target_tokens: [u32; 5],
    pub continuation: Vec<u32>,
    pub capacity: usize,
}

impl Fixture {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let fixture: Self = serde_json::from_slice(bytes).context("parse target batch fixture")?;
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.prefix.is_empty(), "target batch prefix is empty");
        ensure!(
            !self.continuation.is_empty(),
            "target batch continuation is empty"
        );
        let rows = self
            .prefix
            .len()
            .checked_add(5)
            .and_then(|rows| rows.checked_add(self.continuation.len()))
            .context("target batch fixture extent overflow")?;
        ensure!(
            rows <= self.capacity && self.capacity <= 512,
            "target batch fixture requires prefix + 5 + continuation <= capacity <= 512"
        );
        let config = super::decoder::config(self.capacity)?;
        ensure!(
            self.prefix
                .iter()
                .chain(&self.target_tokens)
                .chain(&self.continuation)
                .all(|&token| usize::try_from(token).is_ok_and(|id| id < config.vocabulary)),
            "target batch fixture token is outside vocabulary"
        );
        Ok(())
    }
}

pub struct Request<'a> {
    pub artifact: &'a Path,
    pub ptx: &'a str,
    pub device: i32,
    pub fixture: &'a Fixture,
    pub selected_rows: Option<SelectedRows>,
}

pub fn run(request: Request<'_>) -> Result<serde_json::Value> {
    request.fixture.validate()?;
    ensure!(request.device >= 0, "device ordinal must be nonnegative");
    let mut artifact = ModelArtifact::open(request.artifact)?;
    let result = (|| {
        super::inventory::validate(artifact.directory())?;
        let objects = super::schedule::text_objects(artifact.directory())?;
        let config = super::decoder::config(request.fixture.capacity)?;
        crate::kernels::target_batch_decode_check(TargetBatchLoadRequest {
            ptx: request.ptx,
            device: request.device,
            artifact: &mut artifact,
            objects: &objects,
            config: &config,
            fixture: request.fixture,
            selected_rows: request.selected_rows,
        })
    })();
    let mut report = result.unwrap_or_else(|error| {
        json!({
            "all_passed": false, "error": format!("{error:#}")
        })
    });
    report["identity"] = json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    report["native_mtp_admitted"] = json!(false);
    report["timing_claim"] = json!(false);
    report["selected_rows"] = json!(request.selected_rows);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_bounded_and_default_requires_all_five_cases() {
        assert_eq!(
            SelectedRows::cases(None).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        for rows in 1..=5 {
            let selected = SelectedRows::try_from(rows).unwrap();
            assert_eq!(
                SelectedRows::cases(Some(selected)).collect::<Vec<_>>(),
                vec![rows]
            );
        }
        for rows in [0, 6, usize::MAX] {
            assert!(SelectedRows::try_from(rows).is_err());
        }
    }

    #[test]
    fn selection_leaves_valid_fixture_and_full_extent_requirement_unchanged() {
        let bytes =
            br#"{"prefix":[1],"target_tokens":[2,3,4,5,6],"continuation":[7],"capacity":7}"#;
        let fixture = Fixture::parse(bytes).unwrap();
        let request = Request {
            artifact: Path::new("unused"),
            ptx: "unused",
            device: 0,
            fixture: &fixture,
            selected_rows: Some(SelectedRows::try_from(1).unwrap()),
        };
        request.fixture.validate().unwrap();
        assert_eq!(request.fixture.prefix, vec![1]);
        assert_eq!(request.fixture.target_tokens, [2, 3, 4, 5, 6]);
        assert_eq!(request.fixture.continuation, vec![7]);
        assert_eq!(request.fixture.capacity, 7);
        let too_small = Fixture {
            capacity: 3,
            ..fixture
        };
        assert!(too_small.validate().is_err());
    }

    #[test]
    fn fixture_accepts_metadata_when_extent_fits() {
        let bytes = br#"{"prefix":[1],"target_tokens":[2,3,4,5,6],"continuation":[7],"capacity":8,"metadata":{"source":"fixture"}}"#;
        let fixture = Fixture::parse(bytes.as_slice()).unwrap();
        assert_eq!(fixture.capacity, 8);
    }

    #[test]
    fn fixture_rejects_malformed_or_unbounded_requests() {
        let valid =
            json!({"prefix":[1],"target_tokens":[2,3,4,5,6],"continuation":[7],"capacity":7});
        for (field, value) in [
            ("prefix", json!([])),
            ("continuation", json!([])),
            ("target_tokens", json!([1, 2, 3, 4])),
            ("prefix", json!([-1])),
            ("prefix", json!([248320])),
            ("capacity", json!(6)),
            ("capacity", json!(513)),
            ("capacity", json!("7")),
        ] {
            let mut input = valid.clone();
            input[field] = value;
            let result = Fixture::parse(&serde_json::to_vec(&input).unwrap());
            assert!(result.is_err(), "accepted invalid {field}");
        }
    }
}
