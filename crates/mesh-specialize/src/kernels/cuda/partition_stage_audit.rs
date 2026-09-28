//! Optional diagnostic stage capture scoped to one model-profile prefill.
use super::driver::Buffer;
use crate::kernels::partition_audit::Rows;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap};

pub(super) struct Stages {
    layer: usize,
    rows: usize,
    stages: BTreeMap<String, Rows>,
}
thread_local! { static ACTIVE: RefCell<Option<Stages>> = const { RefCell::new(None) }; }
pub(super) struct Capture;
impl Capture {
    pub(super) fn start(layers: usize, rows: usize) -> Result<Self> {
        let layer = match std::env::var("MESH_SPECIALIZE_PARTITION_STAGE_LAYER") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) => Some(
                value
                    .parse::<usize>()
                    .context("invalid partition stage layer")?,
            ),
            Err(e) => return Err(e.into()),
        };
        ACTIVE.with_borrow_mut(|active| -> Result<()> {
            ensure!(active.is_none(), "nested partition stage capture");
            if let Some(layer) = layer {
                ensure!(layer < layers, "partition stage layer out of range");
                *active = Some(Stages {
                    layer,
                    rows,
                    stages: BTreeMap::new(),
                });
            }
            Ok(())
        })?;
        Ok(Self)
    }
    pub(super) fn finish(self) -> Option<Stages> {
        ACTIVE.with_borrow_mut(Option::take)
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|active| *active = None);
    }
}
pub(super) fn selected(layer: usize) -> bool {
    ACTIVE.with_borrow(|active| active.as_ref().is_some_and(|a| a.layer == layer))
}
pub(super) fn record(name: &str, rows: usize, buffer: &Buffer<'_>) -> Result<()> {
    ACTIVE.with_borrow_mut(|active| -> Result<()> {
        let audit = active
            .as_mut()
            .context("no active partition stage capture")?;
        ensure!(
            rows > 0 && buffer.len().is_multiple_of(rows),
            "invalid partition stage extent"
        );
        if !audit.stages.contains_key(name) {
            audit.stages.insert(
                name.to_owned(),
                Rows::new(1, audit.rows, buffer.len() / rows)?,
            );
        }
        let mut bytes = vec![0; buffer.len()];
        buffer.download(&mut bytes)?;
        audit
            .stages
            .get_mut(name)
            .context("missing partition stage")?
            .record(0, &bytes)
    })
}
impl Stages {
    pub(super) fn compare(&self, other: &Self) -> Result<Value> {
        ensure!(
            self.layer == other.layer
                && self.rows == other.rows
                && self.stages.keys().eq(other.stages.keys()),
            "partition stage geometry/names differ"
        );
        ensure!(!self.stages.is_empty(), "empty partition stage capture");
        let comparisons = self
            .stages
            .iter()
            .map(|(name, a)| Ok((name.clone(), a.compare(&other.stages[name])?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(
            json!({"diagnostic_only":true,"layer":self.layer,"rows":self.rows,"stages":comparisons,"scope":"existing observed layer path; MLP stage observation uses ordinary allocation path instead of workspace, so profile/control equality must also be checked"}),
        )
    }
}
