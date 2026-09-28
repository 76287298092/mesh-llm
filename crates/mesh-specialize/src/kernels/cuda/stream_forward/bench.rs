//! `MESH_SPECIALIZE_EXECUTION=legacy|stream` selection for the model benchmark.

use super::StreamForward;
use crate::kernels::{
    DecoderConfig,
    cuda::{
        driver::{Context, Module},
        resident_model::{Model, SelectedOutput, Session},
        resident_weights::ResidentWeights,
    },
};
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::kernels::cuda) enum Execution {
    Legacy,
    Stream,
}

impl Execution {
    pub(in crate::kernels::cuda) fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Stream => "stream",
        }
    }

    /// Read once per process; absent means legacy.
    pub(in crate::kernels::cuda) fn current() -> Result<Self> {
        static VALUE: OnceLock<Result<Execution, String>> = OnceLock::new();
        match VALUE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_EXECUTION") {
            Ok(value) => parse(Some(&value)).map_err(|error| error.to_string()),
            Err(std::env::VarError::NotPresent) => parse(None).map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        }) {
            Ok(value) => Ok(*value),
            Err(error) => Err(anyhow!(error.clone())),
        }
    }
}

fn parse(value: Option<&str>) -> Result<Execution> {
    match value {
        None | Some("legacy") => Ok(Execution::Legacy),
        Some("stream") => Ok(Execution::Stream),
        _ => bail!("MESH_SPECIALIZE_EXECUTION must be legacy or stream"),
    }
}

/// Selected-token forward through either the legacy model or `StreamForward`.
pub(in crate::kernels::cuda) struct Runner<'a, 'm, 'w, 'ctx> {
    model: &'a Model<'w, 'ctx>,
    stream: Option<StreamForward<'m, 'w, 'ctx>>,
}

impl<'a, 'm, 'w, 'ctx> Runner<'a, 'm, 'w, 'ctx> {
    pub(in crate::kernels::cuda) fn new(
        execution: Execution,
        model: &'a Model<'w, 'ctx>,
        weights: &'w ResidentWeights<'ctx>,
        module: &'m Module<'ctx>,
        config: &DecoderConfig,
        max_rows: usize,
    ) -> Result<Self> {
        let stream = match execution {
            Execution::Legacy => None,
            Execution::Stream => Some(StreamForward::new(weights, module, config, max_rows)?),
        };
        Ok(Self { model, stream })
    }

    pub(in crate::kernels::cuda) fn forward_selected(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
    ) -> Result<SelectedOutput> {
        match &self.stream {
            None => self.model.forward_selected(ctx, module, tokens, session),
            Some(stream) => {
                let output = stream.forward(tokens, session, false)?;
                Ok(SelectedOutput {
                    token: output.token,
                    past: output.past,
                })
            }
        }
    }

    pub(in crate::kernels::cuda) fn report(&self) -> Value {
        match &self.stream {
            None => json!({"execution": Execution::Legacy.name()}),
            Some(stream) => json!({
                "execution": Execution::Stream.name(),
                "stream_forward": stream.report(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Execution, parse};

    #[test]
    fn execution_defaults_to_legacy_and_rejects_unknown_values() {
        assert_eq!(parse(None).unwrap(), Execution::Legacy);
        assert_eq!(parse(Some("legacy")).unwrap(), Execution::Legacy);
        assert_eq!(parse(Some("stream")).unwrap(), Execution::Stream);
        assert!(parse(Some("graph")).is_err());
    }
}
