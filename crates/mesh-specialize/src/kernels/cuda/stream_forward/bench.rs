//! `MESH_SPECIALIZE_EXECUTION=legacy|stream|graph` selection for the model benchmark.

use super::{
    StreamForward,
    graph_decode::{GraphDecode, ensure_exact},
};
use crate::kernels::{
    DecoderConfig,
    cuda::{
        driver::{Context, Module},
        resident_model::{Model, SelectedOutput, Session},
        resident_weights::ResidentWeights,
    },
};
use anyhow::{Result, anyhow, bail, ensure};
use serde_json::{Value, json};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::kernels::cuda) enum Execution {
    Legacy,
    Stream,
    Graph,
}

impl Execution {
    pub(in crate::kernels::cuda) fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Stream => "stream",
            Self::Graph => "graph",
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
        Some("graph") => Ok(Execution::Graph),
        _ => bail!("MESH_SPECIALIZE_EXECUTION must be legacy, stream or graph"),
    }
}

/// Selected-token forward through either the legacy model or `StreamForward`.
pub(in crate::kernels::cuda) struct Runner<'a, 'm, 'w, 'ctx> {
    model: &'a Model<'w, 'ctx>,
    execution: Execution,
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
        if execution == Execution::Graph {
            ensure_exact(crate::kernels::attention_profile::current()?)?;
        }
        let stream = match execution {
            Execution::Legacy => None,
            Execution::Stream | Execution::Graph => {
                Some(StreamForward::new(weights, module, config, max_rows)?)
            }
        };
        Ok(Self {
            model,
            execution,
            stream,
        })
    }

    pub(in crate::kernels::cuda) fn forward_selected(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
    ) -> Result<SelectedOutput> {
        ensure!(
            self.execution != Execution::Graph || session.cursor.past() == 0,
            "graph decode requires prepare_decode and replay; eager fallback is forbidden"
        );
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

    pub(in crate::kernels::cuda) fn forward_selected_decode(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        tokens: &[u32],
        session: &mut Session<'_>,
    ) -> Result<SelectedOutput> {
        match &self.stream {
            None => self
                .model
                .forward_selected_decode(ctx, module, tokens, session),
            Some(stream) => {
                let output = stream.forward(tokens, session, false)?;
                Ok(SelectedOutput {
                    token: output.token,
                    past: output.past,
                })
            }
        }
    }

    /// Bind once, after prefill and outside every decode timing interval.
    pub(in crate::kernels::cuda) fn prepare_decode<'s>(
        &'s mut self,
        session: &'s mut Session<'ctx>,
    ) -> Result<DecodeRunner<'s, 'a, 'm, 'w, 'ctx>> {
        if self.execution == Execution::Graph {
            let stream = self
                .stream
                .as_mut()
                .ok_or_else(|| anyhow!("graph stream missing"))?;
            Ok(DecodeRunner::Graph(Box::new(GraphDecode::capture(
                stream, session,
            )?)))
        } else {
            Ok(DecodeRunner::Eager {
                runner: self,
                session,
            })
        }
    }

    pub(in crate::kernels::cuda) fn report(&self) -> Value {
        match &self.stream {
            None => json!({"execution": Execution::Legacy.name()}),
            Some(stream) => json!({
                "execution": self.execution.name(),
                "stream_forward": stream.report(),
            }),
        }
    }
}

/// Holds the session lease for the complete decode phase, never a foreign session.
pub(in crate::kernels::cuda) enum DecodeRunner<'s, 'a, 'm, 'w, 'ctx> {
    Eager {
        runner: &'s Runner<'a, 'm, 'w, 'ctx>,
        session: &'s mut Session<'ctx>,
    },
    Graph(Box<GraphDecode<'s, 'm, 'w, 'ctx>>),
}

impl DecodeRunner<'_, '_, '_, '_, '_> {
    pub(in crate::kernels::cuda) fn is_graph(&self) -> bool {
        matches!(self, Self::Graph(_))
    }

    pub(in crate::kernels::cuda) fn forward(
        &mut self,
        context: &Context,
        module: &Module<'_>,
        token: u32,
    ) -> Result<SelectedOutput> {
        match self {
            Self::Eager { runner, session } => {
                runner.forward_selected_decode(context, module, &[token], session)
            }
            Self::Graph(graph) => {
                let output = graph.replay(token, false)?;
                Ok(SelectedOutput {
                    token: output.token,
                    past: output.past,
                })
            }
        }
    }

    pub(in crate::kernels::cuda) fn report(&self) -> Option<Value> {
        match self {
            Self::Graph(graph) => Some(graph.report()),
            Self::Eager { .. } => None,
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
        assert_eq!(parse(Some("graph")).unwrap(), Execution::Graph);
        assert!(parse(Some("unknown")).is_err());
    }
}
