//! One captured full M=1 exact forward, exclusively bound to a prefilled session.
//!
//! Exclusive borrows prevent eager reuse, foreign-session replay, or freeing the
//! arena/state while graph nodes refer to them. StreamForward's Function/module
//! and weights borrows retain the remaining dependencies. Drop drains first, then
//! destroys executable/graph before releasing position storage or either borrow.

use super::{StreamForward, StreamOutput, graph_drain, graph_position::Position, layers::Step};
use crate::kernels::{
    ab_schedule, attention_profile,
    cuda::{
        driver::{
            Buffer,
            graph::{Graph, GraphExec, Stream},
        },
        resident_model::Session,
        resident_state::ResidentState,
    },
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::time::Instant;

pub(in crate::kernels::cuda) struct GraphDecode<'a, 'm, 'w, 'ctx> {
    executable: Option<GraphExec<'ctx>>,
    graph: Option<Graph<'ctx>>,
    position_buffer: Buffer<'ctx>,
    position: Position<'m, 'ctx>,
    forward: &'a mut StreamForward<'m, 'w, 'ctx>,
    session: &'a mut Session<'ctx>,
    capture_seconds: f64,
    instantiate_seconds: f64,
    replays: usize,
}

pub(super) fn ensure_exact(profile: attention_profile::Profile) -> Result<()> {
    ensure!(
        profile == attention_profile::Profile::Exact,
        "graph execution requires exact attention; {} is not qualified",
        profile.name()
    );
    Ok(())
}

fn ensure_baseline_ab(schedule: ab_schedule::Schedule) -> Result<()> {
    ensure!(
        schedule == ab_schedule::Schedule::Baseline,
        "graph execution requires MESH_SPECIALIZE_AB_SCHEDULE=baseline; paired A/B is not combined-qualified"
    );
    Ok(())
}

impl<'a, 'm, 'w, 'ctx> GraphDecode<'a, 'm, 'w, 'ctx> {
    /// Capture without running any node or changing the committed cursor. Prefill
    /// must already have completed eagerly on this executor and fresh session.
    pub(in crate::kernels::cuda) fn capture(
        forward: &'a mut StreamForward<'m, 'w, 'ctx>,
        session: &'a mut Session<'ctx>,
    ) -> Result<Self> {
        ensure_exact(forward.attention_profile)?;
        ensure_baseline_ab(forward.ab_schedule)?;
        ensure!(!session.cursor.is_poisoned(), "graph session is poisoned");
        ensure!(
            session.cursor.past() > 0,
            "graph binding requires completed eager prefill"
        );
        ensure!(
            session.cursor.past() < session.cursor.capacity(),
            "graph session has no decode capacity"
        );
        ensure!(
            session.cursor.capacity() == forward.rope.positions,
            "graph session capacity differs from RoPE capacity"
        );
        ensure!(
            session.state.belongs_to(forward.context),
            "graph session belongs to another context"
        );
        let position_buffer = Buffer::new(forward.context, 4)?;
        let position = Position::new(forward.module, position_buffer.pointer())?;
        let mut bound = Self {
            executable: None,
            graph: None,
            position_buffer,
            position,
            forward,
            session,
            capture_seconds: 0.0,
            instantiate_seconds: 0.0,
            replays: 0,
        };
        if let Err(error) = bound.capture_inner() {
            // Capture does not execute, but the failed binding is never reusable.
            drop(bound.session.cursor.begin(1)?);
            return Err(error);
        }
        Ok(bound)
    }

    fn capture_inner(&mut self) -> Result<()> {
        graph_drain::drain(&self.forward.stream, self.forward.context)?;
        let step = Step {
            rows: 1,
            past: 0,
            capacity: self.session.cursor.capacity(),
            cos: self.forward.rope.cos(0)?,
            sin: self.forward.rope.sin(0)?,
        };
        let start = Instant::now();
        let capture = self.forward.stream.begin_capture()?;
        {
            let active = self.forward.stream.enter()?;
            self.forward.enqueue_layers(
                &active,
                &self.session.state,
                &step,
                Some(&self.position),
            )?;
        }
        // SAFETY: This object exclusively borrows all state/arena owners and
        // retains module/weights through forward. Drop drains before their release.
        self.graph = Some(unsafe { capture.finish()? });
        self.capture_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        // SAFETY: The captured dependencies are retained through this object's Drop.
        self.executable = Some(unsafe {
            self.graph
                .as_ref()
                .context("capture omitted graph")?
                .instantiate()?
        });
        self.instantiate_seconds = start.elapsed().as_secs_f64();
        Ok(())
    }

    pub(in crate::kernels::cuda) fn replay(
        &mut self,
        token: u32,
        logits: bool,
    ) -> Result<StreamOutput> {
        ensure!(
            (token as usize) < self.forward.shapes.vocabulary,
            "graph token outside vocabulary"
        );
        let transaction = self.session.cursor.begin(1)?;
        let mut bytes = [0_u8; 8];
        bytes[..4].copy_from_slice(&token.to_le_bytes());
        bytes[4..].copy_from_slice(&u32::try_from(transaction.past())?.to_le_bytes());
        let mut uploads = graph_drain::Pending::new(bytes, || {
            graph_drain::drain(&self.forward.stream, self.forward.context)
        });
        let result = enqueue_replay(
            &self.forward.stream,
            self.executable
                .as_ref()
                .context("graph executable missing")?,
            &uploads.value,
            self.forward.slots.tokens,
            self.position_buffer.pointer(),
        );
        // Unconditionally complete even after upload or launch failure. Pending
        // also drains on unwinding while its host byte storage still exists.
        let drained = uploads.finish();
        result.and(drained)?;
        let token = self.forward.read_selection()?;
        let logits = logits.then(|| self.forward.download_logits()).transpose()?;
        let past = transaction.commit();
        self.replays += 1;
        Ok(StreamOutput {
            token,
            past,
            logits,
        })
    }

    /// Completed-output snapshot for the capture-does-not-execute diagnostic.
    pub(super) fn snapshot(&self) -> Result<StreamOutput> {
        Ok(StreamOutput {
            token: self.forward.read_selection()?,
            past: self.session.cursor.past(),
            logits: Some(self.forward.download_logits()?),
        })
    }

    pub(in crate::kernels::cuda) fn state(&self) -> &ResidentState<'ctx> {
        &self.session.state
    }

    pub(in crate::kernels::cuda) fn report(&self) -> Value {
        json!({
            "execution": "graph", "rows": 1, "arithmetic": "exact",
            "capture_seconds": self.capture_seconds,
            "instantiate_seconds": self.instantiate_seconds,
            "capture_executes_nodes": false, "successful_replays": self.replays,
            "capture_scope": "one full row including embedding, all decoder layers, head and GPU greedy; no host uploads/readbacks",
            "timing_scope": "capture and instantiate excluded from prefill/decode intervals; no warm replay hidden in setup",
            "per_replay": "token and u32 past HtoD on replay stream; one cuGraphLaunch; one stream completion; selection readback",
            "node_parameter_updates": 0, "position_bytes": 4,
            "allocation_scope": "Source-derived: replay calls no Buffer allocation/free; not an instrumented driver allocation counter. CUDA graph internal storage is unmeasured.",
            "drain_failure_policy": "Stream failure falls back to context drain and returns error; failure of both exits this prototype harness with status 70 without unwinding owners.",
        })
    }
}

impl Drop for GraphDecode<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // A failed EndCapture activation can leave capture active. End it before
        // attempting synchronization; abort errors remain diagnostic, not success.
        if let Err(error) = self.forward.stream.abort_capture() {
            tracing::error!(error = %format!("{error:#}"), "graph capture abort failed during cleanup");
        }
        if let Err(error) = graph_drain::drain(&self.forward.stream, self.forward.context) {
            tracing::error!(error = %format!("{error:#}"), "graph cleanup required context drain");
        }
        // Explicit order documents and enforces the raw driver's retention contract.
        drop(self.executable.take());
        drop(self.graph.take());
    }
}

fn enqueue_replay(
    stream: &Stream<'_>,
    graph: &GraphExec<'_>,
    bytes: &[u8; 8],
    token: u64,
    past: u64,
) -> Result<()> {
    let active = stream.enter()?;
    // SAFETY: Pending retains these host bytes through completion on success,
    // error or unwind. Destinations are the exclusive token and position slots;
    // GraphDecode retains all captured allocations/modules through replay/drain.
    unsafe {
        active.copy_from_host(token, &bytes[..4])?;
        active.copy_from_host(past, &bytes[4..])?;
        graph.launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{ensure_baseline_ab, ensure_exact};
    use crate::kernels::ab_schedule::Schedule;

    #[test]
    fn rejects_paired_ab_before_graph_capture() {
        assert!(ensure_baseline_ab(Schedule::Baseline).is_ok());
        assert!(ensure_baseline_ab(Schedule::PairedFp64).is_err());
    }
    use crate::{engine::session::Cursor, kernels::attention_profile::Profile};

    #[test]
    fn rejects_nonexact_attention_without_reinterpreting_source_dtypes() {
        assert!(ensure_exact(Profile::Exact).is_ok());
        for profile in [
            Profile::SplitDecode,
            Profile::Online,
            Profile::OnlineAudit,
            Profile::WarpFp64,
            Profile::UnrolledFp64,
        ] {
            assert!(ensure_exact(profile).is_err());
        }
    }

    #[test]
    fn graph_transaction_failures_poison_without_advancing() {
        let mut cursor = Cursor::new(8).unwrap();
        cursor.begin(3).unwrap().commit();
        drop(cursor.begin(1).unwrap());
        assert!(cursor.is_poisoned());
        assert_eq!(cursor.past(), 3);
        assert!(cursor.begin(1).is_err());
    }
}
