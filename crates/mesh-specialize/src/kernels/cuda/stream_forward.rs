//! Whole-model forward on one explicit stream over one preplanned arena.
//!
//! `StreamForward` launches the same kernels, with the same arguments, grids and
//! arithmetic, as the legacy default exact profile (`resident_model.rs`), but:
//! the context is entered once per forward; every kernel handle is resolved at
//! construction; every intermediate lives in one arena planned on the host with
//! liveness-based reuse; the forward enqueues no allocation, free or
//! synchronization; and selection runs on device, ending in one stream
//! synchronize and one 16-byte readback. Persistent state (K/V, convolution
//! history, GDN recurrence) and cursor transaction semantics are unchanged.
//! With the explicit SplitDecode profile, M<=8 uses split/reduce instead, with
//! one separately owned persistent workspace; larger chunks keep exact attention.
//!
//! Per-forward host transfers: one stream-ordered pageable upload of the token
//! IDs (4 bytes per row), one stream synchronize, and one 16-byte selection
//! readback. Full BF16 logits are downloaded only when requested.

pub(super) mod bench;
pub(in crate::kernels) mod check;
pub(in crate::kernels) mod chunked_bench;
mod functions;
mod layers;
mod ops;
mod plan;
mod program;
mod split_attention;
mod weights;

use super::{
    driver::{
        Buffer, Context, Module,
        graph::{ActiveStream, Stream},
    },
    resident_model::Session,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    engine::rope::TextRope,
    kernels::{DecoderConfig, attention_profile, fp8_profile, nvfp4_profile},
};
use anyhow::{Context as _, Result, anyhow, ensure};
use functions::Functions;
use layers::Step;
use ops::Enqueue;
use plan::ArenaPlan;
use program::{MAX_ROWS, Shapes, Slots, forward_program};
use serde_json::{Value, json};
use split_attention::SplitAttention;
use weights::{Block, ModelWeights};

const ROPE_CHUNK_ROWS: usize = 2048;

pub(super) struct StreamForward<'m, 'w, 'ctx> {
    context: &'ctx Context,
    _weights: &'w ResidentWeights<'ctx>,
    kernels: Functions<'m, 'ctx>,
    attention_profile: attention_profile::Profile,
    split_attention: Option<SplitAttention<'m, 'ctx>>,
    stream: Stream<'ctx>,
    arena: Buffer<'ctx>,
    rope: Rope<'ctx>,
    slots: Slots,
    bound: ModelWeights,
    shapes: Shapes,
    max_rows: usize,
    arena_bytes: usize,
    peak_live_bytes: usize,
}

/// Selected token, committed cursor, and optional diagnostic logits.
pub(super) struct StreamOutput {
    pub(super) token: u32,
    pub(super) past: usize,
    pub(super) logits: Option<Vec<u16>>,
}

/// BF16 cos/sin tables for every position up to the configured capacity.
struct Rope<'ctx> {
    buffer: Buffer<'ctx>,
    positions: usize,
    half: usize,
}

impl<'m, 'w, 'ctx> StreamForward<'m, 'w, 'ctx> {
    /// Bind weights, plan and allocate the arena for up to `max_rows` rows, upload
    /// the row-ID and RoPE tables, resolve kernels, and create the stream.
    pub(super) fn new(
        weights: &'w ResidentWeights<'ctx>,
        module: &'m Module<'ctx>,
        config: &DecoderConfig,
        max_rows: usize,
    ) -> Result<Self> {
        let attention_profile = ensure_supported_profiles()?;
        let context = weights.context();
        ensure!(
            module.belongs_to(context),
            "stream forward module belongs to another context"
        );
        ensure!(
            (1..=MAX_ROWS).contains(&max_rows),
            "stream forward max rows must be in 1..={MAX_ROWS}"
        );
        let shapes = Shapes::from_config(config)?;
        // Default construction neither resolves split handles nor allocates scratch.
        let split_attention = (attention_profile == attention_profile::Profile::SplitDecode)
            .then(|| SplitAttention::new(context, module, &shapes, max_rows, config.capacity))
            .transpose()?;
        let bound = ModelWeights::bind(weights, config)?;
        let specs = forward_program(&shapes, max_rows)?;
        let plan = ArenaPlan::place(&specs)?;
        let arena = Buffer::new(context, plan.total_bytes)?;
        let slots = Slots::resolve(&plan, arena.pointer())?;
        arena.upload_at(slots.row_ids_offset, &row_id_bytes(max_rows)?)?;
        let rope = Rope::new(context, config)?;
        let kernels = Functions::new(module)?;
        let stream = Stream::new(context)?;
        // Construction uploads use the legacy stream; finish them before any
        // work on the nonblocking stream can read the tables.
        context.synchronize()?;
        Ok(Self {
            context,
            _weights: weights,
            kernels,
            attention_profile,
            split_attention,
            stream,
            arena,
            rope,
            slots,
            bound,
            shapes,
            max_rows,
            arena_bytes: plan.total_bytes,
            peak_live_bytes: plan.peak_live_bytes,
        })
    }

    pub(super) fn report(&self) -> Value {
        json!({
            "max_rows": self.max_rows,
            "weight_representations": self.bound.representation_report(),
            "arena_bytes": self.arena_bytes,
            "arena_peak_live_bytes": self.peak_live_bytes,
            "attention_profile": self.attention_profile.name(),
            "attention_workspace_bytes": self.split_attention.as_ref().map_or(0, SplitAttention::workspace_bytes),
            "split_attention": self.split_attention.as_ref().map(SplitAttention::report),
            "rope_table_bytes": self.rope.buffer.len(),
            "rope_positions": self.rope.positions,
            "kernels": functions::KERNEL_NAMES,
            "per_forward_host_transfers": [
                "token IDs: stream-ordered pageable HtoD, 4 bytes per row",
                "selection: 16-byte DtoH after one stream synchronize",
                "logits: optional diagnostic DtoH of vocabulary*2 bytes",
            ],
        })
    }

    /// Run one forward of `tokens` against `session`.
    ///
    /// Callers must ensure no legacy-stream work that writes `session` state is
    /// still pending (the legacy path and `Session::new` finish with a context
    /// synchronize before this is called). A failure leaves the cursor poisoned.
    pub(super) fn forward(
        &self,
        tokens: &[u32],
        session: &mut Session<'_>,
        return_logits: bool,
    ) -> Result<StreamOutput> {
        let rows = tokens.len();
        ensure!(
            (1..=self.max_rows).contains(&rows),
            "stream forward rows must be in 1..={}",
            self.max_rows
        );
        ensure!(
            tokens
                .iter()
                .all(|&id| (id as usize) < self.shapes.vocabulary),
            "decoder token is outside vocabulary"
        );
        ensure!(
            session.state.belongs_to(self.context),
            "stream forward session belongs to another context"
        );
        ensure!(
            session
                .cursor
                .past()
                .checked_add(rows)
                .is_some_and(|end| end <= self.rope.positions),
            "stream forward positions exceed the precomputed RoPE capacity"
        );
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let transaction = session.cursor.begin(rows)?;
        let step = Step {
            rows,
            past: transaction.past(),
            capacity: transaction.capacity(),
            cos: self.rope.cos(transaction.past())?,
            sin: self.rope.sin(transaction.past())?,
        };
        if let Some(split) = self.split_attention.as_ref().filter(|_| step.rows <= 8) {
            split.plan(&step)?;
        }
        let active = self.stream.enter()?;
        let enqueued = self.enqueue(&active, &token_bytes, &session.state, &step);
        let synchronized = active.synchronize();
        drop(active);
        match (enqueued, synchronized) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) => return Err(error),
            (Ok(()), Err(error)) => return Err(error.context("synchronize stream forward")),
            (Err(error), Err(sync)) => {
                return Err(error.context(format!("stream synchronize also failed: {sync:#}")));
            }
        }
        let token = self.read_selection()?;
        let logits = return_logits.then(|| self.download_logits()).transpose()?;
        let past = transaction.commit();
        Ok(StreamOutput {
            token,
            past,
            logits,
        })
    }

    fn enqueue(
        &self,
        active: &ActiveStream<'_, 'ctx>,
        tokens: &[u8],
        state: &ResidentState<'_>,
        step: &Step,
    ) -> Result<()> {
        ensure!(
            tokens.len() <= self.max_rows * 4,
            "token upload exceeds planned slot"
        );
        // SAFETY: The token slot lies in the owned arena with max_rows*4 bytes; the
        // previous forward synchronized this stream and no other stream uses the arena.
        unsafe { active.copy_from_host(self.slots.tokens, tokens)? };
        // SAFETY: Every slot address was resolved from an arena plan for max_rows >=
        // step.rows inside `self.arena`; weight addresses come from verified bindings
        // borrowed for 'w; state addresses are checked against their region extents;
        // `forward` synchronizes this stream before returning or releasing anything.
        let e = unsafe { Enqueue::new(active, &self.kernels) };
        let (slots, shapes) = (&self.slots, &self.shapes);
        layers::entry(&e, &self.bound, slots, shapes, step.rows)?;
        for layer in &self.bound.layers {
            match &layer.block {
                Block::Gdn(weights) => layers::gdn(&e, weights, slots, shapes, state, step.rows)?,
                Block::Attention(weights) => {
                    layers::attention(
                        &e,
                        weights,
                        slots,
                        shapes,
                        state,
                        step,
                        self.split_attention.as_ref(),
                    )?;
                }
            }
            layers::mlp(&e, &layer.mlp, slots, shapes, step.rows)?;
        }
        layers::head(&e, &self.bound, slots, shapes, step.rows)
    }

    /// Validate the device greedy record exactly as `resident_greedy::Selector` does.
    fn read_selection(&self) -> Result<u32> {
        let mut bytes = [0_u8; 16];
        self.arena
            .download_at(self.slots.result_offset, &mut bytes)?;
        let words: [u32; 4] = std::array::from_fn(|index| {
            let mut word = [0; 4];
            word.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            u32::from_le_bytes(word)
        });
        let vocabulary = u32::try_from(self.shapes.vocabulary)?;
        ensure!(words[1] <= 1, "invalid GPU greedy status");
        if words[1] != 0 {
            ensure!(words[2] < vocabulary, "invalid nonfinite GPU position");
            return Err(anyhow!("nonfinite logit at {}", words[2]));
        }
        ensure!(
            words[0] < vocabulary
                && words[2] == u32::MAX
                && words[3] <= u32::from(u16::MAX)
                && words[3] & 0x7f80 != 0x7f80,
            "invalid finite GPU selection"
        );
        Ok(words[0])
    }

    fn download_logits(&self) -> Result<Vec<u16>> {
        let mut raw = vec![0_u8; self.shapes.vocabulary * 2];
        self.arena.download_at(self.slots.logits_offset, &mut raw)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect())
    }
}

impl<'ctx> Rope<'ctx> {
    /// Build `[cos | sin]` for positions `0..capacity` with the legacy host tables.
    fn new(context: &'ctx Context, config: &DecoderConfig) -> Result<Self> {
        let shape = &config.attention_shape;
        let rope = TextRope::new(shape.rotary_dim, shape.rope_theta)?;
        let positions = config.capacity;
        ensure!(positions > 0, "stream forward capacity must be positive");
        let half = shape.rotary_dim / 2;
        let table_bytes = positions
            .checked_mul(half * 2)
            .context("RoPE table extent overflows usize")?;
        let mut cos = Vec::with_capacity(table_bytes);
        let mut sin = Vec::with_capacity(table_bytes);
        let mut past = 0;
        while past < positions {
            let rows = (positions - past).min(ROPE_CHUNK_ROWS);
            let tables = rope.tables(past, rows)?;
            cos.extend(tables.cos.iter().flat_map(|value| value.to_le_bytes()));
            sin.extend(tables.sin.iter().flat_map(|value| value.to_le_bytes()));
            past += rows;
        }
        ensure!(
            cos.len() == table_bytes && sin.len() == table_bytes,
            "RoPE table extent mismatch"
        );
        let buffer = Buffer::new(context, table_bytes * 2)?;
        buffer.upload_at(0, &cos)?;
        buffer.upload_at(table_bytes, &sin)?;
        Ok(Self {
            buffer,
            positions,
            half,
        })
    }

    fn cos(&self, past: usize) -> Result<u64> {
        self.address(0, past)
    }

    fn sin(&self, past: usize) -> Result<u64> {
        self.address(self.positions * self.half * 2, past)
    }

    fn address(&self, table: usize, past: usize) -> Result<u64> {
        ensure!(past < self.positions, "RoPE position is out of range");
        let offset = u64::try_from(table + past * self.half * 2)?;
        self.buffer
            .pointer()
            .checked_add(offset)
            .context("RoPE address overflow")
    }
}

fn row_id_bytes(rows: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(rows * 4);
    for row in 0..rows {
        bytes.extend_from_slice(&u32::try_from(row)?.to_le_bytes());
    }
    Ok(bytes)
}

/// Projection restrictions are unchanged; attention admits an explicit decode-only opt-in.
fn ensure_supported_profiles() -> Result<attention_profile::Profile> {
    let fp8 = fp8_profile::current()?;
    ensure!(
        fp8 == fp8_profile::Profile::Exact,
        "stream execution requires MESH_SPECIALIZE_FP8_PROFILE=exact (found {})",
        fp8.name()
    );
    let nvfp4 = nvfp4_profile::current()?;
    ensure!(
        nvfp4 == nvfp4_profile::Profile::Baseline,
        "stream execution requires MESH_SPECIALIZE_NVFP4_PROFILE=baseline (found {})",
        nvfp4.name()
    );
    let attention = attention_profile::current()?;
    ensure!(
        attention.supports_stream(),
        "stream execution requires MESH_SPECIALIZE_ATTENTION_PROFILE=exact or split-decode (found {})",
        attention.name()
    );
    ensure!(
        !super::model_workspace::enabled()?,
        "stream execution requires MESH_SPECIALIZE_MLP_WORKSPACE=off"
    );
    ensure!(
        super::resident_fp8_splitk::configured_splits()?.is_none(),
        "stream execution requires MESH_SPECIALIZE_FP8_SPLIT_K=off"
    );
    ensure!(
        !super::nvfp4_projection_audit::enabled()?,
        "stream execution does not support MESH_SPECIALIZE_NVFP4_AUDIT"
    );
    Ok(attention)
}

#[cfg(test)]
mod tests {
    use super::row_id_bytes;

    #[test]
    fn row_ids_are_consecutive_little_endian_words() {
        assert_eq!(
            row_id_bytes(3).unwrap(),
            [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
        );
    }
}
