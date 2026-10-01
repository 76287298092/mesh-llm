//! Resident BF16 multi-token prediction path with an independent KV arena.

use super::{
    driver::{Buffer, Context, Module},
    resident_attention::{Layer, Step},
    resident_embedding::Embedding,
    resident_norm::Norm,
    resident_projection::{Projection, Quantization},
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    engine::{layout::Layout, sampling, session::Cursor},
    kernels::DecoderConfig,
};
use anyhow::{Context as _, Result, ensure};

const MAX_ROWS: usize = 2_048;
const BF16_BYTES: usize = 2;
const NORM_EPSILON: f32 = 1e-6;

pub(super) struct Model<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    target_norm: Norm<'w, 'ctx>,
    embedding: Embedding<'w, 'ctx>,
    hidden_norm: Norm<'w, 'ctx>,
    fc: Projection<'w, 'ctx>,
    attention: Layer<'w, 'ctx>,
    output_norm: Norm<'w, 'ctx>,
    head: Projection<'w, 'ctx>,
    state_layout: Layout,
    hidden: usize,
    vocabulary: usize,
    capacity: usize,
}

pub(super) struct Session<'ctx> {
    pub(super) state: ResidentState<'ctx>,
    pub(super) cursor: Cursor,
}

pub(super) struct Output<'ctx> {
    pub(super) hidden: Buffer<'ctx>,
    pub(super) logits: Vec<u16>,
    pub(super) token: u32,
    pub(super) past: usize,
}

impl<'ctx> Session<'ctx> {
    pub(super) fn new(ctx: &'ctx Context, config: &DecoderConfig) -> Result<Self> {
        validate_model_config(config)?;
        let layout = mtp_state_layout(config)?;
        let cursor = Cursor::new(config.capacity)?;
        Ok(Self {
            state: ResidentState::new(ctx, &layout)?,
            cursor,
        })
    }

    pub(super) fn fork<'a>(&self, ctx: &'a Context) -> Result<Session<'a>> {
        let cursor = self.cursor.fork()?;
        let state = self.state.fork(ctx)?;
        Ok(Session { state, cursor })
    }
}

impl<'w, 'ctx> Model<'w, 'ctx> {
    pub(super) fn new(owner: &'w ResidentWeights<'ctx>, config: &DecoderConfig) -> Result<Self> {
        validate_model_config(config)?;
        let state_layout = mtp_state_layout(config)?;
        let hidden = config.hidden;
        let vocabulary = config.vocabulary;
        let attention = Layer::new(
            owner,
            "tensors/mtp.layers.0",
            "mtp",
            &config.attention_shape,
            Quantization::Bf16,
        )?;

        Ok(Self {
            owner,
            target_norm: Norm::new(owner, &config.final_norm, hidden, NORM_EPSILON)?,
            embedding: Embedding::new(
                owner,
                &config.embedding_table,
                "tensors/mtp.pre_fc_norm_embedding.weight",
                [vocabulary, hidden],
                NORM_EPSILON,
            )?,
            hidden_norm: Norm::new(
                owner,
                "tensors/mtp.pre_fc_norm_hidden.weight",
                hidden,
                NORM_EPSILON,
            )?,
            fc: Projection::new(
                owner,
                "tensors/mtp.fc",
                hidden
                    .checked_mul(2)
                    .context("MTP FC input width overflows usize")?,
                hidden,
                Quantization::Bf16,
            )?,
            attention,
            output_norm: Norm::new(owner, "tensors/mtp.norm.weight", hidden, NORM_EPSILON)?,
            head: Projection::new(
                owner,
                &config.head_prefix,
                hidden,
                vocabulary,
                Quantization::Fp8,
            )?,
            state_layout,
            hidden,
            vocabulary,
            capacity: config.capacity,
        })
    }

    /// Apply the target model's final norm to raw target hidden rows for MTP input.
    pub(super) fn target_hidden<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        raw: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'a>> {
        validate_context(self.owner, ctx, module)?;
        validate_matrix(ctx, raw, rows, self.hidden, "target hidden")?;
        self.target_norm.run(ctx, module, raw, rows)
    }

    /// Run one MTP block and commit its KV prefix only after logits are valid.
    pub(super) fn forward<'a>(
        &self,
        ctx: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
        target_or_draft_hidden: &Buffer<'_>,
        session: &mut Session<'_>,
    ) -> Result<Output<'a>> {
        let rows = validate_forward(self, ctx, module, tokens, target_or_draft_hidden, session)?;
        let transaction = session.cursor.begin(rows)?;
        let normalized_embedding = self.embedding.run(ctx, module, tokens)?;
        let normalized_hidden = self
            .hidden_norm
            .run(ctx, module, target_or_draft_hidden, rows)?;
        let concatenated = concatenate_inputs(
            ctx,
            &normalized_embedding.normalized,
            &normalized_hidden,
            rows,
            self.hidden,
        )?;
        let fc = self.fc.run(ctx, module, &concatenated, rows)?;
        let attention_hidden = self.attention.forward(
            ctx,
            module,
            &fc.values,
            &mut session.state,
            &Step {
                rows,
                past: transaction.past(),
                capacity: transaction.capacity(),
                decode: false,
            },
        )?;
        let hidden = self.output_norm.run(ctx, module, &attention_hidden, rows)?;
        let last_hidden = last_row(ctx, &hidden, rows, self.hidden)?;
        let projected = self.head.run(ctx, module, &last_hidden, 1)?;
        let logits = download_last_logits(&projected.values, 1, self.vocabulary)?;
        let token = sampling::greedy(&logits)?;
        let past = transaction.commit();

        Ok(Output {
            hidden,
            logits,
            token,
            past,
        })
    }
}

fn validate_model_config(config: &DecoderConfig) -> Result<()> {
    ensure!(
        config.hidden == config.attention_shape.hidden,
        "MTP hidden widths disagree"
    );
    ensure!(
        (1..=262_144).contains(&config.vocabulary),
        "MTP vocabulary is out of range"
    );
    ensure!(
        (1..=262_144).contains(&config.capacity),
        "MTP context capacity is out of range"
    );
    ensure!(
        config.hidden.checked_mul(2).is_some(),
        "MTP FC input width overflows usize"
    );
    Ok(())
}

pub(super) fn mtp_state_layout(config: &DecoderConfig) -> Result<Layout> {
    let shape = &config.attention_shape;
    let elements = config
        .capacity
        .checked_mul(shape.kv_heads)
        .and_then(|value| value.checked_mul(shape.head_width))
        .context("MTP KV element count overflows usize")?;
    let bytes = elements
        .checked_mul(BF16_BYTES)
        .context("MTP KV byte extent overflows usize")?;
    let bytes = u64::try_from(bytes).context("MTP KV byte extent does not fit u64")?;
    Layout::new([
        ("mtp.attention.k".to_owned(), bytes),
        ("mtp.attention.v".to_owned(), bytes),
    ])
}

fn validate_context(owner: &ResidentWeights<'_>, ctx: &Context, module: &Module<'_>) -> Result<()> {
    ensure!(
        owner.belongs_to(ctx),
        "MTP weights belong to another CUDA context"
    );
    ensure!(
        module.belongs_to(ctx),
        "MTP module belongs to another CUDA context"
    );
    Ok(())
}

fn validate_forward(
    model: &Model<'_, '_>,
    ctx: &Context,
    module: &Module<'_>,
    tokens: &[u32],
    hidden: &Buffer<'_>,
    session: &Session<'_>,
) -> Result<usize> {
    validate_context(model.owner, ctx, module)?;
    ensure!(
        (1..=MAX_ROWS).contains(&tokens.len()),
        "MTP row count is out of range"
    );
    let vocabulary = u64::try_from(model.vocabulary).context("MTP vocabulary does not fit u64")?;
    ensure!(
        tokens.iter().all(|&token| u64::from(token) < vocabulary),
        "MTP token is out of vocabulary range"
    );
    validate_matrix(ctx, hidden, tokens.len(), model.hidden, "MTP hidden")?;
    ensure!(
        session.state.belongs_to(ctx),
        "MTP session state belongs to another CUDA context"
    );
    ensure!(
        session.state.layout() == &model.state_layout,
        "MTP session state layout differs from the model"
    );
    ensure!(
        session.cursor.capacity() == model.capacity,
        "MTP session capacity differs from the model"
    );
    ensure!(
        !session.cursor.is_poisoned(),
        "MTP session cursor is poisoned"
    );
    Ok(tokens.len())
}

fn validate_matrix(
    ctx: &Context,
    input: &Buffer<'_>,
    rows: usize,
    width: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        input.belongs_to(ctx),
        "{label} buffer belongs to another CUDA context"
    );
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "{label} row count is out of range"
    );
    let expected = rows
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("{label} BF16 extent overflows usize")?;
    ensure!(input.len() == expected, "{label} BF16 extent mismatch");
    Ok(())
}

fn concatenate_inputs<'a>(
    ctx: &'a Context,
    embedding: &Buffer<'_>,
    hidden: &Buffer<'_>,
    rows: usize,
    width: usize,
) -> Result<Buffer<'a>> {
    let hidden_row_bytes = width
        .checked_mul(BF16_BYTES)
        .context("MTP input row byte extent overflows usize")?;
    let combined_row_bytes = hidden_row_bytes
        .checked_mul(2)
        .context("MTP concatenated row byte extent overflows usize")?;
    let output_bytes = rows
        .checked_mul(combined_row_bytes)
        .context("MTP concatenated extent overflows usize")?;
    let input_bytes = rows
        .checked_mul(hidden_row_bytes)
        .context("MTP input extent overflows usize")?;
    ensure!(
        embedding.belongs_to(ctx) && hidden.belongs_to(ctx),
        "MTP concatenation inputs belong to another CUDA context"
    );
    ensure!(
        embedding.len() == input_bytes && hidden.len() == input_bytes,
        "MTP concatenation input extent mismatch"
    );
    let output = Buffer::new(ctx, output_bytes)?;
    for row in 0..rows {
        let source_offset = row
            .checked_mul(hidden_row_bytes)
            .context("MTP row source offset overflows usize")?;
        let destination_offset = row
            .checked_mul(combined_row_bytes)
            .context("MTP row destination offset overflows usize")?;
        output.copy_from_at(
            destination_offset,
            embedding,
            source_offset,
            hidden_row_bytes,
        )?;
        output.copy_from_at(
            destination_offset
                .checked_add(hidden_row_bytes)
                .context("MTP hidden row offset overflows usize")?,
            hidden,
            source_offset,
            hidden_row_bytes,
        )?;
    }
    Ok(output)
}

fn download_last_logits(output: &Buffer<'_>, rows: usize, vocabulary: usize) -> Result<Vec<u16>> {
    let row_bytes = vocabulary
        .checked_mul(BF16_BYTES)
        .context("MTP logits row byte extent overflows usize")?;
    let output_bytes = rows
        .checked_mul(row_bytes)
        .context("MTP logits byte extent overflows usize")?;
    ensure!(output.len() == output_bytes, "MTP logits extent mismatch");
    let offset = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(row_bytes))
        .context("MTP last-row logits offset overflows usize")?;
    let mut bytes = vec![0_u8; row_bytes];
    output.download_at(offset, &mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| u16::from_le_bytes(*word))
        .collect())
}

fn last_row<'a>(
    ctx: &'a Context,
    input: &Buffer<'_>,
    rows: usize,
    width: usize,
) -> Result<Buffer<'a>> {
    ensure!(
        input.belongs_to(ctx),
        "MTP hidden belongs to another CUDA context"
    );
    let row_bytes = width
        .checked_mul(BF16_BYTES)
        .context("MTP hidden row byte extent overflows usize")?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .context("MTP hidden extent overflows usize")?;
    ensure!(input.len() == expected_bytes, "MTP hidden extent mismatch");
    let source_offset = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(row_bytes))
        .context("MTP last hidden row offset overflows usize")?;
    let output = Buffer::new(ctx, row_bytes)?;
    output.copy_from_at(0, input, source_offset, row_bytes)?;
    Ok(output)
}
