use super::{
    BF16_BYTES, HEAD_WIDTH, HIDDEN, INTERMEDIATE, KV_HEADS, MAX_CONTEXT, MAX_ROWS, Model,
    PROPOSAL_ROWS, QUERY_HEADS, ROTARY_DIM, Session, VOCABULARY,
};
use crate::kernels::cuda::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, ensure};

pub(super) fn validate_model_config(config: &crate::kernels::DecoderConfig) -> Result<()> {
    let shape = &config.attention_shape;
    ensure!(
        config.hidden == HIDDEN && config.vocabulary == VOCABULARY,
        "native MTP hidden or vocabulary differs from the checked model"
    );
    ensure!(
        shape.hidden == HIDDEN
            && shape.intermediate == INTERMEDIATE
            && shape.query_heads == QUERY_HEADS
            && shape.kv_heads == KV_HEADS
            && shape.head_width == HEAD_WIDTH
            && shape.rotary_dim == ROTARY_DIM,
        "native MTP requires the checked Qwen 3.5 27B geometry"
    );
    ensure!(
        (1..=MAX_CONTEXT).contains(&config.capacity),
        "native MTP context capacity is out of range"
    );
    Ok(())
}

pub(super) fn validate_forward(
    model: &Model<'_, '_>,
    context: &Context,
    module: &Module<'_>,
    tokens: &[u32],
    target_hidden: &Buffer<'_>,
    session: &Session<'_>,
) -> Result<usize> {
    ensure!(
        model.target_owner.belongs_to(context)
            && model.native.belongs_to(context)
            && module.belongs_to(context),
        "native MTP weights or module belong to another CUDA context"
    );
    let rows = tokens.len();
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "native MTP row count must be in 1..=5"
    );
    let vocabulary = u64::try_from(VOCABULARY).context("native MTP vocabulary does not fit u64")?;
    ensure!(
        tokens.iter().all(|&token| u64::from(token) < vocabulary),
        "native MTP token is outside the target vocabulary"
    );
    validate_matrix(
        context,
        target_hidden,
        rows,
        HIDDEN,
        "native MTP target hidden",
    )?;
    ensure!(
        session.state.belongs_to(context)
            && session.state.layout() == &model.layout
            && session.cursor.capacity() == model.capacity,
        "native MTP session ownership or state layout mismatch"
    );
    validate_session_capacity(&session.cursor, rows)?;
    validate_proposal_parent(model)?;
    ensure!(
        model.native.views().proposal_tokens.len() == PROPOSAL_ROWS,
        "native MTP proposal token map extent mismatch"
    );
    Ok(rows)
}

pub(super) fn validate_session_capacity(
    cursor: &crate::engine::session::Cursor,
    rows: usize,
) -> Result<()> {
    let end = cursor
        .past()
        .checked_add(rows)
        .context("native MTP cursor position overflows")?;
    ensure!(
        end <= cursor.capacity(),
        "native MTP request exceeds session capacity"
    );
    ensure!(
        !cursor.is_poisoned(),
        "native MTP session cursor is poisoned"
    );
    Ok(())
}

pub(super) fn validate_proposal_parent(model: &Model<'_, '_>) -> Result<()> {
    crate::kernels::cuda::native_mtp_q4_operator::resident::validate_head(
        &model.native.views().proposal_head,
        model
            .native
            .layout()
            .region(&model.native.views().proposal_head.object_id)?
            .length,
    )
}

pub(super) fn validate_matrix(
    context: &Context,
    input: &Buffer<'_>,
    rows: usize,
    width: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        input.belongs_to(context),
        "{label} belongs to another CUDA context"
    );
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "{label} row count is out of range"
    );
    let bytes = rows
        .checked_mul(width)
        .and_then(|count| count.checked_mul(BF16_BYTES))
        .context("native MTP matrix extent overflows")?;
    ensure!(input.len() == bytes, "{label} extent mismatch");
    Ok(())
}
