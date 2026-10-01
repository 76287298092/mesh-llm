use super::Request;
use crate::kernels::cuda::{driver::Buffer, resident_model::{Session, LogitsSelection}, resident_mtp};
use anyhow::{Context as _, Result, ensure};

pub(super) struct Oracle<'ctx> {
    pub target: Session<'ctx>,
    pub draft: resident_mtp::Session<'ctx>,
    pub pending: u32,
    pub tokens: Vec<u32>,
}

pub(super) fn prepare<'ctx>(request: &Request<'_, '_, '_, 'ctx>) -> Result<Oracle<'ctx>> {
    let mut base = Session::new(request.context, request.config)?;
    let prefill = request.target.forward_detailed(request.context, request.module,
        request.prompt, &mut base, LogitsSelection::Last, None)?;
    let pending = *prefill.tokens.first().context("prefill selected no token")?;
    ensure!(prefill.tokens.len() == 1, "prefill selection extent differs");
    let mut draft = resident_mtp::Session::new(request.context, request.config)?;
    let row_bytes = request.config.hidden.checked_mul(2).context("hidden row overflow")?;
    let bytes = row_bytes.checked_mul(request.prompt.len()).context("prefill hidden overflow")?;
    ensure!(prefill.hidden.len() == bytes, "prefill raw hidden extent differs");
    for (row, token) in request.prompt.iter().skip(1).copied()
        .chain(std::iter::once(pending)).enumerate() {
        let raw = Buffer::new(request.context, row_bytes)?;
        raw.copy_from_at(0, &prefill.hidden, row.checked_mul(row_bytes)
            .context("prefill row offset overflow")?, row_bytes)?;
        teacher(request, &mut draft, (token, &raw))?;
    }
    Ok(Oracle { target: base.fork(request.context)?, draft, pending, tokens: vec![pending] })
}

pub(super) fn teacher(request: &Request<'_, '_, '_, '_>,
    draft: &mut resident_mtp::Session<'_>, input: (u32, &Buffer<'_>)) -> Result<()> {
    let normalized = request.draft.target_hidden(request.context, request.module, input.1, 1)?;
    request.draft.forward(request.context, request.module, &[input.0], &normalized, draft)?;
    Ok(())
}

pub(super) fn advance(request: &Request<'_, '_, '_, '_>, oracle: &mut Oracle<'_>) -> Result<u32> {
    let output = request.target.forward_detailed_decode(request.context, request.module,
        oracle.pending, &mut oracle.target, None)?;
    let next = *output.tokens.first().context("ordinary decode selected no token")?;
    ensure!(output.tokens.len() == 1, "ordinary decode selection extent differs");
    teacher(request, &mut oracle.draft, (next, &output.hidden))?;
    oracle.pending = next;
    oracle.tokens.push(next);
    Ok(next)
}

pub(super) fn proposals(request: &Request<'_, '_, '_, '_>, base: &Oracle<'_>) -> Result<Vec<u32>> {
    let mut target = base.target.fork(request.context)?;
    let mut pending = base.pending;
    let mut tokens = Vec::with_capacity(4);
    for _ in 0..4 {
        let output = request.target.forward_detailed_decode(request.context, request.module,
            pending, &mut target, None)?;
        ensure!(output.tokens.len() == 1, "fixture ordinary decode selection extent differs");
        pending = *output.tokens.first().context("fixture selected no token")?;
        tokens.push(pending);
    }
    Ok(tokens)
}
